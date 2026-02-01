import logging
import re
import sqlite3
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests

import config

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# API Helpers
# ------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def fetch_json(endpoint_url: str) -> Dict[str, Any]:
    """
    Fetch JSON data from an API endpoint with caching and retry logic.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(3):
        try:
            r = requests.get(endpoint_url, headers=headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"⚠️ API attempt {attempt + 1} failed for {endpoint_url}: {e}")
            time.sleep(1)

    logger.error(f"❌ Failed to fetch {endpoint_url} after 3 attempts.")
    return {}


# ------------------------------------------------------------------------------
# Data Fetching
# ------------------------------------------------------------------------------

def get_current_gw() -> Optional[int]:
    """
    Return the current Gameweek (GW) number from the FPL API.
    """
    data = fetch_json(f"{config.FPL_API_URL}bootstrap-static/")
    for gw in data.get('events', []):
        if gw.get('is_current'):
            logger.info(f"📅 Current Gameweek identified: {gw['id']}")
            return gw['id']

    # Fallback
    for gw in data.get('events', []):
        if gw.get('is_next'):
            logger.info(f"📅 Next Gameweek identified: {gw['id']}")
            return gw['id']

    logger.warning("⚠️ No current or next gameweek found.")
    return None


def get_next_fixtures(conn: sqlite3.Connection, current_gw: int) -> Dict[int, str]:
    """
    Creates a lookup map for the current gameweek's fixtures.
    Returns: {team_id: "vs Arsenal (H) [Diff: 4]"}
    """
    # Get all team names first
    teams_df = pd.read_sql_query("SELECT team_id, name FROM teams", conn)
    team_map = teams_df.set_index("team_id")["name"].to_dict()

    # Get fixtures for this gameweek
    query = """
            SELECT home_team_id, away_team_id, home_difficulty, away_difficulty
            FROM fixtures
            WHERE gameweek = ? \
            """
    fixtures_df = pd.read_sql_query(query, conn, params=(current_gw,))

    fixture_map = {}
    for _, row in fixtures_df.iterrows():
        h_id, a_id = row['home_team_id'], row['away_team_id']
        h_name, a_name = team_map.get(h_id, "Unknown"), team_map.get(a_id, "Unknown")

        # Map for Home Team
        fixture_map[h_id] = f"vs {a_name} (H) [Diff: {row['home_difficulty']}]"
        # Map for Away Team
        fixture_map[a_id] = f"vs {h_name} (A) [Diff: {row['away_difficulty']}]"

    return fixture_map


def get_user_team(team_id: int, gw: int) -> Dict[str, Any]:
    """
    Fetch user's FPL team info AND explicitly attach their next fixture.
    """
    team_info = {'player_data': [], 'rank_string': "Rank info unavailable."}

    if not team_id:
        return team_info

    # 1. Fetch Rank & History
    history_data = fetch_json(f"{config.FPL_API_URL}entry/{team_id}/history/")
    if "current" in history_data and history_data["current"]:
        last_gw = history_data["current"][-1]
        team_info['rank_string'] = (
            f"Overall Rank: {last_gw.get('overall_rank', 'Unranked'):,}. "
            f"Total Points: {last_gw.get('total_points', 0)}. "
            f"Team Value: £{last_gw.get('value', 0) / 10:.1f}m. "
            f"Bank: £{last_gw.get('bank', 0) / 10:.1f}m."
        )

    # 2. Fetch Current Team Picks
    if gw:
        picks_url = f"{config.FPL_API_URL}entry/{team_id}/event/{gw}/picks/"
        picks_data = fetch_json(picks_url)

        player_ids = [p['element'] for p in picks_data.get('picks', [])]

        if player_ids:
            try:
                with sqlite3.connect(config.DB_NAME) as conn:
                    # 1. Get the fixture map for the upcoming week
                    fixture_map = get_next_fixtures(conn, gw)

                    # 2. Get the players
                    placeholders = ",".join("?" * len(player_ids))
                    query = f"""
                        SELECT player_id, name, position, team_id, price, form, xG, xA 
                        FROM players 
                        WHERE player_id IN ({placeholders})
                    """

                    players_df = pd.read_sql_query(query, conn, params=player_ids)

                    # 3. Merging: Apply the fixture string directly to the dataframe
                    # This prevents hallucinations by hardcoding the opponent
                    players_df['next_match'] = players_df['team_id'].map(fixture_map).fillna("No Match")

                    team_info['player_data'] = players_df.to_dict('records')
                    logger.info(f"✅ Fetched details for {len(players_df)} players with fixtures.")
            except Exception as e:
                logger.error(f"❌ Database error fetching team picks: {e}")

    return team_info


# ------------------------------------------------------------------------------
# RAG Retrieval
# ------------------------------------------------------------------------------

def query_faiss(
        query_text: str,
        n_results: int,
        faiss_index,
        documents,
        embedding_model,
        metadatas: List[Dict],
        doc_type: str = "all"
) -> List[str]:
    """
    Query FAISS index for the most relevant documents, with metadata filtering.
    """
    try:
        query_vec = embedding_model.encode([query_text], normalize_embeddings=True)

        # Retrieve more results initially to allow for filtering
        k = n_results * 4
        distances, idxs = faiss_index.search(query_vec.astype('float32'), k)

        results = []
        seen_indices = set()

        for i in idxs[0]:
            # Ensure index is valid and not a duplicate
            if 0 <= i < len(documents) and i not in seen_indices:
                # Check metadata for doc_type if a filter is applied
                if doc_type == "all" or (i < len(metadatas) and metadatas[i].get("doc_type") == doc_type):
                    results.append(documents[i])
                    seen_indices.add(i)

            # Stop when we have enough results
            if len(results) >= n_results:
                break

        logger.info(f"🔍 FAISS retrieved {len(results)} docs (type: {doc_type}) for: '{query_text[:30]}...'")
        return results
    except Exception as e:
        logger.error(f"❌ FAISS query error: {e}")
        return []


def get_context_for_players(
        player_names: List[str],
        faiss_index,
        documents,
        embedding_model,
        metadatas: List[Dict]
) -> List[str]:
    """
    Specific retrieval strategy for player analysis.
    """
    docs = []
    unique_names = list(set(player_names))

    for name in unique_names:
        query = f"Player Analysis: {name} [Current Status] form fixtures"
        # We ONLY want player status docs here
        player_docs = query_faiss(query, 5, faiss_index, documents, embedding_model, metadatas,
                                  doc_type="current_status")
        docs.extend(player_docs)

    unique_docs = list(dict.fromkeys(docs))
    return unique_docs


# ------------------------------------------------------------------------------
# Analysis Logic
# ------------------------------------------------------------------------------

def find_transfer_targets(conn: sqlite3.Connection, current_gw: int, exclude_ids: List[int] = None,
                          num_gameweeks: int = 3, top_n: int = 15) -> List[str]:
    """
    Identify transfer targets, EXCLUDING players the user already owns.
    """
    try:
        # 1. Calculate Fixture Difficulty (Same as before)
        fixtures_df = pd.read_sql_query("SELECT * FROM fixtures", conn)
        future_mask = (fixtures_df["gameweek"] >= current_gw) & (fixtures_df["gameweek"] < current_gw + num_gameweeks)
        future_games = fixtures_df[future_mask].copy()

        if future_games.empty:
            return []

        team_difficulty = {}
        home_diff = future_games.groupby("home_team_id")["home_difficulty"].mean()
        away_diff = future_games.groupby("away_team_id")["away_difficulty"].mean()
        all_teams = set(home_diff.index).union(set(away_diff.index))
        for team_id in all_teams:
            d_h = home_diff.get(team_id, np.nan)
            d_a = away_diff.get(team_id, np.nan)
            team_difficulty[team_id] = np.nanmean([d_h, d_a])

        # 2. Select Potential Targets (Added player_id and position)
        players_df = pd.read_sql_query(
            "SELECT player_id, name, position, team_id, form, price, transfers_in FROM players WHERE status='a' AND price < 13.0",
            conn
        )

        # 3. EXCLUSION LOGIC (The Fix)
        if exclude_ids:
            players_df = players_df[~players_df['player_id'].isin(exclude_ids)]

        # 4. Scoring Logic
        players_df['next_difficulty'] = players_df['team_id'].map(team_difficulty).fillna(5)
        players_df["form"] = pd.to_numeric(players_df["form"], errors="coerce").fillna(0)

        # Score = Form + Fixtures + Market Trend
        players_df["score"] = (players_df["form"] * 1.5) + ((5 - players_df["next_difficulty"]) * 2.0)

        # 5. Format Output with Position (So LLM doesn't make illegal swaps)
        top_players = players_df.nlargest(top_n, "score")

        results = []
        for _, p in top_players.iterrows():
            # "Semenyo (MID)"
            results.append(f"{p['name']} ({p['position']})")

        logger.info(f"🎯 Top targets (excluding owned): {results[:5]}")
        return results

    except Exception as e:
        logger.error(f"❌ Error finding transfer targets: {e}")
        return []


def get_chip_strategy_advice(user_team_players: List[Dict], current_gw: int) -> str:
    """
    Returns advice string based on gameweek phase.
    """
    if current_gw < 30:
        return "Strategy: Hold your chips (Bench Boost/Triple Captain) for Double Gameweeks later in the season."
    else:
        return "Strategy: We are in the end-game. Look for Double Gameweeks to deploy Bench Boost or Triple Captain immediately."


# ------------------------------------------------------------------------------
# Utility Helpers
# ------------------------------------------------------------------------------

def compress_prompt(prompt: str) -> str:
    """
    Clean and compress long text prompts to save tokens.
    """
    prompt = re.sub(r'\n\s*\n+', '\n', prompt)
    prompt = re.sub(r'[ \t]+', ' ', prompt)
    return prompt.strip()
