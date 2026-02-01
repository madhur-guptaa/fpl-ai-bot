import logging
import pickle
import sqlite3
from typing import List, Dict, Tuple

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

import config  # Import centralized configuration

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fpl_rag.log", mode="w"),  # Overwrite log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------

def get_current_gw(fixtures_df: pd.DataFrame) -> int:
    """Return the current or next gameweek based on fixture times."""
    now = pd.to_datetime("now", utc=True)
    future_fixtures = fixtures_df[pd.to_datetime(fixtures_df["kickoff_time"]) > now]

    if not future_fixtures.empty:
        gw = int(future_fixtures["gameweek"].min())
    else:
        gw = int(fixtures_df["gameweek"].max())  # Season finished

    logger.info(f"📅 Determined Current Gameweek: {gw}")
    return gw


def load_all_data_from_db() -> Tuple[pd.DataFrame, ...]:
    """Load all required FPL data tables from SQLite."""
    try:
        with sqlite3.connect(config.DB_NAME) as conn:
            logger.info(f"📥 Loading data from {config.DB_NAME}...")
            players = pd.read_sql_query("SELECT * FROM players", conn)
            teams = pd.read_sql_query("SELECT * FROM teams", conn)
            fixtures = pd.read_sql_query("SELECT * FROM fixtures", conn)
            season_summary = pd.read_sql_query(f"SELECT * FROM {config.DESTINATION_TABLE}", conn)
            historical_gw = pd.read_sql_query(f"SELECT * FROM {config.SOURCE_TABLE}", conn)
            player_gw_stats = pd.read_sql_query("SELECT * FROM player_gameweek_stats", conn)

        logger.info("✅ All tables loaded.")
        return players, teams, fixtures, season_summary, historical_gw, player_gw_stats
    except Exception as e:
        logger.exception(f"❌ Database load failed: {e}")
        return (None,) * 6


# ------------------------------------------------------------------------------
# Document Creation (The "Knowledge" Layer)
# ------------------------------------------------------------------------------

def create_player_status_documents(players_df, teams_df, fixtures_df, team_records, current_gw) -> Tuple[
    List[str], List[Dict]]:
    """
    Creates the 'Main' document for every player.
    Includes rich context like next opponent, form, and xG/xA.
    """
    documents, metadatas = [], []
    logger.info("📝 Creating Player Status documents...")

    # Pre-calculate next fixtures for speed
    next_fixtures = fixtures_df[fixtures_df["gameweek"] == current_gw]

    # Create a quick lookup for team names
    team_map = teams_df.set_index('team_id')['name'].to_dict()

    for _, player in players_df.iterrows():
        try:
            team_id = player["team_id"]
            team_name = team_map.get(team_id, "Unknown Team")

            # Find next opponent
            fixture = next_fixtures[
                (next_fixtures["home_team_id"] == team_id) |
                (next_fixtures["away_team_id"] == team_id)
                ]

            if not fixture.empty:
                fix = fixture.iloc[0]
                is_home = fix["home_team_id"] == team_id
                opp_id = fix["away_team_id"] if is_home else fix["home_team_id"]
                opp_name = team_map.get(opp_id, "Unknown")

                # Contextual Difficulty
                difficulty = fix["home_difficulty"] if not is_home else fix["away_difficulty"]
                venue_str = "(Home)" if is_home else "(Away)"

                opponent_info = f"Next Match: vs {opp_name} {venue_str}. Difficulty Rating: {difficulty}/5."
            else:
                opponent_info = "No fixture this gameweek (Blank Gameweek)."
                difficulty = 5  # Default to hard if unknown

            # --- Keyword Injection for RAG ---
            # These tags help the vector search find 'concepts' even if the user doesn't type them.
            tags = []
            if float(player['form']) > 5.0: tags.append("[High Form]")
            if float(player['selected_by_percent']) > 30.0: tags.append("[Highly Owned]")
            if difficulty <= 2: tags.append("[Easy Fixture]")
            if player['price'] < 5.0 and float(player['form']) > 3.0: tags.append("[Budget Gem]")

            tag_str = " ".join(tags)

            doc_text = (
                f"Player Analysis: {player['name']} ({team_name}, {player['position']}). "
                f"Price: £{player['price']:.1f}m. Selected by {player['selected_by_percent']}%. "
                f"Current Form: {player['form']}. Status: {player['status']}. "
                f"Season Stats: {int(player['current_goals'])} goals, {int(player['current_assists'])} assists, {player['total_points']} pts. "
                f"Advanced Stats: xG {player.get('xG', 'N/A')}, xA {player.get('xA', 'N/A')}. "
                f"{opponent_info} "
                f"Team Record: {team_records.get(player['team_id'], 'N/A')}. "
                f"Key Tags: {tag_str}"
            )

            documents.append(doc_text)
            metadatas.append({
                "doc_type": "current_status",
                "player_id": int(player["player_id"]),
                "name": player["name"]
            })

        except Exception as e:
            continue  # Skip bad records silently

    return documents, metadatas


def create_fixture_run_documents(teams_df, fixtures_df, current_gw) -> Tuple[List[str], List[Dict]]:
    """
    Aggregates the next 3 games into a single 'Fixture Run' document.
    This helps the LLM answer "Who has good upcoming fixtures?" without retrieving 3 separate docs.
    """
    documents, metadatas = [], []
    logger.info("🗓️ Creating Fixture Run documents...")

    team_map = teams_df.set_index('team_id')['name'].to_dict()

    for team_id, team_name in team_map.items():
        # Get next 3 fixtures
        upcoming = fixtures_df[
            ((fixtures_df['home_team_id'] == team_id) | (fixtures_df['away_team_id'] == team_id))
            & (fixtures_df['gameweek'] >= current_gw)
            ].sort_values('gameweek').head(3)

        if upcoming.empty:
            continue

        run_desc = []
        total_difficulty = 0

        for _, fix in upcoming.iterrows():
            is_home = fix['home_team_id'] == team_id
            opp_id = fix['away_team_id'] if is_home else fix['home_team_id']
            opp_name = team_map.get(opp_id, "Unknown")
            diff = fix['home_difficulty'] if not is_home else fix['away_difficulty']
            venue = "(H)" if is_home else "(A)"

            run_desc.append(f"GW{fix['gameweek']}: {opp_name} {venue} [Diff: {diff}]")
            total_difficulty += diff

        # Heuristic for the LLM
        verdict = "Excellent" if total_difficulty <= 7 else "Difficult" if total_difficulty >= 11 else "Mixed"

        doc_text = (
            f"Fixture Analysis for {team_name}: "
            f"Upcoming run is considered {verdict}. "
            f"Schedule: {', '.join(run_desc)}."
        )

        documents.append(doc_text)
        metadatas.append({"doc_type": "fixture_run", "team_id": team_id})

    return documents, metadatas


# ------------------------------------------------------------------------------
# Main Builder
# ------------------------------------------------------------------------------

def create_documents(players_df, teams_df, fixtures_df, season_summary_df, historical_gw_df, player_gw_stats_df):
    """Orchestrates the creation of all knowledge base documents."""

    current_gw = get_current_gw(fixtures_df)

    # 1. Enrich Players DF with aggregated stats
    current_season_agg = player_gw_stats_df.groupby('player_id').agg(
        current_goals=('goals_scored', 'sum'),
        current_assists=('assists', 'sum')
    ).reset_index()

    players_df = pd.merge(players_df, current_season_agg, on='player_id', how='left').fillna(0)

    # Map position IDs to names
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_df['position'] = players_df['position'].map(pos_map)

    # 2. Calculate Team Win/Loss Records
    # (Placeholder: In a full app, you'd calculate W-D-L from fixtures_df)
    team_records = {}

    # 3. Generate Documents
    all_docs = []
    all_metas = []

    # A. Player Status (The Core)
    docs, metas = create_player_status_documents(players_df, teams_df, fixtures_df, team_records, current_gw)
    all_docs.extend(docs)
    all_metas.extend(metas)

    # B. Fixture Runs (New!)
    docs, metas = create_fixture_run_documents(teams_df, fixtures_df, current_gw)
    all_docs.extend(docs)
    all_metas.extend(metas)

    # C. Historical Summaries
    for _, row in season_summary_df.iterrows():
        text = f"History: {row['name']} ({row['season']}) - {row['total_points']} pts, {row['goals_scored']} goals."
        all_docs.append(text)
        all_metas.append({'doc_type': 'history', 'player_name': row['name']})

    logger.info(f"📚 Total Documents Prepared: {len(all_docs)}")
    return all_docs, all_metas


def build_vector_store_with_faiss(documents, metadatas):
    """Embeds text and saves to FAISS index."""
    if not documents:
        logger.warning("⚠️ No documents to embed.")
        return

    try:
        logger.info(f"🧠 Loading Model: {config.EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        logger.info(f"📉 Encoding {len(documents)} documents...")
        embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)

        # FAISS Index
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype('float32'))

        # Save Artifacts
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        with open(config.DOCUMENTS_PATH, "wb") as f:
            pickle.dump(documents, f)
        with open(config.METADATA_PATH, "wb") as f:
            pickle.dump(metadatas, f)

        logger.info(f"✅ Knowledge Base Built! Index saved to {config.FAISS_INDEX_PATH}")

    except Exception as e:
        logger.exception(f"❌ FAISS Build Failed: {e}")


def main():
    logger.info("--- 🚀 Starting Knowledge Base Builder ---")
    data = load_all_data_from_db()
    if data[0] is None:
        return

    documents, metadatas = create_documents(*data)
    build_vector_store_with_faiss(documents, metadatas)
    logger.info("--- 🎉 Pipeline Complete ---")


if __name__ == "__main__":
    main()
