import sqlite3
import time
from datetime import datetime
from multiprocessing import Pool
from typing import List, Dict, Optional, Any

import pandas as pd
import requests
from tqdm import tqdm

import config  # Import centralized configuration

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Limit parallel processes to avoid hitting FPL API rate limits (429 errors).
# 4-8 is usually safe.
MAX_WORKERS = 4


# ------------------------------------------------------------------------------
# API Helpers
# ------------------------------------------------------------------------------

def fetch_api_data(endpoint: str) -> Optional[Dict[str, Any]]:
    """
    Fetch JSON data from a specified FPL API endpoint.

    Args:
        endpoint (str): The specific API path (e.g., 'bootstrap-static').

    Returns:
        dict: The JSON response if successful.
        None: If the request fails.
    """
    url = f"{config.FPL_API_URL}{endpoint}/"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        # We don't print every error to avoid console spam, but we return None
        return None


def fetch_player_history(player_id: int) -> List[Dict[str, Any]]:
    """
    Worker function for parallel processing.
    Fetches detailed gameweek history for a specific player.
    """
    # Small sleep to be polite to the API server
    time.sleep(0.05)

    summary = fetch_api_data(f"element-summary/{player_id}")
    if summary and 'history' in summary:
        history = summary['history']
        # Inject player_id so we can link this data back to the player table later
        for entry in history:
            entry['player_id'] = player_id
        return history
    return []


# ------------------------------------------------------------------------------
# Main ETL Logic
# ------------------------------------------------------------------------------

def populate_current_season_data(conn: sqlite3.Connection):
    """
    Orchestrator function:
    1. Fetches static data (Players, Teams, Fixtures).
    2. Fetches dynamic data (Detailed Player Histories) in parallel.
    3. Saves everything to SQLite.
    """
    print("\n--- 🚀 Populating Current Season Data ---")

    # 1. Fetch Bootstrap Static Data (The massive JSON with everything)
    bootstrap = fetch_api_data("bootstrap-static")
    if not bootstrap:
        print("❌ Critical Error: Could not fetch 'bootstrap-static'. Aborting.")
        return

    # --------------------------------------------------------------------------
    # Table 1: Players (Elements)
    # --------------------------------------------------------------------------
    print("Processing Players...")
    # Select only the columns we actually care about
    player_cols = [
        'id', 'web_name', 'team', 'element_type', 'now_cost', 'status', 'news',
        'total_points', 'form', 'points_per_game', 'selected_by_percent',
        'ict_index', 'ep_next', 'transfers_in_event', 'transfers_out_event',
        'expected_goals', 'expected_assists'
    ]

    players_df = pd.DataFrame(bootstrap.get("elements", []))

    # Filter and Rename
    players_df = players_df[player_cols].rename(columns={
        'id': 'player_id',
        'web_name': 'name',
        'team': 'team_id',
        'element_type': 'position',
        'now_cost': 'price',
        'transfers_in_event': 'transfers_in',
        'transfers_out_event': 'transfers_out',
        'expected_goals': 'xG',
        'expected_assists': 'xA'
    })

    # Normalize price (FPL stores £10.0m as 100)
    players_df['price'] = players_df['price'] / 10.0
    players_df['last_updated'] = datetime.utcnow().isoformat()

    players_df.to_sql('players', conn, if_exists='replace', index=False)
    print(f"✅ Saved {len(players_df)} players.")

    # --------------------------------------------------------------------------
    # Table 2: Teams
    # --------------------------------------------------------------------------
    print("Processing Teams...")
    team_cols = ['id', 'name', 'short_name', 'strength_attack_home',
                 'strength_attack_away', 'strength_defence_home', 'strength_defence_away']

    team_df = pd.DataFrame(bootstrap.get("teams", []))
    team_df = team_df[team_cols].rename(columns={'id': 'team_id'})

    team_df.to_sql('teams', conn, if_exists='replace', index=False)
    print(f"✅ Saved {len(team_df)} teams.")

    # --------------------------------------------------------------------------
    # Table 3: Fixtures
    # --------------------------------------------------------------------------
    print("Processing Fixtures...")
    fixture_data = fetch_api_data("fixtures")
    if fixture_data:
        fixture_df = pd.DataFrame(fixture_data)

        # Select and Rename
        fixture_df = fixture_df.rename(columns={
            'id': 'fixture_id',
            'event': 'gameweek',
            'team_h': 'home_team_id',
            'team_a': 'away_team_id',
            'team_h_difficulty': 'home_difficulty',
            'team_a_difficulty': 'away_difficulty'
        })

        # Filter for relevant columns
        cols_to_keep = ['fixture_id', 'gameweek', 'home_team_id', 'away_team_id',
                        'home_difficulty', 'away_difficulty', 'kickoff_time',
                        'team_h_score', 'team_a_score', 'finished']

        fixture_df = fixture_df[cols_to_keep]

        fixture_df.to_sql('fixtures', conn, if_exists='replace', index=False)
        print(f"✅ Saved {len(fixture_df)} fixtures.")
    else:
        print("⚠️ Warning: No fixture data found.")

    # --------------------------------------------------------------------------
    # Table 4: Player Gameweek Histories (Parallel Fetch)
    # --------------------------------------------------------------------------
    print("\n--- ⚡ Fetching detailed stats for all players ---")
    player_ids = players_df['player_id'].tolist()

    # Use 'MAX_WORKERS' to control concurrency
    with Pool(processes=MAX_WORKERS) as pool:
        # imap_unordered yields results as soon as they are ready
        results = list(tqdm(
            pool.imap_unordered(fetch_player_history, player_ids),
            total=len(player_ids),
            desc="Downloading Histories"
        ))

    # Flatten the list of lists
    all_player_stats = [item for sublist in results for item in sublist]

    if all_player_stats:
        player_gw_df = pd.DataFrame(all_player_stats)
        player_gw_df.to_sql('player_gameweek_stats', conn, if_exists='replace', index=False)
        print(f"✅ Saved {len(player_gw_df)} gameweek records.")
    else:
        print("⚠️ Warning: No gameweek stats were fetched.")


def main():
    """Main ETL entry point for Live Data."""
    print("--- Starting Live Data ETL ---")
    try:
        with sqlite3.connect(config.DB_NAME) as conn:
            populate_current_season_data(conn)
            print(f"\n✨ Live API data successfully updated in '{config.DB_NAME}'.")
    except sqlite3.Error as e:
        print(f"❌ Database Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    main()
