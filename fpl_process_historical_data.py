import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

import config  # Import centralized configuration


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def load_teams_map(teams_csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads the teams.csv file for a specific season to create a mapping
    between opponent IDs and team names.

    Args:
        teams_csv_path (Path): Path to the teams.csv file.

    Returns:
        pd.DataFrame: A subset dataframe with 'opposition_team_id' and 'opposition_team_name',
                      or None if loading fails.
    """
    try:
        # Explicitly use utf-8 to handle accented team names if present
        teams_df = pd.read_csv(teams_csv_path, encoding="utf-8")

        # We only need the ID and Name to map against the gameweek data
        return teams_df[["id", "name"]].rename(
            columns={"id": "opposition_team_id", "name": "opposition_team_name"}
        )
    except Exception as e:
        print(f"  [ERROR] Could not read teams map at {teams_csv_path}: {e}")
        return None


def process_gameweek_file(gw_file: Path, season_name: str, teams_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Reads a single gameweek CSV, enriches it with metadata (season, GW number),
    and merges it with the opponent team name.

    Args:
        gw_file (Path): Path to the specific gameweek CSV (e.g., 'gw1.csv').
        season_name (str): The season label (e.g., '2020-21').
        teams_df (pd.DataFrame): The team mapping dataframe for this season.

    Returns:
        pd.DataFrame: The processed dataframe for this gameweek, or None if skipped.
    """
    try:
        # Extract gameweek number safely using regex (e.g., "gw12.csv" -> 12)
        match = re.search(r"gw(\d+)\.csv", gw_file.name)
        if not match:
            return None

        gameweek_number = int(match.group(1))

        # Explicit encoding to be safe
        gw_df = pd.read_csv(gw_file, encoding="utf-8")

        if gw_df.empty:
            return None

        # --- Enrichment Step ---
        # Add metadata so we can distinguish this data later in the massive SQL table
        gw_df['season'] = season_name
        gw_df['gameweek'] = gameweek_number

        # Map 'opponent_team' ID to the actual text name (e.g., 14 -> 'Man Utd')
        # This makes the data human-readable for the LLM later.
        gw_df = gw_df.merge(
            teams_df,
            left_on='opponent_team',
            right_on='opposition_team_id',
            how='left'
        )
        return gw_df

    except Exception as e:
        print(f"    [SKIP] Failed to process file '{gw_file.name}': {e}")
        return None


# ------------------------------------------------------------------------------
# Main ETL Pipeline
# ------------------------------------------------------------------------------

def create_fpl_database(base_dir_path: str, db_name: str):
    """
    ETL Pipeline: Extracts historical CSV data, Transforms it by adding metadata
    and merging team names, and Loads it into SQLite.

    Structure Expected:
    historical_gw_data/
      ├── 2020-21/
      │   ├── gws/ (gw1.csv, gw2.csv...)
      │   └── teams.csv
      └── ...
    """
    base_dir = Path(base_dir_path)
    if not base_dir.is_dir():
        print(f"Error: Base directory '{base_dir}' does not exist. Aborting.")
        return

    print(f"--- Starting Historical Data ETL from '{base_dir}' ---")
    all_gw_dfs: List[pd.DataFrame] = []

    # Sort directories to ensure chronological processing order
    season_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for season_dir in season_dirs:
        season_name = season_dir.name
        gws_dir = season_dir / "gws"
        teams_csv_path = season_dir / "teams.csv"

        # Validation: Ensure strict folder structure exists before processing
        if not gws_dir.is_dir() or not teams_csv_path.exists():
            print(f"  [SKIP] Season '{season_name}': Missing 'gws' folder or 'teams.csv'.")
            continue

        print(f"  Processing Season: {season_name}...")

        # Step 1: Load Team Mapping
        teams_df = load_teams_map(teams_csv_path)
        if teams_df is None:
            continue

        # Step 2: Process all Gameweeks for this season
        gw_files = sorted(gws_dir.glob("gw*.csv"))
        for gw_file in gw_files:
            processed_df = process_gameweek_file(gw_file, season_name, teams_df)
            if processed_df is not None:
                all_gw_dfs.append(processed_df)

    if not all_gw_dfs:
        print("Error: No valid data found. Check directory structure.")
        return

    # Step 3: Concatenate and Load to DB
    print("\nMerging all seasons into a single dataset...")
    historical_gameweek_data = pd.concat(all_gw_dfs, ignore_index=True)

    table_name = "historical_gameweek_data"
    print(f"Saving {len(historical_gameweek_data):,} rows to SQLite database: {db_name}...")

    try:
        with sqlite3.connect(db_name) as conn:
            historical_gameweek_data.to_sql(table_name, conn, if_exists="replace", index=False)
        print("✅ Data successfully saved.")
    except Exception as e:
        print(f"❌ Database Write Error: {e}")


if __name__ == "__main__":
    # Ensure config.DB_NAME is correctly set in your config.py
    create_fpl_database(base_dir_path="historical_gw_data", db_name=config.DB_NAME)
    print("\n--- Historical Data Processing Complete ---")
