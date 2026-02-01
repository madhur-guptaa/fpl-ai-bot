import sqlite3

import numpy as np
import pandas as pd

import config  # Import centralized configuration


# ------------------------------------------------------------------------------
# Database Schema
# ------------------------------------------------------------------------------

def create_summary_table(conn: sqlite3.Connection):
    """
    Creates the destination table schema for season summaries.
    This table compresses thousands of gameweek rows into a single row per player per season.
    """
    cursor = conn.cursor()

    query = f"""
    CREATE TABLE IF NOT EXISTS {config.DESTINATION_TABLE} (
        name TEXT,
        season TEXT,
        team TEXT,
        position TEXT,
        appearances INTEGER,
        starts INTEGER,
        minutes INTEGER,
        total_points INTEGER,
        points_per_game REAL,
        goals_scored INTEGER,
        assists INTEGER,
        expected_goals REAL,
        expected_assists REAL,
        expected_goal_involvements REAL,
        goals_per_90 REAL,
        assists_per_90 REAL,
        clean_sheets INTEGER,
        goals_conceded INTEGER,
        expected_goals_conceded REAL,
        saves INTEGER,
        bonus INTEGER,
        bps INTEGER,
        ict_index REAL,
        PRIMARY KEY (name, season)
    )
    """
    try:
        cursor.execute(query)
        conn.commit()
        print(f"✅ Table '{config.DESTINATION_TABLE}' ready.")
    except sqlite3.Error as e:
        print(f"❌ Error creating table: {e}")


# ------------------------------------------------------------------------------
# Transformation Logic
# ------------------------------------------------------------------------------

def calculate_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw gameweek data into season-level statistics.

    Key logic:
    - Sums countable stats (goals, assists, points).
    - Calculates 'appearances' by counting games with minutes > 0.
    - Derives 'per 90' metrics for fair comparison between starters and subs.
    """
    print("📊 Aggregating gameweek data...")

    # Validate required columns exist before processing
    required_cols = ['name', 'season', 'minutes', 'total_points', 'goals_scored', 'assists']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source data: {missing}")

    # Define how to aggregate each metric
    # Note: 'minutes' has two functions: sum (total mins) and custom lambda (appearances)
    agg_functions = {
        'team': 'last',
        'position': 'last',
        'minutes': ['sum', lambda x: (x > 0).sum()],
        'starts': 'sum',
        'total_points': 'sum',
        'goals_scored': 'sum',
        'assists': 'sum',
        'expected_goals': 'sum',
        'expected_assists': 'sum',
        'expected_goal_involvements': 'sum',
        'clean_sheets': 'sum',
        'goals_conceded': 'sum',
        'expected_goals_conceded': 'sum',
        'saves': 'sum',
        'bonus': 'sum',
        'bps': 'sum',
        'ict_index': 'sum'
    }

    # Group by Player and Season
    summary_df = df.groupby(['name', 'season']).agg(agg_functions)

    # Flatten the MultiIndex columns created by aggregation
    # The order here MUST match the order of keys in agg_functions + the extra lambda column
    summary_df.columns = [
        'team', 'position', 'minutes', 'appearances', 'starts', 'total_points',
        'goals_scored', 'assists', 'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'clean_sheets', 'goals_conceded',
        'expected_goals_conceded', 'saves', 'bonus', 'bps', 'ict_index'
    ]
    summary_df.reset_index(inplace=True)

    # --- Derived Metrics Calculation ---
    print("🧮 Calculating derived metrics (xG/90, PPG)...")

    # Vectorized calculation is faster and safer than loops
    # Replace infinities with 0 (happens if minutes = 0)
    summary_df['points_per_game'] = (summary_df['total_points'] / summary_df['appearances']).fillna(0).replace(
        [np.inf, -np.inf], 0).round(2)

    # Per 90 stats
    for stat in ['goals', 'assists']:
        col_name = f'{stat}_per_90'
        source_col = f'{stat}_scored' if stat == 'goals' else stat

        summary_df[col_name] = (
                (summary_df[source_col] * 90) / summary_df['minutes']
        ).fillna(0).replace([np.inf, -np.inf], 0).round(2)

    return summary_df


# ------------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------------

def main():
    """Main ETL process for historical summaries."""
    print("--- Starting Historical Summary Pipeline ---")

    try:
        with sqlite3.connect(config.DB_NAME) as conn:
            create_summary_table(conn)

            print(f"📥 Loading raw data from '{config.SOURCE_TABLE}'...")
            try:
                gw_data_df = pd.read_sql_query(f"SELECT * FROM {config.SOURCE_TABLE}", conn)
            except pd.errors.DatabaseError:
                print(f"❌ Source table '{config.SOURCE_TABLE}' not found. Run 'fpl_process_historical_data.py' first.")
                return

            if gw_data_df.empty:
                print("⚠️ Source table is empty. Aborting.")
                return

            # Transform
            summary_df = calculate_summaries(gw_data_df)

            # Load
            print(f"💾 Saving {len(summary_df)} records to '{config.DESTINATION_TABLE}'...")
            summary_df.to_sql(config.DESTINATION_TABLE, conn, if_exists='replace', index=False)
            print("✅ Success.")

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
    print("\n--- Pipeline Complete ---")
