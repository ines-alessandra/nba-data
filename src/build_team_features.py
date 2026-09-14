#!/usr/bin/env python3
"""
NBA Team Feature Engineering Script for Clustering (Seasons 2012-13 to 2024-25)

This script:
1. Connects to PostgreSQL using SQLAlchemy.
2. Queries team game logs (silver.fact_team_gamelogs), games metadata (silver.dim_games),
   and team Strength of Schedule/Simple Rating System (gold.team_sos) for regular season games
   from the 2012-13 to 2024-25 seasons.
3. Computes:
   - Traditional statistics averages (PTS, FG%, 3P%, FT%, REB, AST, etc.)
   - Consistency features (Standard Deviation of traditional statistics)
   - Temporal Trend features (rolling window trend: average of last 10 games - first 10 games net rating)
   - Clutch performance (win percentage in games decided by <= 5 points)
   - Performance against positive SRS teams (average net rating vs. teams with SRS > 0)
   - Opponent-adjusted net rating (weighted average of net rating using opponent's SRS as weight)
4. Combines everything into a single Pandas DataFrame.
5. Saves the output to:
   - A PostgreSQL table `gold.team_season_features`
   - A CSV file `data/team_season_features.csv` for analysis
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Default Database connection URL
DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Output Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "team_season_features.csv"

# Target Seasons (2012-13 to 2024-25 are season_end_years 2013 to 2025).
# 2011-12 (the lockout season) is excluded -- see trade_deadline_dates.py for
# why: activity that season is atypical for anything trade-deadline-related,
# and this feature matrix is shared with the salary-inequality analysis.
START_YEAR = 2013
END_YEAR = 2025

# Numeric metrics to calculate average and standard deviation
NUMERIC_METRICS = [
    'pts', 'pts_allowed', 'possessions', 'off_rating', 'def_rating', 'net_rating', 
    'minutes', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 
    'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'pf', 'plus_minus'
]

def get_db_engine(db_url: str):
    """Create and return a SQLAlchemy database engine."""
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"Error: Failed to connect to database at {db_url}")
        print(f"Details: {e}")
        sys.exit(1)

def fetch_raw_data(engine, start_year: int, end_year: int):
    """Fetch raw game log data joined with game info and opponent SRS rating."""
    print(f"Fetching regular season game logs for seasons ending in {start_year} to {end_year}...")
    
    query = """
        SELECT 
            f.game_id,
            f.team_id,
            f.opponent_team_id,
            f.is_home,
            f.wl,
            f.pts,
            f.pts_allowed,
            f.possessions,
            f.off_rating,
            f.def_rating,
            f.net_rating,
            f.minutes,
            f.fgm,
            f.fga,
            f.fg_pct,
            f.fg3m,
            f.fg3a,
            f.fg3_pct,
            f.ftm,
            f.fta,
            f.ft_pct,
            f.oreb,
            f.dreb,
            f.reb,
            f.ast,
            f.tov,
            f.stl,
            f.blk,
            f.pf,
            f.plus_minus,
            g.game_date,
            g.season_year,
            g.season_end_year,
            t.team_abbreviation,
            t.team_name,
            s.srs_rating AS opp_srs_rating
        FROM silver.fact_team_gamelogs f
        JOIN silver.dim_games g ON f.game_id = g.game_id
        JOIN silver.dim_teams t ON f.team_id = t.team_id
        LEFT JOIN gold.team_sos s ON f.opponent_team_id = s.team_id AND g.season_year = s.season_year
        WHERE g.game_type = 'Regular Season'
          AND g.season_end_year BETWEEN :start_year AND :end_year
        ORDER BY f.team_id, g.season_end_year, g.game_date ASC;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query), 
            con=conn, 
            params={"start_year": start_year, "end_year": end_year}
        )
        
    print(f"Retrieved {len(df):,} game records successfully.")
    return df

def build_features_matrix(df: pd.DataFrame):
    """Build the feature matrix using Pandas group computations."""
    print("Building advanced features matrix...")
    
    # Calculate the minimum opponent SRS per season to shift weights for opponent adjustment
    # This prevents negative weights in the weighted average while preserving interval differences
    min_srs_by_season = df.groupby('season_end_year')['opp_srs_rating'].min().to_dict()
    
    # Sort df to guarantee chronological order for windowing
    df_sorted = df.sort_values(by=['team_id', 'season_end_year', 'game_date']).copy()
    
    features_records = []
    
    # Group by team and season
    grouped = df_sorted.groupby(['team_id', 'season_end_year'])
    
    for (team_id, season_end_year), group in grouped:
        # Retrieve metadata
        team_abbr = group['team_abbreviation'].iloc[0]
        team_name = group['team_name'].iloc[0]
        season_year = group['season_year'].iloc[0]
        
        # Base record
        rec = {
            'team_id': int(team_id),
            'team_abbreviation': team_abbr,
            'team_name': team_name,
            'season_end_year': int(season_end_year),
            'season_year': season_year,
            'games_played': len(group)
        }
        
        # A. Traditional Means
        # B. Consistencies (Standard Deviation)
        for col in NUMERIC_METRICS:
            rec[f'avg_{col}'] = float(group[col].mean())
            rec[f'std_{col}'] = float(group[col].std())
            
        # C. Temporal Trend (Rolling Windows)
        group_chrono = group.sort_values('game_date')
        
        if len(group_chrono) >= 10:
            first_10 = group_chrono.head(10)
            last_10 = group_chrono.tail(10)
            avg_first10 = float(first_10['net_rating'].mean())
            avg_last10 = float(last_10['net_rating'].mean())
        elif len(group_chrono) > 0:
            avg_first10 = float(group_chrono['net_rating'].mean())
            avg_last10 = float(group_chrono['net_rating'].mean())
        else:
            avg_first10 = 0.0
            avg_last10 = 0.0
            
        rec['avg_net_rating_first10'] = avg_first10
        rec['avg_net_rating_last10'] = avg_last10
        rec['trend_net_rating'] = avg_last10 - avg_first10
        
        # D. Performance in Critical Situations (Clutch & vs Strong Opponents)
        # 1. Win % in close games (margin <= 5 pts based on absolute plus_minus)
        clutch_games = group_chrono[group_chrono['plus_minus'].abs() <= 5]
        if len(clutch_games) > 0:
            rec['clutch_win_pct'] = float((clutch_games['wl'] == 'W').mean())
            rec['clutch_games_count'] = int(len(clutch_games))
        else:
            rec['clutch_win_pct'] = 0.0
            rec['clutch_games_count'] = 0
            
        # 2. Avg net_rating against teams with positive SRS (opp_srs_rating > 0)
        vs_pos_srs = group_chrono[group_chrono['opp_srs_rating'] > 0]
        if len(vs_pos_srs) > 0:
            rec['avg_net_rating_vs_pos_srs'] = float(vs_pos_srs['net_rating'].mean())
            rec['vs_pos_srs_games_count'] = int(len(vs_pos_srs))
        else:
            rec['avg_net_rating_vs_pos_srs'] = 0.0
            rec['vs_pos_srs_games_count'] = 0
            
        # E. Opponent-Adjusted Net Rating (Weighted Average by Opponent SRS)
        min_srs_season = min_srs_by_season.get(season_end_year, 0.0)
        opp_srs = group_chrono['opp_srs_rating'].fillna(0.0)
        
        # Weight formula: Shift by min_srs so the worst team gets weight 1.0, 
        # and stronger teams get linearly larger weights.
        weights = opp_srs - min_srs_season + 1.0
        net_ratings = group_chrono['net_rating'].fillna(0.0)
        
        if weights.sum() > 0:
            rec['adjusted_net_rating'] = float(np.average(net_ratings, weights=weights))
        else:
            rec['adjusted_net_rating'] = float(net_ratings.mean())
            
        features_records.append(rec)
        
    features_df = pd.DataFrame(features_records)
    print(f"Successfully constructed feature matrix for {len(features_df)} team-seasons.")
    return features_df

def save_features(df: pd.DataFrame, engine, csv_path: Path):
    """Save features to database and CSV."""
    # 1. Save to PostgreSQL (gold schema)
    print("\nSaving feature matrix to PostgreSQL gold.team_season_features...")
    try:
        # Create schema if not exists
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
            conn.commit()
            
        with engine.begin() as conn:
            df.to_sql(
                name="team_season_features",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
        print("  Successfully saved to PostgreSQL: gold.team_season_features")
    except Exception as e:
        print(f"  Warning: Failed to save to PostgreSQL table. Error: {e}")
        
    # 2. Save to CSV file
    print(f"Saving feature matrix to CSV at {csv_path}...")
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Successfully saved to CSV: {csv_path}")
    except Exception as e:
        print(f"  Warning: Failed to save to CSV file. Error: {e}")

def main():
    print("=================================================================")
    print("         NBA Team Clustering Feature Matrix Builder             ")
    print("=================================================================")
    
    # 1. Connect to DB
    engine = get_db_engine(DB_URL)
    
    # 2. Retrieve regular season game logs
    raw_df = fetch_raw_data(engine, START_YEAR, END_YEAR)
    
    # Check if we have data
    if raw_df.empty:
        print("Error: No data retrieved. Please verify that the database is loaded.")
        sys.exit(1)
        
    # 3. Process features
    features_df = build_features_matrix(raw_df)
    
    # 4. Save results
    save_features(features_df, engine, DEFAULT_CSV_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
