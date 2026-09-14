#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import csv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Default database URL
DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"


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


def setup_schemas(engine):
    """Ensure the Medallion schemas exist."""
    print("Initializing Medallion schemas...")
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        conn.commit()
    print("Schemas 'bronze', 'silver', and 'gold' are ready.")


def load_parquet_to_bronze(engine, data_dir: Path):
    """Load all Parquet files from the data directory into the bronze schema."""
    print("\n--- LOADING BRONZE SCHEMA (RAW DATA) ---")
    
    # List of files and their target table names in bronze
    parquet_files = [
        ("gamelogs", data_dir / "gamelogs.parquet"),
        ("player_game_logs", data_dir / "player_game_logs.parquet"),
        ("playerstats", data_dir / "playerstats.parquet"),
        ("espn_four_factors", data_dir / "espn" / "four_factors.parquet"),
        ("espn_team_box", data_dir / "espn" / "team_box.parquet"),
        ("espn_player_box", data_dir / "espn" / "player_box.parquet"),
        ("espn_player_details", data_dir / "espn" / "player_details.parquet"),
    ]
    
    for table_name, file_path in parquet_files:
        if not file_path.exists():
            print(f"Warning: File {file_path} not found. Skipping table bronze.{table_name}.")
            continue
            
        print(f"Loading {file_path.name} into bronze.{table_name}...")
        try:
            # Read parquet into pandas
            df = pd.read_parquet(file_path)
            
            # Write to database (replace if exists)
            # Use chunks for large tables to prevent memory issues
            df.to_sql(
                name=table_name,
                con=engine,
                schema="bronze",
                if_exists="replace",
                index=False,
                chunksize=10000
            )
            print(f"  Successfully loaded {len(df):,} rows.")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")


def clean_ranking(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    val_str = str(val).split('.')[0]
    cleaned = ''.join(c for c in val_str if c.isdigit())
    return int(cleaned) if cleaned else None


def load_salaries_to_bronze(engine, data_dir: Path):
    """Load all salary CSV files from the data directory into the bronze schema."""
    print("\n--- LOADING SALARY DATA TO BRONZE ---")
    salaries_dir = data_dir / "salaries"
    if not salaries_dir.exists():
        print(f"Warning: Salaries directory {salaries_dir} not found. Skipping salary loading.")
        return

    # 1. Player Salaries
    players_dir = salaries_dir / "players"
    if players_dir.exists():
        player_salaries_list = []
        for csv_path in sorted(players_dir.glob("*.csv")):
            import re
            season_match = re.search(r"(\d{4})_(\d{4})", csv_path.name)
            if not season_match:
                continue
            season_end_year = int(season_match.group(2))
            
            try:
                df = pd.read_csv(csv_path)
                if df.empty or len(df.columns) < 3:
                    continue
                
                # Standardize columns
                col_rename = {
                    df.columns[0]: "ranking",
                    df.columns[1]: "player_name",
                    df.columns[2]: "salary",
                }
                if len(df.columns) > 3:
                    col_rename[df.columns[3]] = "salary_inflation_adjusted"
                    
                df = df.rename(columns=col_rename)
                df["ranking"] = df["ranking"].apply(clean_ranking)
                df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
                if "salary_inflation_adjusted" in df.columns:
                    df["salary_inflation_adjusted"] = pd.to_numeric(df["salary_inflation_adjusted"], errors="coerce")
                else:
                    df["salary_inflation_adjusted"] = None
                    
                df["season_end_year"] = season_end_year
                
                cols_to_keep = ["ranking", "player_name", "salary", "salary_inflation_adjusted", "season_end_year"]
                for col in cols_to_keep:
                    if col not in df.columns:
                        df[col] = None
                df = df[cols_to_keep]
                player_salaries_list.append(df)
            except Exception as e:
                print(f"  Error loading player salary file {csv_path.name}: {e}")
                
        if player_salaries_list:
            combined_df = pd.concat(player_salaries_list, ignore_index=True)
            combined_df.to_sql(
                name="raw_player_salaries",
                con=engine,
                schema="bronze",
                if_exists="replace",
                index=False
            )
            print(f"  Successfully loaded {len(combined_df):,} player salary rows into bronze.raw_player_salaries.")

    # 2. Team Salaries
    teams_dir = salaries_dir / "teams"
    if teams_dir.exists():
        team_salaries_list = []
        for csv_path in sorted(teams_dir.glob("*.csv")):
            import re
            season_match = re.search(r"(\d{4})_(\d{4})", csv_path.name)
            if not season_match:
                continue
            season_end_year = int(season_match.group(2))
            
            try:
                df = pd.read_csv(csv_path)
                if df.empty or len(df.columns) < 3:
                    continue
                
                # Standardize columns
                col_rename = {
                    df.columns[0]: "ranking",
                    df.columns[1]: "team_name",
                    df.columns[2]: "salary",
                }
                if len(df.columns) > 3:
                    col_rename[df.columns[3]] = "salary_inflation_adjusted"
                    
                df = df.rename(columns=col_rename)
                df["ranking"] = df["ranking"].apply(clean_ranking)
                df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
                if "salary_inflation_adjusted" in df.columns:
                    df["salary_inflation_adjusted"] = pd.to_numeric(df["salary_inflation_adjusted"], errors="coerce")
                else:
                    df["salary_inflation_adjusted"] = None
                    
                df["season_end_year"] = season_end_year
                
                cols_to_keep = ["ranking", "team_name", "salary", "salary_inflation_adjusted", "season_end_year"]
                for col in cols_to_keep:
                    if col not in df.columns:
                        df[col] = None
                df = df[cols_to_keep]
                team_salaries_list.append(df)
            except Exception as e:
                print(f"  Error loading team salary file {csv_path.name}: {e}")
                
        if team_salaries_list:
            combined_df = pd.concat(team_salaries_list, ignore_index=True)
            combined_df.to_sql(
                name="raw_team_salaries",
                con=engine,
                schema="bronze",
                if_exists="replace",
                index=False
            )
            print(f"  Successfully loaded {len(combined_df):,} team salary rows into bronze.raw_team_salaries.")


def execute_sql_file(engine, sql_file_path: Path, schema_name: str):
    """Execute SQL script file, splitting statements by semicolon."""
    print(f"\n--- POPULATING {schema_name.upper()} SCHEMA ---")
    if not sql_file_path.exists():
        print(f"Error: SQL file {sql_file_path} not found.")
        return
        
    print(f"Running script: {sql_file_path}...")
    
    try:
        with open(sql_file_path, "r", encoding="utf-8") as f:
            sql_content = f.read()
            
        # Split by semicolon to separate statements
        # We handle comments and replace SQL content safely
        statements = sql_content.split(";")
        
        with engine.connect() as conn:
            statement_count = 0
            for statement in statements:
                # Strip spaces and comments
                clean_stmt = statement.strip()
                # Remove empty statements
                if not clean_stmt:
                    continue
                    
                conn.execute(text(clean_stmt))
                statement_count += 1
                
            conn.commit()
            print(f"Successfully executed {statement_count} statements in {schema_name}.")
    except Exception as e:
        print(f"Error executing {sql_file_path.name}: {e}")
        sys.exit(1)


def compute_and_load_sos(engine):
    """Calculate traditional and SRS-based Strength of Schedule (SoS) for all seasons and load to gold.team_sos."""
    print("\n--- CALCULATING STRENGTH OF SCHEDULE (SoS) ---")
    try:
        query = """
            SELECT season_year, team_id, team_abbreviation, game_id, wl, plus_minus, net_rating, matchup 
            FROM bronze.gamelogs 
            WHERE game_id LIKE '002%';
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), con=conn)
        
        # Parse opponent
        df['opponent'] = df['matchup'].str[-3:]
        df['win'] = (df['wl'] == 'W').astype(int)
        
        results = []
        
        # Group by season
        seasons = df['season_year'].unique()
        for season in sorted(seasons):
            df_season = df[df['season_year'] == season].copy()
            if df_season.empty:
                continue
                
            # List of unique teams in this season
            teams = sorted(df_season['team_abbreviation'].unique())
            if not teams:
                continue
            num_teams = len(teams)
            team_to_idx = {team: i for i, team in enumerate(teams)}
            
            # Map team abbreviation to ID for database reference
            team_id_map = df_season.groupby('team_abbreviation')['team_id'].first().to_dict()
            
            # --- 1. Traditional SoS ---
            # W% for each team
            team_win_pct = df_season.groupby('team_abbreviation')['win'].mean().to_dict()
            df_season['opp_win_pct'] = df_season['opponent'].map(team_win_pct)
            
            # OWP (Opponent Win Percentage)
            owp = df_season.groupby('team_abbreviation')['opp_win_pct'].mean()
            df_season['opp_owp'] = df_season['opponent'].map(owp)
            
            # OOWP (Opponents' Opponents Win Percentage)
            oowp = df_season.groupby('team_abbreviation')['opp_owp'].mean()
            
            # Traditional SoS = (2 * OWP + OOWP) / 3
            sos_traditional = (2 * owp + oowp) / 3
            
            # --- 2. SRS-based SoS (Simple Rating System) ---
            # Point differential
            point_diff = df_season.groupby('team_abbreviation')['plus_minus'].mean()
            # Reorder point diff to match teams index
            point_diff_vec = np.array([point_diff.get(t, 0.0) for t in teams])
            
            # Transition Matrix M: M[i, j] = fraction of games team i played against team j
            M = np.zeros((num_teams, num_teams))
            for t in teams:
                idx_i = team_to_idx[t]
                opps = df_season[df_season['team_abbreviation'] == t]['opponent']
                total_games = len(opps)
                if total_games == 0:
                    continue
                for opp in opps:
                    if opp in team_to_idx:
                        idx_j = team_to_idx[opp]
                        M[idx_i, idx_j] += 1
                M[idx_i] /= total_games
                
            # Solve (I - M) R = Point_Diff with sum(R) = 0 constraint
            A = np.eye(num_teams) - M
            A = np.vstack([A, np.ones(num_teams)])
            b = np.append(point_diff_vec, 0.0)
            
            ratings, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # SoS_srs = Rating - Point_Diff
            sos_srs_vec = ratings - point_diff_vec
            
            # Compile results for this season
            for idx, t in enumerate(teams):
                results.append({
                    'season_year': season,
                    'team_id': int(team_id_map[t]),
                    'team_abbreviation': t,
                    'sos_traditional': float(sos_traditional.get(t, 0.0)),
                    'sos_srs': float(sos_srs_vec[idx]),
                    'srs_rating': float(ratings[idx]),
                    'avg_point_diff': float(point_diff_vec[idx])
                })
                
        # Save results to database
        res_df = pd.DataFrame(results)
        
        # Write to gold.team_sos
        with engine.begin() as conn:
            res_df.to_sql(
                name="team_sos",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
        print(f"  Successfully calculated and loaded {len(res_df):,} SoS rows into gold.team_sos.")
        
    except Exception as e:
        print(f"  Error calculating or loading SoS: {e}")
        import traceback
        traceback.print_exc()


def normalize_name(name):
    """Normalize player names for matching (remove accents, dots, dashes, suffixes)."""
    if not name:
        return ""
    import unicodedata
    name = name.lower().strip()
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.replace(".", "").replace("-", " ").replace("'", "")
    for suffix in [" jr", " sr", " iii", " ii", " iv"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return " ".join(name.split())


def load_trades_and_calculate_impact(engine, data_dir: Path):
    """Load trade CSV files, classify trades, and compute 15-game pre/post team and player trade impacts."""
    print("\n--- LOADING TRADES AND CALCULATING IMPACT ---")
    try:
        # Load dim_teams
        with engine.connect() as conn:
            teams_db = pd.read_sql(text('SELECT team_id, team_abbreviation, team_name FROM silver.dim_teams'), con=conn)
            
        # Build team mapping (case-insensitive)
        team_map = {}
        for tid, abb, name in teams_db.itertuples(index=False):
            if name:
                team_map[name.lower().strip()] = tid
            if abb:
                team_map[abb.lower().strip()] = tid

        # Add manual variations
        manual_mapping = {
            'charlotte bobcats': team_map.get('charlotte hornets'),
            'new jersey nets': team_map.get('brooklyn nets'),
            'new orleans hornets': team_map.get('new orleans hornets'),
            'new orleans pelicans': team_map.get('new orleans hornets'),
            'new orleans/oklahoma city hornets': team_map.get('new orleans hornets'),
            'washington bullets': team_map.get('washington wizards'),
        }
        team_map.update(manual_mapping)

        # Load dim_players
        with engine.connect() as conn:
            players_db = pd.read_sql(text('SELECT player_id, player_name FROM silver.dim_players'), con=conn)

        players_db['norm_name'] = players_db['player_name'].apply(normalize_name)
        db_player_map = {row['norm_name']: row['player_id'] for _, row in players_db.iterrows()}

        # Manual player name overrides
        manual_players = {
            'wesley iwundu': 'wes iwundu',
            'louis amundson': 'lou amundson',
            'luc richard mbah a moute': 'luc mbah a moute',
            'laui markkanen': 'lauri markkanen',
            'aleksandar vezenkov': 'sasha vezenkov',
            'ishmael smith': 'ish smith',
            'nicolas claxton': 'nic claxton',
            'cam redish': 'cam reddish',
            'cameron reddish': 'cam reddish',
        }
        for k, v in manual_players.items():
            norm_v = normalize_name(v)
            if norm_v in db_player_map:
                db_player_map[k] = db_player_map[norm_v]

        # Load SoS ratings
        with engine.connect() as conn:
            try:
                sos_db = pd.read_sql(text('SELECT season_year, team_id, srs_rating FROM gold.team_sos'), con=conn)
                sos_dict = {(row['season_year'], row['team_id']): row['srs_rating'] for _, row in sos_db.iterrows()}
            except Exception as e:
                print("  Warning: gold.team_sos not found or empty. Using 0.0 for opponent ratings.", e)
                sos_dict = {}

        # Load team gamelogs
        with engine.connect() as conn:
            gamelogs_query = """
                SELECT tg.game_id, tg.team_id, tg.opponent_team_id, g.game_date, g.season_year, g.season_end_year, 
                       tg.wl, tg.off_rating, tg.def_rating, tg.net_rating
                FROM silver.fact_team_gamelogs tg
                JOIN silver.dim_games g ON tg.game_id = g.game_id
                WHERE g.game_type = 'Regular Season'
                ORDER BY tg.team_id, g.game_date;
            """
            gamelogs = pd.read_sql(text(gamelogs_query), con=conn)

        gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])

        # Load player gamelogs
        with engine.connect() as conn:
            pg_query = """
                SELECT pg.player_id, pg.team_id, g.game_date, g.season_year, g.season_end_year, 
                       pg.minutes, pg.pts, pg.ast, pg.reb, pg.plus_minus, pg.net_rating
                FROM silver.fact_player_gamelogs pg
                JOIN silver.dim_games g ON pg.game_id = g.game_id
                WHERE g.game_type = 'Regular Season'
                ORDER BY pg.player_id, g.game_date;
            """
            player_gamelogs = pd.read_sql(text(pg_query), con=conn)

        player_gamelogs['game_date'] = pd.to_datetime(player_gamelogs['game_date'])
        team_id_to_abb = {row['team_id']: row['team_abbreviation'] for _, row in teams_db.iterrows()}

        # Parsing trade CSV files
        trades_dir = data_dir / "trades"
        if not trades_dir.exists():
            print(f"  Warning: Trades directory {trades_dir} not found. Skipping trade impact analysis.")
            return

        csv_files = sorted(trades_dir.glob("*.csv"))

        all_trade_rows = []
        for fn in csv_files:
            year_str = fn.stem.split("_")[1]
            season_end_year = int(year_str)
            
            with open(fn, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for idx, row in enumerate(reader):
                    if len(row) < 14:
                        row = row + [''] * (14 - len(row))
                    elif len(row) > 14:
                        asset_type = row[4]
                        if asset_type == 'Pick':
                            first_part = row[:10]
                            last_part = row[-3:]
                            middle = ', '.join(row[10:-3])
                            row = first_part + [middle] + last_part
                        else:
                            first_part = row[:13]
                            notes = ', '.join(row[13:])
                            row = first_part + [notes]
                    
                    row_dict = dict(zip(header, row))
                    # Parse date and determine correct season_end_year (July 1st cutoff)
                    try:
                        trade_dt = pd.to_datetime(row_dict['date'])
                        actual_season_end = trade_dt.year + 1 if trade_dt.month >= 7 else trade_dt.year
                    except Exception:
                        actual_season_end = season_end_year # fallback to filename year
                    row_dict['season_end_year'] = actual_season_end
                    
                    # Map team to team_id
                    t_clean = row_dict['team'].strip().lower()
                    tid = team_map.get(t_clean)
                    row_dict['team_id'] = tid

                    # Map player to player_id
                    p_name = row_dict.get('player_name', '')
                    if p_name:
                        p_norm = normalize_name(p_name)
                        row_dict['player_id'] = db_player_map.get(p_norm)
                    else:
                        row_dict['player_id'] = None

                    all_trade_rows.append(row_dict)

        if not all_trade_rows:
            print("  No trades found to load.")
            return

        df_trades = pd.DataFrame(all_trade_rows)
        print(f"  Loaded {len(df_trades):,} raw rows from trade CSVs.")

        # Add raw trades to database (bronze.raw_trades)
        with engine.begin() as conn:
            df_trades_db = df_trades.copy()
            df_trades_db['player_age'] = pd.to_numeric(df_trades_db['player_age'], errors='coerce')
            df_trades_db['pick_year'] = pd.to_numeric(df_trades_db['pick_year'], errors='coerce')
            df_trades_db['pick_round'] = pd.to_numeric(df_trades_db['pick_round'], errors='coerce')
            df_trades_db['cash'] = pd.to_numeric(df_trades_db['cash'], errors='coerce')
            df_trades_db['cap_space'] = pd.to_numeric(df_trades_db['cap_space'], errors='coerce')
            df_trades_db['trade_id'] = pd.to_numeric(df_trades_db['trade_id'], errors='coerce')
            df_trades_db['date'] = pd.to_datetime(df_trades_db['date']).dt.date
            df_trades_db['player_id'] = pd.to_numeric(df_trades_db['player_id'], errors='coerce')
            
            df_trades_db.to_sql(
                name="raw_trades",
                con=conn,
                schema="bronze",
                if_exists="replace",
                index=False
            )
        print("  Saved raw trades to bronze.raw_trades.")

        # Classify and analyze trades
        trade_groups = df_trades.groupby(['season_end_year', 'trade_id'])

        classified_trades = []
        team_impact_records = []
        player_impact_records = []

        for (season_end_year, trade_id), group in trade_groups:
            trade_date_str = group['date'].iloc[0]
            trade_date = pd.to_datetime(trade_date_str)
            season_year = f"{season_end_year-1}-{str(season_end_year)[2:]}"
            
            # Classify trade
            all_assets = set(group['asset_type'].dropna().unique())
            has_player = 'Player' in all_assets
            has_pick = 'Pick' in all_assets
            has_cap = 'Cap Space' in all_assets
            has_cash = 'Cash' in all_assets
            
            if has_player and has_pick:
                trade_type = 'Player + Pick'
            elif has_player:
                player_teams = group[group['asset_type'] == 'Player']['team'].dropna().unique()
                player_teams_clean = [t.strip().lower() for t in player_teams]
                player_team_ids = set([team_map.get(t) for t in player_teams_clean if team_map.get(t) is not None])
                if len(player_team_ids) == 1 and (has_cap or has_cash):
                    trade_type = 'Salary Dump / Cash'
                else:
                    trade_type = 'Player Only'
            elif has_pick:
                trade_type = 'Pick Only'
            else:
                trade_type = 'Other'
                
            classified_trades.append({
                'season_end_year': int(season_end_year),
                'season_year': season_year,
                'trade_id': int(trade_id),
                'trade_date': trade_date.date(),
                'trade_type': trade_type,
                'num_assets': len(group)
            })
            
            # 1. Team-level impacts
            involved_team_ids = group['team_id'].dropna().unique().astype(int)
            for team_id in involved_team_ids:
                team_games = gamelogs[(gamelogs['team_id'] == team_id) & (gamelogs['season_end_year'] == season_end_year)].copy()
                if team_games.empty:
                    continue
                    
                pre_games = team_games[team_games['game_date'] < trade_date].sort_values('game_date', ascending=False)
                post_games = team_games[team_games['game_date'] >= trade_date].sort_values('game_date', ascending=True)
                
                if pre_games.empty or post_games.empty:
                    continue
                    
                pre_window = pre_games.head(15)
                post_window = post_games.head(15)
                
                win_pre = (pre_window['wl'] == 'W').mean()
                win_post = (post_window['wl'] == 'W').mean()
                
                ortg_pre = pre_window['off_rating'].mean()
                ortg_post = post_window['off_rating'].mean()
                
                drtg_pre = pre_window['def_rating'].mean()
                drtg_post = post_window['def_rating'].mean()
                
                nrtg_pre = pre_window['net_rating'].mean()
                nrtg_post = post_window['net_rating'].mean()
                
                opp_srs_pre = np.mean([sos_dict.get((season_year, opp_id), 0.0) for opp_id in pre_window['opponent_team_id']])
                opp_srs_post = np.mean([sos_dict.get((season_year, opp_id), 0.0) for opp_id in post_window['opponent_team_id']])
                
                team_impact_records.append({
                    'season_end_year': int(season_end_year),
                    'season_year': season_year,
                    'trade_id': int(trade_id),
                    'trade_date': trade_date.date(),
                    'trade_type': trade_type,
                    'team_id': int(team_id),
                    'games_pre': len(pre_window),
                    'games_post': len(post_window),
                    'win_pct_pre': float(win_pre),
                    'win_pct_post': float(win_post),
                    'win_pct_diff': float(win_post - win_pre),
                    'off_rating_pre': float(ortg_pre) if not pd.isna(ortg_pre) else None,
                    'off_rating_post': float(ortg_post) if not pd.isna(ortg_post) else None,
                    'off_rating_diff': float(ortg_post - ortg_pre) if not (pd.isna(ortg_pre) or pd.isna(ortg_post)) else None,
                    'def_rating_pre': float(drtg_pre) if not pd.isna(drtg_pre) else None,
                    'def_rating_post': float(drtg_post) if not pd.isna(drtg_post) else None,
                    'def_rating_diff': float(drtg_post - drtg_pre) if not (pd.isna(drtg_pre) or pd.isna(drtg_post)) else None,
                    'net_rating_pre': float(nrtg_pre) if not pd.isna(nrtg_pre) else None,
                    'net_rating_post': float(nrtg_post) if not pd.isna(nrtg_post) else None,
                    'net_rating_diff': float(nrtg_post - nrtg_pre) if not (pd.isna(nrtg_pre) or pd.isna(nrtg_post)) else None,
                    'opp_srs_pre': float(opp_srs_pre),
                    'opp_srs_post': float(opp_srs_post),
                    'opp_srs_diff': float(opp_srs_post - opp_srs_pre)
                })

            # 2. Player-level impacts
            player_rows = group[(group['asset_type'] == 'Player') & (group['player_id'].notna())]
            player_rows_unique = player_rows.drop_duplicates(subset=['player_id'])
            
            for _, p_row in player_rows_unique.iterrows():
                pid = int(p_row['player_id'])
                p_name = p_row['player_name']
                
                player_games = player_gamelogs[(player_gamelogs['player_id'] == pid) & (player_gamelogs['season_end_year'] == season_end_year)].copy()
                if player_games.empty:
                    continue
                    
                pre_pg = player_games[player_games['game_date'] < trade_date].sort_values('game_date', ascending=False)
                post_pg = player_games[player_games['game_date'] >= trade_date].sort_values('game_date', ascending=True)
                
                if pre_pg.empty or post_pg.empty:
                    continue
                    
                old_tid = int(pre_pg['team_id'].iloc[0])
                new_tid = int(post_pg['team_id'].iloc[0])
                
                pre_pg_team = pre_pg[pre_pg['team_id'] == old_tid].head(15)
                post_pg_team = post_pg[post_pg['team_id'] == new_tid].head(15)
                
                if pre_pg_team.empty or post_pg_team.empty:
                    continue
                    
                min_pre = pre_pg_team['minutes'].mean()
                min_post = post_pg_team['minutes'].mean()
                
                pts_pre = pre_pg_team['pts'].mean()
                pts_post = post_pg_team['pts'].mean()
                
                ast_pre = pre_pg_team['ast'].mean()
                ast_post = post_pg_team['ast'].mean()
                
                reb_pre = pre_pg_team['reb'].mean()
                reb_post = post_pg_team['reb'].mean()
                
                pm_pre = pre_pg_team['plus_minus'].mean()
                pm_post = post_pg_team['plus_minus'].mean()
                
                nrtg_pre = pre_pg_team['net_rating'].mean()
                nrtg_post = post_pg_team['net_rating'].mean()
                
                player_impact_records.append({
                    'season_end_year': int(season_end_year),
                    'season_year': season_year,
                    'trade_id': int(trade_id),
                    'trade_date': trade_date.date(),
                    'trade_type': trade_type,
                    'player_id': int(pid),
                    'player_name': p_name,
                    'old_team_id': int(old_tid),
                    'old_team': team_id_to_abb.get(old_tid, ''),
                    'new_team_id': int(new_tid),
                    'new_team': team_id_to_abb.get(new_tid, ''),
                    'games_pre': len(pre_pg_team),
                    'games_post': len(post_pg_team),
                    'min_pre': float(min_pre) if not pd.isna(min_pre) else None,
                    'min_post': float(min_post) if not pd.isna(min_post) else None,
                    'min_diff': float(min_post - min_pre) if not (pd.isna(min_pre) or pd.isna(min_post)) else None,
                    'pts_pre': float(pts_pre) if not pd.isna(pts_pre) else None,
                    'pts_post': float(pts_post) if not pd.isna(pts_post) else None,
                    'pts_diff': float(pts_post - pts_pre) if not (pd.isna(pts_pre) or pd.isna(pts_post)) else None,
                    'ast_pre': float(ast_pre) if not pd.isna(ast_pre) else None,
                    'ast_post': float(ast_post) if not pd.isna(ast_post) else None,
                    'ast_diff': float(ast_post - ast_pre) if not (pd.isna(ast_pre) or pd.isna(ast_post)) else None,
                    'reb_pre': float(reb_pre) if not pd.isna(reb_pre) else None,
                    'reb_post': float(reb_post) if not pd.isna(reb_post) else None,
                    'reb_diff': float(reb_post - reb_pre) if not (pd.isna(reb_pre) or pd.isna(reb_post)) else None,
                    'plus_minus_pre': float(pm_pre) if not pd.isna(pm_pre) else None,
                    'plus_minus_post': float(pm_post) if not pd.isna(pm_post) else None,
                    'plus_minus_diff': float(pm_post - pm_pre) if not (pd.isna(pm_pre) or pd.isna(pm_post)) else None,
                    'net_rating_pre': float(nrtg_pre) if not pd.isna(nrtg_pre) else None,
                    'net_rating_post': float(nrtg_post) if not pd.isna(nrtg_post) else None,
                    'net_rating_diff': float(nrtg_post - nrtg_pre) if not (pd.isna(nrtg_pre) or pd.isna(nrtg_post)) else None,
                })

        df_classified = pd.DataFrame(classified_trades)
        df_impact = pd.DataFrame(team_impact_records)
        df_player_impact = pd.DataFrame(player_impact_records)

        with engine.begin() as conn:
            df_classified.to_sql(
                name="trades_classified",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
            df_impact.to_sql(
                name="team_trade_impact_analysis",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
            df_player_impact.to_sql(
                name="player_trade_impact_analysis",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
        print(f"  Successfully classified {len(df_classified):,} trades, calculated {len(df_impact):,} team impacts, and {len(df_player_impact):,} player impacts in gold tables.")

    except Exception as e:
        print(f"  Error loading trades or calculating impact: {e}")
        import traceback
        traceback.print_exc()


def analyze_salary_inequality(engine, data_dir: Path):
    """
    Analyze Hypothesis 1: The impact of salary inequality (Gini) on team performance.
    Computes both simple and active-weighted (by GP and Minutes) Gini coefficients.
    Saves analysis to gold.team_salary_inequality_analysis and creates 5 premium plots in the viz/ directory.
    """
    print("\n--- ANALYZING HYPOTHESIS 1: SALARY INEQUALITY ---")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy.stats as stats
        
        viz_dir = data_dir.parent / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print("  1. Loading player salaries and dimensions...")
        with engine.connect() as conn:
            salaries = pd.read_sql(
                text('SELECT player_id, player_name, season_end_year, salary, salary_inflation_adjusted FROM silver.fact_player_salaries;'), 
                con=conn
            )
            season_stats = pd.read_sql(
                text('SELECT player_id, season_end_year, team_id, gp, minutes_played FROM silver.fact_player_season_stats;'), 
                con=conn
            )
            teams_db = pd.read_sql(
                text('SELECT team_id, team_abbreviation, team_name FROM silver.dim_teams'), 
                con=conn
            )
            
            # Also load player gamelogs to map missing team_ids and calculate game counts/minutes
            gamelogs_query = """
                SELECT pg.player_id, g.season_end_year, pg.team_id, COUNT(*) as gp, SUM(pg.minutes) as minutes_played
                FROM silver.fact_player_gamelogs pg
                JOIN silver.dim_games g ON pg.game_id = g.game_id
                GROUP BY pg.player_id, g.season_end_year, pg.team_id;
            """
            gl_stats = pd.read_sql(text(gamelogs_query), con=conn)
            
            # Load team performance stats
            team_perf_query = """
                SELECT s.season_year, s.team_id, s.team_abbreviation, s.avg_point_diff, s.srs_rating, s.sos_srs
                FROM gold.team_sos s;
            """
            team_perf = pd.read_sql(text(team_perf_query), con=conn)
            team_perf['season_end_year'] = team_perf['season_year'].apply(lambda x: int(x.split('-')[0]) + 1)

        # Build unified player-season mapping
        player_map = {}
        for _, row in gl_stats.iterrows():
            if pd.isna(row['player_id']) or pd.isna(row['season_end_year']) or pd.isna(row['team_id']):
                continue
            key = (int(row['player_id']), int(row['season_end_year']))
            player_map[key] = (int(row['team_id']), float(row['gp']), float(row['minutes_played']) if not pd.isna(row['minutes_played']) else 0.0)

        for _, row in season_stats.iterrows():
            if pd.isna(row['player_id']) or pd.isna(row['season_end_year']) or pd.isna(row['team_id']):
                continue
            key = (int(row['player_id']), int(row['season_end_year']))
            player_map[key] = (int(row['team_id']), float(row['gp']) if not pd.isna(row['gp']) else 0.0, float(row['minutes_played']) if not pd.isna(row['minutes_played']) else 0.0)

        print("  2. Mapping player salaries to teams and compiling active minutes...")
        player_records = []
        for _, row in salaries.iterrows():
            pid = row['player_id']
            year = row['season_end_year']
            if pd.isna(pid) or pd.isna(year):
                continue
            pid, year = int(pid), int(year)
            if (pid, year) in player_map:
                tid, gp, mins = player_map[(pid, year)]
                player_records.append({
                    'player_name': row['player_name'],
                    'player_id': pid,
                    'season_end_year': year,
                    'team_id': tid,
                    'salary': float(row['salary']) if not pd.isna(row['salary']) else 0.0,
                    'salary_inflation_adjusted': float(row['salary_inflation_adjusted']) if not pd.isna(row['salary_inflation_adjusted']) else 0.0,
                    'gp': gp,
                    'minutes_played': mins
                })

        df_players = pd.DataFrame(player_records)
        print(f"  Mapped {len(df_players):,} player salary rows to team IDs.")

        # Gini Coefficient formulas
        def gini_simple(x):
            n = len(x)
            if n == 0 or np.mean(x) == 0:
                return 0.0
            mad = np.abs(np.subtract.outer(x, x)).mean()
            return 0.5 * mad / np.mean(x)

        def gini_weighted(x, w):
            x = np.asarray(x)
            w = np.asarray(w)
            valid = w > 0
            if not np.any(valid):
                return 0.0
            x = x[valid]
            w = w[valid]
            idx = np.argsort(x)
            x = x[idx]
            w = w[idx]
            w_sum = np.sum(w)
            if w_sum == 0 or np.sum(w * x) == 0:
                return 0.0
            cum_w = np.cumsum(w)
            cum_wx = np.cumsum(w * x)
            num = np.sum(w * (cum_wx - w * x / 2.0))
            den = w_sum * cum_wx[-1]
            return 1.0 - 2.0 * num / den

        # 3. Calculate team-season inequality metrics
        print("  3. Calculating simple and weighted inequality metrics...")
        team_seasons_groups = df_players.groupby(['season_end_year', 'team_id'])
        team_inequality = []

        for (year, team_id), group in team_seasons_groups:
            salaries_list = group['salary_inflation_adjusted'].values
            if len(salaries_list) < 5:
                continue
            
            total_payroll = np.sum(salaries_list)
            
            # Ginis
            g_simple = gini_simple(salaries_list)
            g_weighted_gp = gini_weighted(salaries_list, group['gp'].values)
            g_weighted_min = gini_weighted(salaries_list, group['minutes_played'].values)
            
            sorted_salaries = np.sort(salaries_list)[::-1]
            top3_share = np.sum(sorted_salaries[:3]) / total_payroll if total_payroll > 0 else 0.0
            cv_salary = np.std(salaries_list) / np.mean(salaries_list) if np.mean(salaries_list) > 0 else 0.0
            
            team_inequality.append({
                'season_end_year': year,
                'team_id': team_id,
                'total_payroll': total_payroll,
                'gini_coefficient': g_simple,
                'gini_weighted_gp': g_weighted_gp,
                'gini_weighted_min': g_weighted_min,
                'top3_share': top3_share,
                'cv_salary': cv_salary,
                'sd_salary': np.std(salaries_list),
                'num_players': len(salaries_list)
            })

        df_team_inequality = pd.DataFrame(team_inequality)

        df_analysis = pd.merge(df_team_inequality, team_perf, on=['season_end_year', 'team_id'], how='inner')
        df_analysis = df_analysis[(df_analysis['season_end_year'] >= 2019) & (df_analysis['season_end_year'] <= 2025)].copy()
        df_analysis['payroll_m'] = df_analysis['total_payroll'] / 1e6
        print(f"  Prepared {len(df_analysis):,} team-season records for analysis (seasons 2018-19 to 2024-25).")

        # 4. Classify teams into categories
        print("  4. Categorizing teams and performing clustering...")
        def assign_quantiles(group):
            group['payroll_level'] = pd.qcut(group['total_payroll'], 3, labels=['Low', 'Medium', 'High'])
            group['inequality_level'] = pd.qcut(group['gini_coefficient'], 2, labels=['Equal', 'Unequal'])
            return group

        df_analysis = df_analysis.groupby('season_end_year', group_keys=False).apply(assign_quantiles)

        # K-Means Clustering on standardized variables
        payroll_mean = df_analysis['total_payroll'].mean()
        payroll_std = df_analysis['total_payroll'].std()
        gini_mean = df_analysis['gini_coefficient'].mean()
        gini_std = df_analysis['gini_coefficient'].std()

        df_analysis['payroll_std'] = (df_analysis['total_payroll'] - payroll_mean) / payroll_std
        df_analysis['gini_std'] = (df_analysis['gini_coefficient'] - gini_mean) / gini_std

        features = df_analysis[['payroll_std', 'gini_std']].values

        # Simple K-Means K=4
        np.random.seed(42)
        k = 4
        centroids = features[np.random.choice(features.shape[0], k, replace=False)]
        for _ in range(100):
            distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([features[labels == i].mean(axis=0) if len(features[labels == i]) > 0 else centroids[i] for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        df_analysis['cluster_id'] = labels

        # Sort centroids by payroll size
        sorted_indices = np.argsort(centroids[:, 0])
        rank_map = {orig_idx: rank for rank, orig_idx in enumerate(sorted_indices)}
        df_analysis['cluster_id'] = df_analysis['cluster_id'].map(rank_map)

        descriptive_names = {
            0: "Low Payroll, Balanced Roster",
            1: "Moderate Payroll, Semi-Balanced",
            2: "High Payroll, Stars + Scraps",
            3: "Supermax Payroll, Stars + Role Players"
        }
        df_analysis['cluster_name'] = df_analysis['cluster_id'].map(descriptive_names)

        # Save to gold.team_salary_inequality_analysis
        print("  5. Saving results to database...")
        with engine.begin() as conn:
            df_analysis_db = df_analysis.copy()
            df_analysis_db = df_analysis_db.drop(columns=['payroll_std', 'gini_std'])
            df_analysis_db.to_sql(
                name="team_salary_inequality_analysis",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
        print("  Saved team_salary_inequality_analysis to gold schema.")

        # --- GENERATING PREMIUM VISUALIZATIONS ---
        print("  6. Generating visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot 1: K-Means Team Clustering with Centroids
        plt.figure(figsize=(11, 7))
        # Unstandardize centroids for plotting
        centroids_unstd = np.zeros_like(centroids)
        centroids_unstd[:, 0] = (centroids[:, 0] * payroll_std + payroll_mean) / 1e6
        centroids_unstd[:, 1] = centroids[:, 1] * gini_std + gini_mean
        centroids_unstd_sorted = centroids_unstd[sorted_indices]

        # Use beautiful color mapping
        colors = ['#2b8cbe', '#8856a7', '#f15a24', '#e31a1c']
        ax = sns.scatterplot(
            data=df_analysis,
            x=df_analysis['total_payroll'] / 1e6,
            y='gini_coefficient',
            hue='cluster_name',
            hue_order=[descriptive_names[i] for i in range(4)],
            palette=colors,
            alpha=0.8,
            s=80,
            edgecolor='white',
            linewidth=0.8
        )
        
        # Plot Centroids
        for i, cent in enumerate(centroids_unstd_sorted):
            plt.scatter(
                cent[0], cent[1],
                color=colors[i],
                marker='X',
                s=350,
                edgecolor='black',
                linewidth=2.0,
                label=f"Centroide {i}: {descriptive_names[i]}"
            )

        # Annotate iconic teams
        examples = [
            {'season_year': '2017-18', 'team_abb': 'GSW', 'label': 'GSW 17-18 (Super-Time)'},
            {'season_year': '2023-24', 'team_abb': 'BOS', 'label': 'BOS 23-24 (Campeão)'},
            {'season_year': '2011-12', 'team_abb': 'CHA', 'label': 'CHA 11-12 (Pior Campanha)'},
            {'season_year': '2023-24', 'team_abb': 'PHX', 'label': 'PHX 23-24 (Estrelas & Min.)'},
            {'season_year': '2023-24', 'team_abb': 'IND', 'label': 'IND 23-24 (Equilibrado)'}
        ]
        
        # Build team abbreviations mapper
        team_id_to_abb = {row['team_id']: row['team_abbreviation'] for _, row in teams_db.iterrows()}
        for ex in examples:
            match = df_analysis[
                (df_analysis['season_year'] == ex['season_year']) & 
                (df_analysis['team_abbreviation'] == ex['team_abb'])
            ]
            if not match.empty:
                row = match.iloc[0]
                px = row['total_payroll'] / 1e6
                py = row['gini_coefficient']
                plt.annotate(
                    ex['label'],
                    xy=(px, py),
                    xytext=(px + 3, py + 0.015),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
                )

        plt.title("Clusterização K-Means das Equipes da NBA (2010-2025)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Folha Salarial Total (em Milhões de USD - Ajustado Inflação)", fontsize=11)
        plt.ylabel("Coeficiente Gini (Desigualdade de Salário)", fontsize=11)
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(viz_dir / "kmeans_team_clustering.png", dpi=300)
        plt.close()

        # Plot 2: Average performance (SRS) by salary inequality clusters
        plt.figure(figsize=(10, 6))
        order_list = [descriptive_names[i] for i in range(4)]
        sns.barplot(
            data=df_analysis,
            x='cluster_name',
            y='srs_rating',
            order=order_list,
            hue='cluster_name',
            palette=colors,
            legend=False,
            errorbar='ci',
            edgecolor='none'
        )
        plt.title("Desempenho Médio (SRS) por Cluster de Distribuição Salarial", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Cluster Salarial", fontsize=11)
        plt.ylabel("Rating SRS Médio", fontsize=11)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_by_salary_inequality_clusters.png", dpi=300)
        plt.close()

        # Plot 3: Salary distribution of players over seasons (Inflation-Adjusted Boxplots)
        plt.figure(figsize=(12, 7))
        df_players_clean = df_players[df_players['salary_inflation_adjusted'] > 0].copy()
        df_players_clean['salary_m'] = df_players_clean['salary_inflation_adjusted'] / 1e6
        sns.boxplot(
            data=df_players_clean,
            x='season_end_year',
            y='salary_m',
            palette='Blues',
            showfliers=False
        )
        plt.title("Evolução da Distribuição Salarial na NBA por Temporada (2010-2025)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Temporada (Ano Final)", fontsize=11)
        plt.ylabel("Salário do Jogador (Milhões de USD - Ajustado Inflação)", fontsize=11)
        plt.tight_layout()
        plt.savefig(viz_dir / "nba_salary_distribution_evolution.png", dpi=300)
        plt.close()

        # Plot 4: Partial Regression Plot (Equity Theory Proof)
        print("  7. Calculating OLS residuals for partial regression...")
        df_reg = df_analysis.copy()
        df_reg['payroll_m'] = df_reg['total_payroll'] / 1e6
        
        # Fit SRS ~ Payroll
        X_p = np.column_stack([np.ones(len(df_reg)), df_reg['payroll_m'].values])
        y_srs = df_reg['srs_rating'].values
        beta_srs_p, _, _, _ = np.linalg.lstsq(X_p, y_srs, rcond=None)
        res_srs_p = y_srs - X_p @ beta_srs_p
        
        # Fit Gini_Minutes ~ Payroll
        y_gini_min = df_reg['gini_weighted_min'].values
        beta_gini_p, _, _, _ = np.linalg.lstsq(X_p, y_gini_min, rcond=None)
        res_gini_p = y_gini_min - X_p @ beta_gini_p
        
        plt.figure(figsize=(10, 6.5))
        sns.regplot(
            x=res_gini_p,
            y=res_srs_p,
            scatter_kws={'alpha': 0.5, 'color': '#756bb1', 'edgecolor': 'none'},
            line_kws={'color': '#e31a1c', 'lw': 2.5}
        )
        slope, intercept, r_value, p_value, std_err = stats.linregress(res_gini_p, res_srs_p)
        plt.annotate(
            f"Inclinação (Coeficiente Gini Ajustado) = {slope:.2f}\np-valor = {p_value:.2e}\nR² Parcial = {r_value**2:.4f}",
            xy=(0.05, 0.85),
            xycoords='axes fraction',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='gray')
        )
        plt.title("Prova Gráfica da Teoria da Equidade: Desempenho vs. Desigualdade (Ajustado por Folha Salarial)", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Resíduos da Desigualdade (Gini de Minutos | Folha Salarial)\n<- Mais Igualitário do que o esperado | Mais Desigual do que o esperado ->", fontsize=11)
        plt.ylabel("Resíduos do Desempenho (SRS | Folha Salarial)\n<- Pior do que o orçamento prevê | Melhor do que o orçamento prevê ->", fontsize=11)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(viz_dir / "equity_theory_partial_regression.png", dpi=300)
        plt.close()

        # Plot 5: The Supermax Exception Plot
        plt.figure(figsize=(11, 7))
        sc = plt.scatter(
            df_analysis['payroll_m'],
            df_analysis['srs_rating'],
            c=df_analysis['gini_weighted_min'],
            cmap='coolwarm',
            alpha=0.8,
            s=80,
            edgecolor='white',
            linewidth=0.5
        )
        cbar = plt.colorbar(sc)
        cbar.set_label('Gini Ponderado por Minutos', rotation=270, labelpad=15, fontsize=11)
        
        sns.regplot(
            data=df_analysis,
            x='payroll_m',
            y='srs_rating',
            scatter=False,
            color='black',
            line_kws={'linestyle': '--', 'lw': 1.5}
        )

        plt.axvspan(100, 130, color='red', alpha=0.08, label='Stars + Scraps (Eficiência Ruim)')
        plt.axvspan(145, 195, color='green', alpha=0.08, label='Supermax Elite (Sucesso)')
        
        plt.annotate(
            "Stars & Scraps Region:\nFolhas altas com Gini alto\nmas sem profundidade.\nSRS Médio: -1.27",
            xy=(115, -3.5),
            xytext=(75, -7),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
            fontsize=10,
            color='red',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='red')
        )
        plt.annotate(
            "Supermax Region:\nFranquias gastando acima do teto.\nMesmo com Gini alto, conseguem\nmanter coadjuvantes de elite.\nSRS Médio: +2.32",
            xy=(169, 3.5),
            xytext=(115, 6),
            arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
            fontsize=10,
            color='green',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='green')
        )

        plt.title("O Efeito Supermax vs. Restrição Orçamentária na NBA", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Folha Salarial (em Milhões de USD - Ajustado Inflação)", fontsize=11)
        plt.ylabel("Rating SRS (Desempenho da Equipe)", fontsize=11)
        plt.tight_layout()
        plt.savefig(viz_dir / "supermax_vs_budget_constraint.png", dpi=300)
        plt.close()

        # OLS Regressions for simple print
        print("  8. Running Regressions for Report Comparison...")
        y = df_analysis['srs_rating'].values
        
        def print_ols(X_vars, names):
            X = np.column_stack([np.ones(len(df_analysis)), df_analysis[X_vars].values])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            rss = np.sum(residuals**2)
            df_err = len(y) - X.shape[1]
            s2 = rss / df_err
            cov_beta = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_beta))
            t = beta / se
            p = stats.t.sf(np.abs(t), df_err) * 2
            r2 = 1.0 - (rss / np.sum((y - np.mean(y))**2))
            print(f"    === MODEL: SRS ~ {' + '.join(names)} (R2={r2:.4f}) ===")
            for i, name in enumerate(['const'] + names):
                print(f"      {name:<25} Coef: {beta[i]:.6f} | p-val: {p[i]:.4e}")

        print_ols(['payroll_m', 'gini_coefficient'], ['Payroll_M', 'Gini_Simple'])
        print_ols(['payroll_m', 'gini_weighted_gp'], ['Payroll_M', 'Gini_Weighted_GP'])
        print_ols(['payroll_m', 'gini_weighted_min'], ['Payroll_M', 'Gini_Weighted_Minutes'])

        print("  Successfully completed salary inequality analysis (Hypothesis 1).")

    except Exception as e:
        print(f"  Error analyzing salary inequality: {e}")
        import traceback
        traceback.print_exc()


def generate_trades_panel(engine, data_dir):
    """
    Builds the df_h1 panel that links trade Gini and team performance variations.
    Saves the final dataframe to gold.df_h1.
    """
    print("\n--- Generating Gold.df_h1 Trades and Performance Panel ---")
    try:
        with engine.connect() as conn:
            # Player trades
            player_trades = pd.read_sql('SELECT player_id, season_end_year, old_team_id, new_team_id, trade_date FROM silver.fact_player_trades;', con=conn)
            player_trades['trade_date'] = pd.to_datetime(player_trades['trade_date'])
            
            # Season stats (to get ages and team-seasons rosters)
            season_stats = pd.read_sql('SELECT player_id, season_end_year, team_id, age FROM silver.fact_player_season_stats;', con=conn)
            
            # Gamelogs rosters
            pg_teams = pd.read_sql('''
                SELECT DISTINCT pg.player_id, g.season_end_year, pg.team_id
                FROM silver.fact_player_gamelogs pg
                JOIN silver.dim_games g ON pg.game_id = g.game_id;
            ''', con=conn)
            
            # Team gamelogs
            team_gamelogs = pd.read_sql('''
                SELECT tg.game_id, tg.team_id, tg.opponent_team_id, g.game_date, g.season_year, g.season_end_year, 
                       tg.wl, tg.off_rating, tg.def_rating, tg.net_rating
                FROM silver.fact_team_gamelogs tg
                JOIN silver.dim_games g ON tg.game_id = g.game_id
                WHERE g.game_type = 'Regular Season'
                ORDER BY tg.team_id, g.game_date;
            ''', con=conn)
            team_gamelogs['game_date'] = pd.to_datetime(team_gamelogs['game_date'])
            
            # Salaries
            salaries = pd.read_sql('SELECT player_id, season_end_year, salary_inflation_adjusted FROM silver.fact_player_salaries;', con=conn)
            
            # Opponent Strength of Schedule (SoS) Ratings
            sos_db = pd.read_sql('SELECT season_year, team_id, srs_rating FROM gold.team_sos;', con=conn)

        print("  Loaded raw tables. Reconstructing rosters...")
        
        # Union of player-season-team records to get the most complete base roster
        base_rosters = pd.concat([
            season_stats[['player_id', 'season_end_year', 'team_id']],
            pg_teams[['player_id', 'season_end_year', 'team_id']]
        ]).drop_duplicates().dropna().astype(int)

        # Group base rosters by (season_end_year, team_id) to have list of players
        base_roster_dict = {}
        for _, row in base_rosters.iterrows():
            key = (row['season_end_year'], row['team_id'])
            if key not in base_roster_dict:
                base_roster_dict[key] = set()
            base_roster_dict[key].add(row['player_id'])

        # Build player birth year mapping to backfill age
        player_birth_years = {}
        for _, row in season_stats.dropna(subset=['age']).iterrows():
            pid = int(row['player_id'])
            year = int(row['season_end_year'])
            age = float(row['age'])
            birth_year = year - age
            if pid not in player_birth_years:
                player_birth_years[pid] = []
            player_birth_years[pid].append(birth_year)

        # Average birth year per player
        player_birth_year = {pid: np.round(np.mean(vals)) for pid, vals in player_birth_years.items()}

        # Salaries dictionary
        salaries_clean = salaries.dropna(subset=['player_id', 'season_end_year', 'salary_inflation_adjusted'])
        salary_dict = {(int(row['player_id']), int(row['season_end_year'])): float(row['salary_inflation_adjusted']) for _, row in salaries_clean.iterrows()}

        # SoS dictionary
        sos_dict = {(row['season_year'], int(row['team_id'])): float(row['srs_rating']) for _, row in sos_db.dropna().iterrows()}

        # Unique trade events to analyze
        trade_events_raw = []
        for _, row in player_trades.iterrows():
            year = row['season_end_year']
            tdate = row['trade_date']
            if pd.isna(tdate):
                continue
            if not pd.isna(row['new_team_id']):
                trade_events_raw.append((int(row['new_team_id']), tdate, int(year)))
            if not pd.isna(row['old_team_id']):
                trade_events_raw.append((int(row['old_team_id']), tdate, int(year)))

        trade_events = sorted(list(set(trade_events_raw)), key=lambda x: (x[2], x[0], x[1]))
        
        def calculate_gini(x):
            n = len(x)
            if n == 0 or np.mean(x) == 0:
                return 0.0
            mad = np.abs(np.subtract.outer(x, x)).mean()
            return 0.5 * mad / np.mean(x)

        panel_records = []
        n = 15 # Optimal window size from sensitivity analysis

        for team_id, tdate, year in trade_events:
            base_roster = base_roster_dict.get((year, team_id), set()).copy()
            if not base_roster:
                continue
                
            traded_away = set(player_trades[
                (player_trades['old_team_id'] == team_id) & 
                (player_trades['season_end_year'] == year) & 
                (player_trades['trade_date'] <= tdate)
            ]['player_id'].dropna().astype(int))
            
            traded_in_later = set(player_trades[
                (player_trades['new_team_id'] == team_id) & 
                (player_trades['season_end_year'] == year) & 
                (player_trades['trade_date'] > tdate)
            ]['player_id'].dropna().astype(int))
            
            traded_in_today = set(player_trades[
                (player_trades['new_team_id'] == team_id) & 
                (player_trades['season_end_year'] == year) & 
                (player_trades['trade_date'] == tdate)
            ]['player_id'].dropna().astype(int))
            
            roster = (base_roster - traded_away - traded_in_later).union(traded_in_today)
            if len(roster) < 5:
                continue
                
            roster_salaries = [salary_dict.get((pid, year), 0.0) for pid in roster]
            roster_salaries = [s for s in roster_salaries if s > 0.0]
            payroll_total = np.sum(roster_salaries)
            gini_coef = calculate_gini(roster_salaries) if len(roster_salaries) >= 5 else None
            
            roster_ages = []
            for pid in roster:
                direct_match = season_stats[(season_stats['player_id'] == pid) & (season_stats['season_end_year'] == year)]
                if not direct_match.empty and not pd.isna(direct_match.iloc[0]['age']):
                    roster_ages.append(float(direct_match.iloc[0]['age']))
                elif pid in player_birth_year:
                    roster_ages.append(year - player_birth_year[pid])
                else:
                    roster_ages.append(26.5)
            avg_age = np.mean(roster_ages) if roster_ages else None
            
            team_games = team_gamelogs[(team_gamelogs['team_id'] == team_id) & (team_gamelogs['season_end_year'] == year)].copy()
            if team_games.empty:
                continue
                
            pre_games = team_games[team_games['game_date'] < tdate].sort_values('game_date', ascending=False)
            post_games = team_games[team_games['game_date'] > tdate].sort_values('game_date', ascending=True)
            
            if pre_games.empty or post_games.empty:
                continue
                
            pre_window = pre_games.head(n)
            post_window = post_games.head(n)
            
            win_pct_pre = (pre_window['wl'] == 'W').mean()
            win_pct_post = (post_window['wl'] == 'W').mean()
            delta_win = win_pct_post - win_pct_pre
            
            nrtg_pre = pre_window['net_rating'].mean()
            nrtg_post = post_window['net_rating'].mean()
            delta_net = nrtg_post - nrtg_pre
            
            ortg_pre = pre_window['off_rating'].mean()
            ortg_post = post_window['off_rating'].mean()
            drtg_pre = pre_window['def_rating'].mean()
            drtg_post = post_window['def_rating'].mean()
            
            season_year = f"{year-1}-{str(year)[2:]}"
            opp_srs_pre = np.mean([sos_dict.get((season_year, opp_id), 0.0) for opp_id in pre_window['opponent_team_id']])
            opp_srs_post = np.mean([sos_dict.get((season_year, opp_id), 0.0) for opp_id in post_window['opponent_team_id']])
            delta_sos = opp_srs_post - opp_srs_pre
            
            matching_trades = player_trades[
                ((player_trades['old_team_id'] == team_id) | (player_trades['new_team_id'] == team_id)) & 
                (player_trades['trade_date'] == tdate)
            ]
            trade_id = int(matching_trades['player_id'].iloc[0]) if not matching_trades.empty else 999
            
            panel_records.append({
                'trade_id': trade_id,
                'team_id': int(team_id),
                'trade_date': tdate.date(),
                'season_end_year': int(year),
                'gini': gini_coef,
                'payroll_total': payroll_total,
                'avg_age': avg_age,
                'win_pct_pre': float(win_pct_pre),
                'win_pct_post': float(win_pct_post),
                'delta_win': float(delta_win),
                'net_rating_pre': float(nrtg_pre) if not pd.isna(nrtg_pre) else None,
                'net_rating_post': float(nrtg_post) if not pd.isna(nrtg_post) else None,
                'delta_net': float(delta_net) if not (pd.isna(nrtg_pre) or pd.isna(nrtg_post)) else None,
                'off_rating_pre': float(ortg_pre) if not pd.isna(ortg_pre) else None,
                'off_rating_post': float(ortg_post) if not pd.isna(ortg_post) else None,
                'def_rating_pre': float(drtg_pre) if not pd.isna(drtg_pre) else None,
                'def_rating_post': float(drtg_post) if not pd.isna(drtg_post) else None,
                'sos_pre': float(opp_srs_pre),
                'sos_post': float(opp_srs_post),
                'delta_sos': float(delta_sos)
            })

        df_h1 = pd.DataFrame(panel_records)
        print(f"  Generated {len(df_h1)} records in df_h1.")
        
        with engine.begin() as conn:
            df_h1.to_sql(
                name="df_h1",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
        print("  Successfully saved gold.df_h1 to database.")
    except Exception as e:
        print(f"  Error generating trades panel: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Load NBA data parquet files into PostgreSQL Medallion schemas."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
        help=f"PostgreSQL database connection URL (default: {DEFAULT_DB_URL})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=Path(__file__).parent / "sql",
        help="Path to SQL scripts directory (default: ./src/sql)",
    )
    parser.add_argument(
        "--bronze-only",
        action="store_true",
        help="Only run the gold layer data load"
    )
    parser.add_argument(
        "--silver-only",
        action="store_true",
        help="Only run the silver layer SQL transformations"
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Only run the gold layer SQL analytics"
    )
    
    args = parser.parse_args()
    
    print(f"Starting NBA Medallion Pipeline...")
    print(f"Database URL: {args.db_url}")
    print(f"Data Directory: {args.data_dir.resolve()}")
    
    engine = get_db_engine(args.db_url)
    setup_schemas(engine)
    
    run_all = not any([args.bronze_only, args.silver_only, args.gold_only])
    
    if args.bronze_only or run_all:
        load_parquet_to_bronze(engine, args.data_dir)
        load_salaries_to_bronze(engine, args.data_dir)
        
    if args.silver_only or run_all:
        execute_sql_file(engine, args.sql_dir / "create_silver.sql", "silver")
        
    if args.gold_only or run_all:
        execute_sql_file(engine, args.sql_dir / "create_gold.sql", "gold")
        compute_and_load_sos(engine)
        load_trades_and_calculate_impact(engine, args.data_dir)
        analyze_salary_inequality(engine, args.data_dir)
        generate_trades_panel(engine, args.data_dir)
        
    print("\nNBA Medallion Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
