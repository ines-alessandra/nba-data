#!/usr/bin/env python3
"""
NBA Mid-Season Trade Activity Builder
Author: Data Engineer

Parses the real trade-transaction ledger in data/trades/trades_{YYYY}.csv and
builds a per-team-season summary of how much a team actually transacted in
the run-up to that season's real trade deadline (see trade_deadline_dates.py).

This replaces the old design's implicit assumption that "pré" and "pós"
windows around a calendar date meant something for every team, even teams
that made zero trades. Here, trade activity is measured directly instead of
inferred, which lets downstream analysis (a) restrict to teams that actually
had a trade-deadline event, and (b) use trade magnitude (players moved) as
an explicit variable instead of a binary before/after split.

Season assignment: each row is assigned to whichever season_end_year's
30-day pre-deadline window its date falls into (see
restrict_to_deadline_window), NOT to a season inferred from the filename.
This matters because trades_2018.csv is a "double-wide" file (2017-07-04 to
2019-06-26) -- there is no trades_2017.csv, so the 2017-18 season's
transactions were folded into the 2018 file alongside 2018-19's. Deriving
season purely from each row's date, independent of which file it came from,
handles this correctly without special-casing that one file.

Source file quirks handled:
- Some rows have a trailing stray comma (14-column header, occasional
  15-column rows) -- read with the csv module and pad/truncate to the header
  width instead of relying on pandas' C parser, which raises on this.
- A handful of historical/renamed franchise names appear (e.g. "New Orleans
  Pelicans" vs. the "New Orleans Hornets" name still stored in
  silver.dim_teams for that franchise's team_id) -- resolved via
  TEAM_NAME_OVERRIDES.
- The 2011-12 lockout season is out of scope entirely (see
  trade_deadline_dates.py): trade activity in its 30-day pre-deadline window
  is essentially empty leaguewide, so it's excluded upstream by
  trade_deadline_dates.TRADE_DEADLINES simply not having a 2012 entry.

Output: data/team_trade_activity.csv with one row per team-season that had
at least one qualifying transaction, columns:
  team_id, season_end_year, n_trades, n_players_in, n_players_out,
  n_players_moved (= in + out)
"""

import csv
import glob
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from trade_deadline_dates import TRADE_DEADLINES

DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# How many days before the real deadline to start counting "deadline activity".
# 30 days captures the trade-deadline build-up window without pulling in
# unrelated offseason/summer transactions from earlier in the same file.
WINDOW_DAYS_BEFORE_DEADLINE = 30

# Franchise names in the trade ledger that don't match silver.dim_teams.team_name
TEAM_NAME_OVERRIDES = {
    'New Orleans Pelicans': 'New Orleans Hornets',  # same team_id, dim_teams kept the old name
    'Charlotte Bobcats': 'Charlotte Hornets',        # same team_id, pre-2014 rename
    'Washington Bullets': 'Washington Wizards',      # same team_id, pre-1997 rename
}


def read_trade_csv(path: str) -> pd.DataFrame:
    """Read a trades_{YYYY}.csv file, tolerating ragged trailing commas."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        rows = []
        for r in reader:
            if len(r) > ncols:
                r = r[:ncols]
            elif len(r) < ncols:
                r = r + [''] * (ncols - len(r))
            rows.append(r)
    return pd.DataFrame(rows, columns=header)


def load_all_trades() -> pd.DataFrame:
    project_dir = Path(__file__).resolve().parent.parent
    frames = []
    for path in sorted(glob.glob(str(project_dir / "data" / "trades" / "trades_*.csv"))):
        frames.append(read_trade_csv(path))
    df_all = pd.concat(frames, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce', format='mixed')
    df_all['team'] = df_all['team'].replace(TEAM_NAME_OVERRIDES)
    return df_all.dropna(subset=['date'])


def restrict_to_deadline_window(df_all: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to whichever season_end_year's real deadline window
    [deadline - WINDOW_DAYS_BEFORE_DEADLINE, deadline] its date falls into,
    and drop rows that don't fall into any window. A row's source file does
    not determine its season -- only its date does (see module docstring for
    why: the double-wide trades_2018.csv file needs this to split correctly)."""
    windows = []
    for season_end_year, deadline_str in TRADE_DEADLINES.items():
        deadline = pd.Timestamp(deadline_str)
        start = deadline - pd.Timedelta(days=WINDOW_DAYS_BEFORE_DEADLINE)
        chunk = df_all[(df_all['date'] >= start) & (df_all['date'] <= deadline)].copy()
        chunk['season_end_year'] = season_end_year
        windows.append(chunk)
    return pd.concat(windows, ignore_index=True)


def summarize_trade_activity(df_window: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df_window.groupby(['team', 'season_end_year'])
    for (team, season_end_year), group in grouped:
        players = group[group['asset_type'] == 'Player']
        n_in = int((players['direction'] == 'Incoming').sum())
        n_out = int((players['direction'] == 'Outgoing').sum())
        records.append({
            'team': team,
            'season_end_year': season_end_year,
            'n_trades': int(group['trade_id'].nunique()),
            'n_players_in': n_in,
            'n_players_out': n_out,
            'n_players_moved': n_in + n_out,
        })
    return pd.DataFrame(records)


def attach_team_id(df_summary: pd.DataFrame, engine) -> pd.DataFrame:
    df_teams = pd.read_sql(text("SELECT DISTINCT team_id, team_name FROM silver.dim_teams"), engine)
    df_merged = pd.merge(df_summary, df_teams, left_on='team', right_on='team_name', how='left')

    missing = df_merged[df_merged['team_id'].isna()]['team'].unique()
    if len(missing) > 0:
        print(f"Warning: {len(missing)} team name(s) could not be matched to silver.dim_teams: {list(missing)}")

    df_merged = df_merged.dropna(subset=['team_id']).copy()
    df_merged['team_id'] = df_merged['team_id'].astype('int64')
    return df_merged[['team_id', 'team', 'season_end_year', 'n_trades', 'n_players_in', 'n_players_out', 'n_players_moved']]


def main():
    engine = create_engine(DB_URL)

    df_all = load_all_trades()
    print(f"Loaded {len(df_all)} trade-ledger rows spanning {df_all['date'].min().date()} to {df_all['date'].max().date()}.")

    df_window = restrict_to_deadline_window(df_all)
    print(f"{len(df_window)} rows fall within {WINDOW_DAYS_BEFORE_DEADLINE} days of each season's real trade deadline.")

    df_summary = summarize_trade_activity(df_window)
    df_final = attach_team_id(df_summary, engine)

    print(f"\nTeam-seasons with qualifying trade-deadline activity: {len(df_final)}")
    print(df_final.groupby('season_end_year').size().to_string())

    output_path = Path(__file__).resolve().parent.parent / "data" / "team_trade_activity.csv"
    df_final.sort_values(['season_end_year', 'team']).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
