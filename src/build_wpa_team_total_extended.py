#!/usr/bin/env python3
"""
SUPERSEDED by build_wpa_positional_analysis_extended.py, which now regenerates
data/wpa_team_total_extended.csv itself (computed from the same position-filtered
player population as RCP_pos's own denominator, matching the original ESPN-era
script's convention). Do NOT run this script standalone anymore -- it would
silently revert wpa_team_total_extended.csv to an ALL-PLAYERS sum, which was
consciously rejected: for the 2013-2018 rows specifically, that version differs
from the position-filtered one by up to +/-6.6 WPA (62.8% of those 300 team-season
rows move by more than 0.5), since inpredictable's Pos label is blank for ~19% of
players in 2013 down to ~7% by 2018 (see build_wpa_positional_analysis_extended.py's
docstring for the full picture). Kept in the repo only for its still-used
build_abbrev_crosswalk() helper (imported by build_wpa_positional_analysis_extended.py)
and as a record of the first (team-level-only) pass.

--- Original docstring below ---

Extends data/wpa_team_total.csv's team-season WPA_time coverage from 3 seasons
(2019-2021, ESPN tWPA) to all 13 seasons 2013-2025, by adding the 10 seasons
scraped from inpredictable.com (src/scrape_inpredictable_wpa_batch.py).

Only the TEAM-LEVEL part of the methodology (WPA_time = sum of per-player WPA
over the team's games that season) is extended here. The POSITIONAL breakdown
(RCP_pos/muP/deltaP from build_wpa_season_profile.py) is NOT extended, because
it depends on ESPN's per-game `dAvgPos` field -- a minutes-weighted average
defensive-matchup position with full-roster coverage. Neither inpredictable's
page nor this repo's other data has an equivalent full-roster per-game position
signal (nba.com's own boxscore `position` field is confirmed only ~42% filled --
starters only; see wpa_common.py's docstring). Extending the positional analysis
would require a genuinely different position signal (e.g. a static per-player
season/career position pulled separately) and is intentionally left out of this
script to avoid silently mixing two different position definitions across
periods.

Team-ID resolution: inpredictable's own team abbreviation column uses two
non-standard codes ("UTH" for Utah, "NO" for New Orleans) instead of nba.com's
("UTA", "NOH"/"NOP"). A first version of this script instead tried to resolve
team_id via (game_id, home/away side) against gamelog_<year>.parquet's `matchup`
text -- but that surfaced a real data-quality bug in gamelog_2025.parquet (5
games where BOTH team rows are recorded as "@", i.e. neither is flagged as
home, which would have silently double-counted WPA for those games via a
cartesian merge). Abbreviation->team_id is otherwise 1:1 stable across all 13
seasons in gamelog data (verified: no franchise abbreviation maps to more than
one team_id across 2013-2025), so a direct crosswalk built from gamelog's own
abbreviation->team_id map, plus the 2 known overrides, is both simpler and
avoids that per-game matchup-text fragility entirely.

Output: data/wpa_team_total_extended.csv -- same columns as wpa_team_total.csv
plus `source` ('espn' | 'inpredictable'), covering season_end_year 2013-2025.

Usage:
    python src/build_wpa_team_total_extended.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPREDICTABLE_DIR = DATA_DIR / "inpredictable"
ESPN_TEAM_TOTAL_PATH = DATA_DIR / "wpa_team_total.csv"
CLUSTERS_PATH = DATA_DIR / "team_season_clusters.csv"
OUT_PATH = DATA_DIR / "wpa_team_total_extended.csv"

INPREDICTABLE_SEASONS = [2013, 2014, 2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025]

# Overrides for the 2 inpredictable abbreviations confirmed (empirically, across all
# 10 scraped seasons) not to match nba.com's own gamelog abbreviations directly.
ABBREV_OVERRIDES = {"UTH": "UTA", "NO": "NOH"}


def build_abbrev_crosswalk() -> dict[str, int]:
    """Canonical team_abbreviation -> team_id, built from gamelog data across every
    season in scope (verified 1:1 stable, no franchise-abbreviation collisions),
    plus the known inpredictable-specific overrides."""
    canon: dict[str, int] = {}
    for season_end_year in INPREDICTABLE_SEASONS:
        gl = pd.read_parquet(
            DATA_DIR / f"gamelog_{season_end_year}.parquet",
            columns=["team_id", "team_abbreviation"],
        ).drop_duplicates()
        for abbrev, team_id in zip(gl["team_abbreviation"], gl["team_id"]):
            if abbrev in canon and canon[abbrev] != team_id:
                raise RuntimeError(
                    f"Abbreviation '{abbrev}' maps to multiple team_ids across seasons "
                    f"({canon[abbrev]} vs {team_id}) -- crosswalk assumption violated, "
                    f"needs season-aware resolution."
                )
            canon[abbrev] = team_id
    for ip_abbrev, canon_abbrev in ABBREV_OVERRIDES.items():
        canon[ip_abbrev] = canon[canon_abbrev]
    return canon


def team_total_for_season(
    season_end_year: int, crosswalk: dict[str, int], canonical_abbrev: dict[int, str]
) -> pd.DataFrame:
    pb = pd.read_parquet(INPREDICTABLE_DIR / f"wpa_player_box_{season_end_year}.parquet")

    pb["team_id"] = pb["team"].map(crosswalk)
    unresolved = pb["team_id"].isna()
    if unresolved.any():
        bad = sorted(pb.loc[unresolved, "team"].unique())
        raise RuntimeError(
            f"[{season_end_year}] {unresolved.sum()} player rows have an unmapped team "
            f"abbreviation: {bad} -- add to ABBREV_OVERRIDES."
        )
    pb["team_id"] = pb["team_id"].astype("int64")

    team_total = pb.groupby("team_id").agg(
        WPA_time=("WPA", "sum"),
        games_played=("game_id", "nunique"),
    ).reset_index()
    # Label with nba.com's canonical abbreviation, not inpredictable's raw text
    # (which is "UTH"/"NO" for 2 franchises -- see ABBREV_OVERRIDES).
    team_total["team_abbreviation"] = team_total["team_id"].map(canonical_abbrev)
    team_total["season_end_year"] = season_end_year
    return team_total


def main() -> int:
    import sys as _sys
    if "--force" not in _sys.argv:
        print("This script is SUPERSEDED by build_wpa_positional_analysis_extended.py "
              "(see this file's module docstring). Running it would overwrite "
              "data/wpa_team_total_extended.csv with the ALL-PLAYERS total, which was "
              "consciously rejected in favor of the position-filtered version. "
              "Re-run with --force if you really mean to do that.", file=_sys.stderr)
        return 1

    print("=" * 70)
    print("  WPA Team Total -- Extended (2013-2025)")
    print("=" * 70)

    clusters = pd.read_csv(CLUSTERS_PATH, usecols=["team_id", "season_end_year", "cluster"])

    espn = pd.read_csv(ESPN_TEAM_TOTAL_PATH)
    espn["source"] = "espn"
    print(f"Loaded {len(espn)} ESPN-sourced team-season rows "
          f"(seasons {sorted(espn['season_end_year'].unique())}).")

    crosswalk = build_abbrev_crosswalk()
    canonical_abbrev = {v: k for k, v in crosswalk.items() if k not in ABBREV_OVERRIDES}

    inpredictable_parts = []
    for season_end_year in INPREDICTABLE_SEASONS:
        season_total = team_total_for_season(season_end_year, crosswalk, canonical_abbrev)
        inpredictable_parts.append(season_total)
        print(f"  [{season_end_year}] {len(season_total)} teams, "
              f"WPA_time range [{season_total['WPA_time'].min():.2f}, "
              f"{season_total['WPA_time'].max():.2f}]")

    inpredictable = pd.concat(inpredictable_parts, ignore_index=True)
    inpredictable = inpredictable.merge(clusters, on=["team_id", "season_end_year"], how="left")
    inpredictable["source"] = "inpredictable"

    combined = pd.concat([
        espn[["team_id", "team_abbreviation", "season_end_year", "WPA_time", "games_played", "cluster", "source"]],
        inpredictable[["team_id", "team_abbreviation", "season_end_year", "WPA_time", "games_played", "cluster", "source"]],
    ], ignore_index=True)
    combined = combined.sort_values(["season_end_year", "team_abbreviation"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(combined)} team-season rows to {OUT_PATH}")
    print(f"Seasons covered: {sorted(combined['season_end_year'].unique())}")
    print(f"Source breakdown:\n{combined['source'].value_counts().to_string()}")
    n_null_cluster = combined["cluster"].isna().sum()
    if n_null_cluster:
        print(f"NOTE: {n_null_cluster} rows have no cluster assignment (missing from {CLUSTERS_PATH.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
