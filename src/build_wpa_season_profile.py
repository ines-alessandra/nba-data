#!/usr/bin/env python3
"""
Builds the season-level WPA positional profile, following the exact formula chain
from Sa (2025), "Contribuicao Relativa: Detectando Buracos de Elenco na NBA"
(Kaio-Lucas-de-Sa-2019006850-Contribuicao-Relativa.pdf), section III (Metodologia):

  WPA_P    = sum of WPA_acc(j) for players j in position group P
  RCP_P    = WPA_P / WPA_acc(time)                    (section III-C)
  RCP_bar_P = RCP_P / N,  N = team's total games that season/window  (III-C, last eq.)
  muP      = (1/K) * sum_k(RCP_bar_P for team k), K=30 teams          (III-D)
  deltaP   = RCP_bar_P(team) - muP                                    (III-D)

for every (team, season) in 2019-2021 (the years with real ESPN WPA data).

Outputs:
  data/wpa_team_total.csv        - one row per team-season (WPA_time, MIN_time)
  data/wpa_positional_analysis.csv - one row per team-season-position
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import (  # noqa: E402
    START_YEAR, END_YEAR, POSITION_GROUPS, MIN_WPA_ACC_FOR_STABLE_RATIO, PROJECT_DIR,
    get_engine, load_espn_player_box, assign_positions, compute_wpa_acc,
)

DEFAULT_TEAM_TOTAL_PATH = PROJECT_DIR / "data" / "wpa_team_total.csv"
DEFAULT_POSITIONAL_PATH = PROJECT_DIR / "data" / "wpa_positional_analysis.csv"


def load_team_meta(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT team_id, team_abbreviation FROM silver.dim_teams "
            "WHERE team_id BETWEEN 1610612737 AND 1610612766",
            con=conn,
        )
    return df


def load_clusters() -> pd.DataFrame:
    path = PROJECT_DIR / "data" / "team_season_clusters.csv"
    df = pd.read_csv(path, usecols=["team_id", "season_end_year", "cluster"])
    return df


def main():
    print("=" * 70)
    print("  WPA Season Positional Profile Builder (2019-2025)")
    print("=" * 70)

    engine = get_engine()
    team_meta = load_team_meta(engine)
    clusters = load_clusters()

    print("Loading ESPN player-game WPA data (Regular Season, 2019-2025)...")
    player_box = load_espn_player_box(engine)
    print(f"  {len(player_box):,} player-game rows loaded.")

    print("Assigning positions (minutes-weighted avg dAvgPos, full season per stint)...")
    positions = assign_positions(player_box, cutoff_date=None)

    print("Computing WPA_acc per player-team-season stint (full season)...")
    wpa_acc = compute_wpa_acc(player_box, cutoff_date=None, start_date=None)

    player_stints = wpa_acc.merge(positions, on=["player_id", "team_id", "season_end_year"], how="left")
    player_stints = player_stints.dropna(subset=["posicao"])

    # Team-level totals: WPA_time = sum of all player WPA_acc for that team-season;
    # MIN_time isn't literally minutes here but games (N) - team total games that season.
    team_games = player_box[player_box["played"] == True].groupby(  # noqa: E712
        ["team_id", "season_end_year"]
    )["game_id"].nunique().rename("games_played").reset_index()

    team_total = player_stints.groupby(["team_id", "season_end_year"]).agg(
        WPA_time=("WPA_acc", "sum")
    ).reset_index()
    team_total = team_total.merge(team_games, on=["team_id", "season_end_year"], how="left")
    team_total = team_total.merge(team_meta, on="team_id", how="left")
    team_total = team_total.merge(clusters, on=["team_id", "season_end_year"], how="left")
    team_total = team_total[
        ["team_id", "team_abbreviation", "season_end_year", "WPA_time", "games_played", "cluster"]
    ].sort_values(["season_end_year", "team_abbreviation"])

    # Positional totals per team-season
    pos_totals = player_stints.groupby(["team_id", "season_end_year", "posicao"]).agg(
        WPA_pos=("WPA_acc", "sum"),
        MIN_pos=("minutes", "sum"),
    ).reset_index()

    # ensure every (team, season, position) combo exists even if a group had 0 WPA
    idx = pd.MultiIndex.from_product(
        [team_meta["team_id"], range(START_YEAR, END_YEAR + 1), POSITION_GROUPS],
        names=["team_id", "season_end_year", "posicao"],
    )
    pos_totals = pos_totals.set_index(["team_id", "season_end_year", "posicao"]).reindex(idx).reset_index()
    pos_totals[["WPA_pos", "MIN_pos"]] = pos_totals[["WPA_pos", "MIN_pos"]].fillna(0.0)

    pos_totals = pos_totals.merge(
        team_total[["team_id", "season_end_year", "WPA_time", "games_played"]],
        on=["team_id", "season_end_year"], how="left"
    )
    # RCP_P = WPA_P / WPA_acc(time)  (paper section III-C)
    pos_totals["RCP_pos"] = pos_totals["WPA_pos"] / pos_totals["WPA_time"]
    # RCP_bar_P = RCP_P / N, N = numero total de jogos da equipe na temporada (III-C, last eq.)
    pos_totals["RCP_pos_bar"] = pos_totals["RCP_pos"] / pos_totals["games_played"]

    # A team-season whose total WPA_time is near zero (common for ~.500 teams, since
    # a full-game WP swing is ~+-0.5 per win/loss and these roughly cancel out over a
    # season) makes RCP_pos = WPA_pos/WPA_time numerically unstable (near-zero-
    # denominator ratio blow-up). Such team-seasons are flagged and excluded from the
    # muP league-average (their own literal RCP_pos/RCP_pos_bar/deltaP are still reported).
    pos_totals["stable"] = pos_totals["WPA_time"].abs() >= MIN_WPA_ACC_FOR_STABLE_RATIO

    # muP = (1/K) * sum(RCP_bar_k): per-season league average of RCP_pos_bar across the
    # K=30 teams (paper section III-D), computed only over stable team-seasons.
    muP = (
        pos_totals[pos_totals["stable"]]
        .groupby(["season_end_year", "posicao"])["RCP_pos_bar"].mean()
        .rename("muP").reset_index()
    )
    pos_totals = pos_totals.merge(muP, on=["season_end_year", "posicao"], how="left")
    # deltaP = RCP_bar_Time - muP (paper section III-D)
    pos_totals["deltaP"] = pos_totals["RCP_pos_bar"] - pos_totals["muP"]

    pos_totals = pos_totals.merge(team_meta, on="team_id", how="left")
    pos_totals = pos_totals.merge(clusters, on=["team_id", "season_end_year"], how="left")

    pos_totals = pos_totals[
        ["team_id", "team_abbreviation", "season_end_year", "posicao",
         "WPA_pos", "MIN_pos", "RCP_pos", "RCP_pos_bar", "stable", "muP", "deltaP", "cluster"]
    ].sort_values(["season_end_year", "team_abbreviation", "posicao"])

    DEFAULT_TEAM_TOTAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_total.to_csv(DEFAULT_TEAM_TOTAL_PATH, index=False)
    pos_totals.to_csv(DEFAULT_POSITIONAL_PATH, index=False)

    n_unstable = (~pos_totals["stable"]).sum()
    print(f"\nSaved {len(team_total)} team-season rows to {DEFAULT_TEAM_TOTAL_PATH}")
    print(f"Saved {len(pos_totals)} team-season-position rows to {DEFAULT_POSITIONAL_PATH}")
    print(f"  ({n_unstable} rows flagged 'stable=False': |WPA_time| < {MIN_WPA_ACC_FOR_STABLE_RATIO}, "
          f"excluded from muP but still reported)")
    print(f"\nSeasons covered: {sorted(pos_totals['season_end_year'].unique())}")
    print(f"muP by season/position (league average RCP_pos):")
    print(muP.pivot(index="season_end_year", columns="posicao", values="muP").to_string())


if __name__ == "__main__":
    main()
