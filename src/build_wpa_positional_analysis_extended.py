#!/usr/bin/env python3
"""
Extends the WPA positional analysis (RCP_pos/muP/deltaP, per Sa 2025 section III)
from 3 seasons (2019-2021, ESPN-native) to 13 seasons (2013-2025), using THREE
different position-signal sources depending on what's actually available:

  2019-2021  "espn_native"          ESPN's own tWPA + dAvgPos, both real,
                                     per-game, minutes-weighted. Unchanged from
                                     build_wpa_season_profile.py.

  2022-2025  "espn_dAvgPos_hybrid"  ESPN's tWPA is 0% populated these seasons
                                     (confirmed in wpa_common.py), but its
                                     dAvgPos field is STILL 100% populated --
                                     ESPN kept computing positions after
                                     apparently disabling the WPA model. So:
                                     WPA per game from inpredictable (scraped;
                                     ESPN doesn't have it), dAvgPos per game
                                     from ESPN, joined on (game_id, player_id)
                                     -- verified 99.8% match rate. This is
                                     methodologically IDENTICAL to 2019-2021:
                                     same continuous per-game signal, same
                                     minutes-weighting, same bucket_position()
                                     thresholds.

  2013-2018  "inpredictable_static"  No dAvgPos exists anywhere for these
                                     seasons (ESPN's own source data starts at
                                     2018-19 per src/espn.py's own comment).
                                     Falls back to inpredictable's ssnPlayer.php
                                     `Pos` field (src/scrape_inpredictable_positions.py)
                                     -- empirically confirmed to be a STATIC
                                     per-player label (100% identical across
                                     3215 consecutive-season player-pairs tested,
                                     and across 12-year gaps and known mid-season
                                     trades like Durant BKN->PHX), not a
                                     per-game or even per-season computed
                                     signal. Weaker approximation than the other
                                     two methods -- flagged per-row via
                                     `position_source` and MIN_pos is left NaN
                                     (no per-game minutes available from
                                     inpredictable's box table for this era).

Team_id resolution for all inpredictable-sourced rows reuses the abbreviation
crosswalk from build_wpa_team_total_extended.py (game-level side-matching via
gamelog's `matchup` text was tried first and abandoned there after it
surfaced a real data bug in gamelog_2025.parquet -- see that script's docstring).

Both WPA_time (team_total) and RCP_pos (positional) are computed from the same
player_stints population (players WITH an assignable position only), matching
the original build_wpa_season_profile.py's own convention exactly -- so this
script's data/wpa_team_total_extended.csv output supersedes
build_wpa_team_total_extended.py's (which summed over ALL players, since no
position signal existed for the inpredictable seasons at the time it was
written; the delta is small -- only affects players ESPN/inpredictable
couldn't assign a position to).

Outputs:
  data/wpa_team_total_extended.csv        (team_id, team_abbreviation,
    season_end_year, WPA_time, games_played, cluster, source)
  data/wpa_positional_analysis_extended.csv (+ posicao, WPA_pos, MIN_pos,
    RCP_pos, RCP_pos_bar, stable, muP, deltaP, position_source)

Usage:
    python src/build_wpa_positional_analysis_extended.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import (  # noqa: E402
    POSITION_GROUPS, MIN_WPA_ACC_FOR_STABLE_RATIO, PROJECT_DIR,
    get_engine, load_espn_player_box, assign_positions, compute_wpa_acc,
    bucket_position, _parse_minutes,
)
from build_wpa_team_total_extended import (  # noqa: E402
    INPREDICTABLE_SEASONS, build_abbrev_crosswalk, ABBREV_OVERRIDES,
)

DATA_DIR = PROJECT_DIR / "data"
INPREDICTABLE_DIR = DATA_DIR / "inpredictable"
ESPN_PLAYER_BOX_PATH = DATA_DIR / "espn" / "player_box.parquet"
CLUSTERS_PATH = DATA_DIR / "team_season_clusters.csv"
TEAM_TOTAL_OUT_PATH = DATA_DIR / "wpa_team_total_extended.csv"
POSITIONAL_OUT_PATH = DATA_DIR / "wpa_positional_analysis_extended.csv"

HYBRID_SEASONS = [2022, 2023, 2024, 2025]
STATIC_SEASONS = [2013, 2014, 2015, 2016, 2017, 2018]

STINT_COLS = ["player_id", "team_id", "season_end_year", "WPA_acc", "games", "minutes", "posicao", "position_source"]


def load_team_meta_from_gamelog(crosswalk: dict[str, int]) -> pd.DataFrame:
    """team_id -> team_abbreviation, using the canonical (non-inpredictable) side
    of the crosswalk built in build_wpa_team_total_extended.py."""
    canonical = {v: k for k, v in crosswalk.items() if k not in ABBREV_OVERRIDES}
    return pd.DataFrame({"team_id": list(canonical.keys()), "team_abbreviation": list(canonical.values())})


def stints_espn_native(engine) -> pd.DataFrame:
    """2019-2021: unchanged logic from build_wpa_season_profile.py."""
    player_box = load_espn_player_box(engine)
    positions = assign_positions(player_box, cutoff_date=None)
    wpa_acc = compute_wpa_acc(player_box, cutoff_date=None, start_date=None)
    stints = wpa_acc.merge(positions, on=["player_id", "team_id", "season_end_year"], how="left")
    stints = stints.dropna(subset=["posicao"])
    stints["position_source"] = "espn_native"
    return stints[STINT_COLS]


def stints_hybrid(crosswalk: dict[str, int]) -> pd.DataFrame:
    """2022-2025: inpredictable WPA + ESPN dAvgPos, joined on (game_id, player_id)."""
    espn_pos = pd.read_parquet(
        ESPN_PLAYER_BOX_PATH,
        columns=["season", "game_id", "player_id", "dAvgPos", "minutes_played"],
    )
    espn_pos = espn_pos[espn_pos["season"].isin(HYBRID_SEASONS)].copy()
    espn_pos["minutes_played"] = _parse_minutes(espn_pos["minutes_played"])
    espn_pos["player_id"] = espn_pos["player_id"].astype("int64")

    all_stints = []
    for season_end_year in HYBRID_SEASONS:
        pb = pd.read_parquet(INPREDICTABLE_DIR / f"wpa_player_box_{season_end_year}.parquet",
                              columns=["game_id", "player_id", "team", "WPA"])
        pb = pb.dropna(subset=["player_id"]).copy()
        pb["player_id"] = pb["player_id"].astype("int64")
        pb["team_id"] = pb["team"].map(crosswalk)
        unresolved = pb["team_id"].isna()
        if unresolved.any():
            raise RuntimeError(f"[{season_end_year}] unmapped team abbreviation(s): "
                                f"{sorted(pb.loc[unresolved, 'team'].unique())}")
        pb["team_id"] = pb["team_id"].astype("int64")

        wpa_acc = pb.groupby(["player_id", "team_id"]).agg(
            WPA_acc=("WPA", "sum"), games=("game_id", "nunique"),
        ).reset_index()
        wpa_acc["season_end_year"] = season_end_year

        merged = pb.merge(espn_pos[espn_pos["season"] == season_end_year],
                           on=["game_id", "player_id"], how="inner")
        w = merged["minutes_played"].clip(lower=0.01)
        tmp = merged[["player_id", "team_id"]].copy()
        tmp["w"] = w
        tmp["wp"] = merged["dAvgPos"] * w
        grouped = tmp.groupby(["player_id", "team_id"], as_index=False).agg(w=("w", "sum"), wp=("wp", "sum"))
        grouped["avg_pos"] = grouped["wp"] / grouped["w"]
        grouped["posicao"] = grouped["avg_pos"].apply(bucket_position)
        grouped["minutes"] = grouped["w"]

        stints = wpa_acc.merge(grouped[["player_id", "team_id", "posicao", "minutes"]],
                                on=["player_id", "team_id"], how="left")
        stints = stints.dropna(subset=["posicao"])
        stints["position_source"] = "espn_dAvgPos_hybrid"
        all_stints.append(stints[STINT_COLS])

    return pd.concat(all_stints, ignore_index=True)


def stints_static(crosswalk: dict[str, int]) -> pd.DataFrame:
    """2013-2018: inpredictable WPA + inpredictable's static per-player Pos bucket."""
    all_stints = []
    for season_end_year in STATIC_SEASONS:
        pb = pd.read_parquet(INPREDICTABLE_DIR / f"wpa_player_box_{season_end_year}.parquet",
                              columns=["game_id", "player_id", "team", "WPA"])
        pb = pb.dropna(subset=["player_id"]).copy()
        pb["player_id"] = pb["player_id"].astype("int64")
        pb["team_id"] = pb["team"].map(crosswalk)
        unresolved = pb["team_id"].isna()
        if unresolved.any():
            raise RuntimeError(f"[{season_end_year}] unmapped team abbreviation(s): "
                                f"{sorted(pb.loc[unresolved, 'team'].unique())}")
        pb["team_id"] = pb["team_id"].astype("int64")

        wpa_acc = pb.groupby(["player_id", "team_id"]).agg(
            WPA_acc=("WPA", "sum"), games=("game_id", "nunique"),
        ).reset_index()
        wpa_acc["season_end_year"] = season_end_year
        wpa_acc["minutes"] = pd.NA  # not available: inpredictable's per-game box has no minutes column

        pos = pd.read_parquet(INPREDICTABLE_DIR / f"ssn_player_pos_{season_end_year}.parquet",
                               columns=["player_id", "bucket"])
        pos = pos.dropna(subset=["bucket"]).rename(columns={"bucket": "posicao"})
        pos["player_id"] = pos["player_id"].astype("int64")
        pos = pos.drop_duplicates(subset="player_id")  # static label, one per player

        stints = wpa_acc.merge(pos, on="player_id", how="left").dropna(subset=["posicao"])
        stints["position_source"] = "inpredictable_static"
        all_stints.append(stints[STINT_COLS])

    return pd.concat(all_stints, ignore_index=True)


def main() -> int:
    print("=" * 70)
    print("  WPA Positional Analysis -- Extended (2013-2025)")
    print("=" * 70)

    engine = get_engine()
    crosswalk = build_abbrev_crosswalk()
    team_meta = load_team_meta_from_gamelog(crosswalk)
    clusters = pd.read_csv(CLUSTERS_PATH, usecols=["team_id", "season_end_year", "cluster"])

    print("Loading ESPN-native player stints (2019-2021)...")
    s_native = stints_espn_native(engine)
    print(f"  {len(s_native):,} stints.")

    print("Loading hybrid (ESPN dAvgPos + inpredictable WPA) player stints (2022-2025)...")
    s_hybrid = stints_hybrid(crosswalk)
    print(f"  {len(s_hybrid):,} stints.")

    print("Loading inpredictable-static-position player stints (2013-2018)...")
    s_static = stints_static(crosswalk)
    print(f"  {len(s_static):,} stints.")

    player_stints = pd.concat([s_native, s_hybrid, s_static], ignore_index=True)

    # --- Team totals (same population as RCP_pos's denominator, matching the
    # original script's own convention) ---
    team_total = player_stints.groupby(["team_id", "season_end_year"]).agg(
        WPA_time=("WPA_acc", "sum"), games_played=("games", "max"),
    ).reset_index()
    # games_played should be the team's actual games that season, not a player's;
    # recompute properly per team-season from the per-stint game counts is wrong
    # (players played different subsets); use nunique game_id per team-season instead.
    team_games = pd.concat([
        pd.read_parquet(INPREDICTABLE_DIR / f"wpa_player_box_{y}.parquet", columns=["game_id", "team"])
          .assign(team_id=lambda d, y=y: d["team"].map(crosswalk), season_end_year=y)
        for y in INPREDICTABLE_SEASONS
    ], ignore_index=True).groupby(["team_id", "season_end_year"])["game_id"].nunique().rename("games_played_actual")
    team_total = team_total.drop(columns=["games_played"]).merge(
        team_games.reset_index().rename(columns={"games_played_actual": "games_played"}),
        on=["team_id", "season_end_year"], how="left")

    # ESPN seasons: games_played should come from the ESPN player_box games too,
    # not from inpredictable (which has no rows for 2019-2021).
    espn_games = load_espn_player_box(engine)[["team_id", "season_end_year", "game_id"]] \
        .drop_duplicates().groupby(["team_id", "season_end_year"])["game_id"].nunique().rename("games_played")
    espn_team_total = player_stints[player_stints["position_source"] == "espn_native"].groupby(
        ["team_id", "season_end_year"]).agg(WPA_time=("WPA_acc", "sum")).reset_index()
    espn_team_total = espn_team_total.merge(espn_games.reset_index(), on=["team_id", "season_end_year"], how="left")

    team_total = pd.concat([
        team_total[~team_total["season_end_year"].isin([2019, 2020, 2021])],
        espn_team_total,
    ], ignore_index=True)

    team_total = team_total.merge(team_meta, on="team_id", how="left")
    team_total = team_total.merge(clusters, on=["team_id", "season_end_year"], how="left")
    team_total["source"] = team_total["season_end_year"].apply(
        lambda y: "espn" if 2019 <= y <= 2021 else "inpredictable")
    team_total = team_total[["team_id", "team_abbreviation", "season_end_year", "WPA_time", "games_played",
                              "cluster", "source"]].sort_values(["season_end_year", "team_abbreviation"])

    TEAM_TOTAL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_total.to_csv(TEAM_TOTAL_OUT_PATH, index=False)
    print(f"\nSaved {len(team_total)} team-season rows to {TEAM_TOTAL_OUT_PATH} (position-filtered WPA_time; "
          f"supersedes build_wpa_team_total_extended.py's all-player version).")

    # --- Positional breakdown ---
    pos_totals = player_stints.groupby(["team_id", "season_end_year", "posicao"]).agg(
        WPA_pos=("WPA_acc", "sum"),
        MIN_pos=("minutes", "sum"),
    ).reset_index()

    idx = pd.MultiIndex.from_product(
        [team_meta["team_id"], sorted(player_stints["season_end_year"].unique()), POSITION_GROUPS],
        names=["team_id", "season_end_year", "posicao"],
    )
    pos_totals = pos_totals.set_index(["team_id", "season_end_year", "posicao"]).reindex(idx).reset_index()
    pos_totals["WPA_pos"] = pos_totals["WPA_pos"].fillna(0.0)
    # MIN_pos: pandas' groupby(...).sum() silently returns 0.0 (not NaN) for an
    # all-NaN group, which is exactly what happens every static-position season
    # (minutes is pd.NA there by construction -- no per-game minutes available
    # from inpredictable's box table for that era). 0.0 would misleadingly read
    # as "this position group played zero minutes"; NaN correctly reads as "not
    # available for this data source".
    pos_totals.loc[pos_totals["season_end_year"].isin(STATIC_SEASONS), "MIN_pos"] = pd.NA

    pos_totals = pos_totals.merge(
        team_total[["team_id", "season_end_year", "WPA_time", "games_played"]],
        on=["team_id", "season_end_year"], how="left")
    pos_totals["RCP_pos"] = pos_totals["WPA_pos"] / pos_totals["WPA_time"]
    pos_totals["RCP_pos_bar"] = pos_totals["RCP_pos"] / pos_totals["games_played"]
    pos_totals["stable"] = pos_totals["WPA_time"].abs() >= MIN_WPA_ACC_FOR_STABLE_RATIO

    muP = (
        pos_totals[pos_totals["stable"]]
        .groupby(["season_end_year", "posicao"])["RCP_pos_bar"].mean()
        .rename("muP").reset_index()
    )
    pos_totals = pos_totals.merge(muP, on=["season_end_year", "posicao"], how="left")
    pos_totals["deltaP"] = pos_totals["RCP_pos_bar"] - pos_totals["muP"]

    # position_source: dominant source per team-season (all rows in a team-season
    # share the same source in this design, so first() is exact, not a guess)
    src_map = player_stints.groupby(["team_id", "season_end_year"])["position_source"].first()
    pos_totals = pos_totals.merge(src_map.reset_index(), on=["team_id", "season_end_year"], how="left")

    pos_totals = pos_totals.merge(team_meta, on="team_id", how="left")
    pos_totals = pos_totals.merge(clusters, on=["team_id", "season_end_year"], how="left")

    pos_totals = pos_totals[[
        "team_id", "team_abbreviation", "season_end_year", "posicao",
        "WPA_pos", "MIN_pos", "RCP_pos", "RCP_pos_bar", "stable", "muP", "deltaP",
        "cluster", "position_source",
    ]].sort_values(["season_end_year", "team_abbreviation", "posicao"])

    pos_totals.to_csv(POSITIONAL_OUT_PATH, index=False)
    n_unstable = (~pos_totals["stable"]).sum()
    print(f"Saved {len(pos_totals)} team-season-position rows to {POSITIONAL_OUT_PATH}")
    print(f"  ({n_unstable} rows flagged 'stable=False', excluded from muP but still reported)")
    print(f"\nSeasons covered: {sorted(pos_totals['season_end_year'].unique())}")
    print(f"Rows per position_source:\n{pos_totals.groupby('position_source').size().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
