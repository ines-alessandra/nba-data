#!/usr/bin/env python3
"""
Builds a `player_box`-shaped DataFrame covering 2013-2025 (13 seasons), in the
exact same shape wpa_common.load_espn_player_box() returns for 2019-2021:
  player_id, team_id, season_end_year, game_id, game_date, minutes_played,
  played, tWPA, dAvgPos

This lets wpa_common's own assign_positions()/compute_wpa_acc() -- which
already support arbitrary [start_date, cutoff_date) window slicing -- work
unmodified against all 13 seasons, not just 2019-2021. That slicing is what
build_wpa_trade_analysis.py needs: deltaP computed using ONLY games strictly
before a given trade date.

Per season, one of three sources feeds this shape (see
build_wpa_positional_analysis_extended.py's docstring for the full rationale
behind each):

  2019-2021  ESPN native (tWPA + dAvgPos both real, per-game, minutes-weighted)
  2022-2025  tWPA from inpredictable (scraped), dAvgPos from ESPN (still 100%
             populated even though ESPN's own tWPA is 0% these seasons),
             joined on (game_id, player_id) -- 99.8% match rate.
  2013-2018  tWPA from inpredictable; dAvgPos is a SYNTHETIC per-player
             constant derived from inpredictable's static Pos bucket (Guards
             ->2.0, Forwards->3.5, Centers->5.0 -- values chosen to fall
             unambiguously inside wpa_common.bucket_position()'s own
             thresholds, <=2.5/<=4.5/>4.5). Since the same constant is used for
             every game of a player-season, ANY minutes-weighted average of it
             over ANY date window reproduces that same constant exactly -- so
             assign_positions()'s window-slicing machinery still runs
             unmodified and always recovers the correct (season-static)
             bucket, it just can't detect an in-season position change (there
             is none to detect: this source's Pos field was already confirmed
             fixed per player, see scrape_inpredictable_positions.py).
             minutes_played is set to a placeholder (1.0, never zero) for
             these rows -- harmless, since with a constant dAvgPos the
             weighting cannot change the resulting bucket; this placeholder
             must not be read as a real minutes value by any other consumer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR, load_espn_player_box, _parse_minutes  # noqa: E402
from build_wpa_team_total_extended import build_abbrev_crosswalk  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
INPREDICTABLE_DIR = DATA_DIR / "inpredictable"
ESPN_PLAYER_BOX_PATH = DATA_DIR / "espn" / "player_box.parquet"

HYBRID_SEASONS = [2022, 2023, 2024, 2025]
STATIC_SEASONS = [2013, 2014, 2015, 2016, 2017, 2018]

SYNTHETIC_DAVGPOS = {"Guards": 2.0, "Forwards": 3.5, "Centers": 5.0}

PLAYER_BOX_COLS = ["player_id", "team_id", "season_end_year", "game_id", "game_date",
                    "minutes_played", "played", "tWPA", "dAvgPos"]


def _load_inpredictable_pb(season_end_year: int, crosswalk: dict[str, int]) -> pd.DataFrame:
    """Common first step for both the hybrid and static eras: inpredictable's per-game
    WPA, with team_id resolved via the abbreviation crosswalk and game_date parsed."""
    pb = pd.read_parquet(
        INPREDICTABLE_DIR / f"wpa_player_box_{season_end_year}.parquet",
        columns=["game_id", "game_date", "player_id", "team", "WPA"],
    )
    pb = pb.dropna(subset=["player_id"]).copy()
    pb["player_id"] = pb["player_id"].astype("int64")
    pb["team_id"] = pb["team"].map(crosswalk)
    unresolved = pb["team_id"].isna()
    if unresolved.any():
        raise RuntimeError(f"[{season_end_year}] unmapped team abbreviation(s): "
                            f"{sorted(pb.loc[unresolved, 'team'].unique())}")
    pb["team_id"] = pb["team_id"].astype("int64")
    pb["game_date"] = pd.to_datetime(pb["game_date"])
    pb["season_end_year"] = season_end_year
    pb["played"] = True
    return pb.rename(columns={"WPA": "tWPA"})


def _hybrid_seasons_pb(crosswalk: dict[str, int]) -> pd.DataFrame:
    espn_pos = pd.read_parquet(
        ESPN_PLAYER_BOX_PATH, columns=["season", "game_id", "player_id", "dAvgPos", "minutes_played"],
    )
    espn_pos = espn_pos[espn_pos["season"].isin(HYBRID_SEASONS)].copy()
    espn_pos["minutes_played"] = _parse_minutes(espn_pos["minutes_played"])
    espn_pos["player_id"] = espn_pos["player_id"].astype("int64")

    frames = []
    for season_end_year in HYBRID_SEASONS:
        pb = _load_inpredictable_pb(season_end_year, crosswalk)
        merged = pb.merge(
            espn_pos[espn_pos["season"] == season_end_year].drop(columns="season"),
            on=["game_id", "player_id"], how="left",
        )
        frames.append(merged[PLAYER_BOX_COLS])
    return pd.concat(frames, ignore_index=True)


def _static_seasons_pb(crosswalk: dict[str, int]) -> pd.DataFrame:
    frames = []
    for season_end_year in STATIC_SEASONS:
        pb = _load_inpredictable_pb(season_end_year, crosswalk)
        pb["minutes_played"] = 1.0  # placeholder -- see module docstring

        pos = pd.read_parquet(INPREDICTABLE_DIR / f"ssn_player_pos_{season_end_year}.parquet",
                               columns=["player_id", "bucket"])
        pos = pos.dropna(subset=["bucket"]).drop_duplicates(subset="player_id").copy()
        pos["player_id"] = pos["player_id"].astype("int64")
        pos["dAvgPos"] = pos["bucket"].map(SYNTHETIC_DAVGPOS)
        pos_map = pos.set_index("player_id")["dAvgPos"]

        pb["dAvgPos"] = pb["player_id"].map(pos_map)
        frames.append(pb[PLAYER_BOX_COLS])
    return pd.concat(frames, ignore_index=True)


def load_extended_player_box(engine) -> pd.DataFrame:
    """Full 2013-2025 player_box, in load_espn_player_box()'s exact output shape."""
    crosswalk = build_abbrev_crosswalk()
    native = load_espn_player_box(engine)[PLAYER_BOX_COLS]
    hybrid = _hybrid_seasons_pb(crosswalk)
    static = _static_seasons_pb(crosswalk)
    return pd.concat([native, hybrid, static], ignore_index=True)
