#!/usr/bin/env python3
"""
Shared utilities for the WPA (Win Probability Added) trade-impact methodology.

Implements the metric definitions exactly as specified:
  - WPA_acc  = sum of per-shift delta-WP over a period (ESPN's tWPA per player-game
               already is the shift-aggregated, per-player-attributed WP delta, so
               WPA_acc(group) = sum of tWPA over all player-games in the group/period).
  - WPA      = WPA_acc / N (N = games played by the group in the period).
  - WPA_pos  = sum of WPA_acc(j) for players j in positional group P.
  - RCP_pos  = WPA_pos / WPA_acc(team)               (same period for both terms).
  - muP      = mean of RCP_pos across the K=30 league teams, for a given season.
  - deltaP   = RCP_pos(team) - muP                    (>0 = virtude, <0 = lacuna).

Positional grouping (Guards / Forwards / Centers) is derived from ESPN's per-game
`dAvgPos` field (continuous defensive-matchup position, 1=PG .. 5=C), which is the
only positional signal with full roster coverage in this dataset (the nba.com
boxscore `position` field is only populated for the 5 starters per team per game).
Bucketing: <=2.5 -> Guards, (2.5, 4.5] -> Forwards, >4.5 -> Centers.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

PROJECT_DIR = Path(__file__).resolve().parent.parent
ESPN_PLAYER_BOX_PATH = PROJECT_DIR / "data" / "espn" / "player_box.parquet"

START_YEAR = 2019
END_YEAR = 2021  # ESPN's oWPA/dWPA/tWPA fields are only populated in the raw archives
                 # for seasons 2019-2021; they are entirely absent (100% NaN, confirmed
                 # against the raw per-day JSON, not just the parquet) for 2022-2025.

MIN_GAMES_FOR_SNAPSHOT = 10  # minimum team games played before a date to trust a rank/RCP snapshot

MIN_WPA_ACC_FOR_STABLE_RATIO = 1.0  # |WPA_acc(team)| below this makes RCP_pos = WPA_pos/WPA_acc(team)
                                     # numerically unstable (near-zero-denominator ratio blow-up: a
                                     # team near .500 has season WPA_acc trending to 0 almost by
                                     # construction, since a full-game WP swing is ~+-0.5 per win/loss).
                                     # Such team-seasons are flagged `stable=False` and excluded from
                                     # muP (league-average) computations, though their own literal
                                     # RCP_pos/deltaP values are still reported.

# Static conference map (stable across the 2019-2025 window; no relocations/realignments).
TEAM_CONFERENCE = {
    1610612737: "East",  # ATL
    1610612738: "East",  # BOS
    1610612739: "East",  # CLE
    1610612740: "West",  # NOP (dim_teams row is stale "New Orleans Hornets" naming, id is correct)
    1610612741: "East",  # CHI
    1610612742: "West",  # DAL
    1610612743: "West",  # DEN
    1610612744: "West",  # GSW
    1610612745: "West",  # HOU
    1610612746: "West",  # LAC
    1610612747: "West",  # LAL
    1610612748: "East",  # MIA
    1610612749: "East",  # MIL
    1610612750: "West",  # MIN
    1610612751: "East",  # BKN
    1610612752: "East",  # NYK
    1610612753: "East",  # ORL
    1610612754: "East",  # IND
    1610612755: "East",  # PHI
    1610612756: "West",  # PHX
    1610612757: "West",  # POR
    1610612758: "West",  # SAC
    1610612759: "West",  # SAS
    1610612760: "West",  # OKC
    1610612761: "East",  # TOR
    1610612762: "West",  # UTA
    1610612763: "West",  # MEM
    1610612764: "East",  # WAS
    1610612765: "East",  # DET
    1610612766: "East",  # CHA
}

POSITION_GROUPS = ["Guards", "Forwards", "Centers"]


def _parse_minutes(series: pd.Series) -> pd.Series:
    """Convert 'MM:SS' strings to decimal minutes (float)."""
    parts = series.astype(str).str.split(":", expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0.0
    return (mins + secs.fillna(0.0) / 60.0).fillna(0.0)


def get_engine(db_url: str = DB_URL):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def bucket_position(avg_pos: float) -> str:
    if pd.isna(avg_pos):
        return None
    if avg_pos <= 2.5:
        return "Guards"
    if avg_pos <= 4.5:
        return "Forwards"
    return "Centers"


def load_espn_player_box(engine) -> pd.DataFrame:
    """Load ESPN per-player-per-game WPA + dAvgPos, restricted to Regular Season
    games in the target window, tagged with the real game_date (needed for all
    pre/post-trade cutoffs).

    IMPORTANT: `team_id` is NaN in the raw parquet for every 2022-2025 row (ESPN
    only backfilled it through 2021); only the `team` text abbreviation is always
    present, and it uses ESPN's own codes (BRK/NOR/PHO/SAN) which differ from
    nba.com's (BKN/NOP/PHX/SAS) for 4 franchises. We recover team_id for every
    row by building an abbreviation->team_id map from the rows where team_id IS
    populated (2019-2021) and applying it league-wide."""
    df = pd.read_parquet(
        ESPN_PLAYER_BOX_PATH,
        columns=["season", "game_id", "player_id", "team_id", "team", "name",
                 "dAvgPos", "minutes_played", "played", "tWPA"],
    )
    df = df[(df["season"] >= START_YEAR) & (df["season"] <= END_YEAR)].copy()

    abbr_to_id = (
        df.dropna(subset=["team_id"])
        .drop_duplicates(subset=["team"])
        .set_index("team")["team_id"]
        .astype("int64")
        .to_dict()
    )
    missing = sorted(set(df["team"].unique()) - set(abbr_to_id.keys()))
    if missing:
        raise RuntimeError(f"No team_id mapping recoverable for ESPN team codes: {missing}")

    df["team_id"] = df["team"].map(abbr_to_id).astype("int64")
    df = df.drop(columns=["team"])
    df["player_id"] = df["player_id"].astype("int64")
    df["minutes_played"] = _parse_minutes(df["minutes_played"])

    with engine.connect() as conn:
        games = pd.read_sql(
            text("""
                SELECT game_id, game_date, season_end_year
                FROM silver.dim_games
                WHERE game_type = 'Regular Season'
                  AND season_end_year BETWEEN :start_year AND :end_year
            """),
            con=conn,
            params={"start_year": START_YEAR, "end_year": END_YEAR},
        )
    games["game_date"] = pd.to_datetime(games["game_date"])

    df = df.merge(games, on="game_id", how="inner")
    # sanity: ESPN's own `season` column should equal the joined season_end_year
    df = df[df["season"] == df["season_end_year"]].drop(columns=["season"])
    return df


def assign_positions(player_box: pd.DataFrame, cutoff_date: pd.Timestamp = None,
                      start_date: pd.Timestamp = None) -> pd.DataFrame:
    """Assign one position bucket per (player_id, team_id, season_end_year) stint,
    using minutes-weighted average dAvgPos over games in [start_date, cutoff_date)
    if given, else the full season. This lets a mid-season-traded player get a
    distinct, correctly-timed position bucket per team stint. Pass start_date alone
    for a POST-trade window (mirrors compute_wpa_acc's start_date/cutoff_date)."""
    df = player_box
    if start_date is not None:
        df = df[df["game_date"] >= start_date]
    if cutoff_date is not None:
        df = df[df["game_date"] < cutoff_date]
    df = df[df["played"] == True]  # noqa: E712

    cols = ["player_id", "team_id", "season_end_year", "posicao"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    w = df["minutes_played"].clip(lower=0.01)
    tmp = df[["player_id", "team_id", "season_end_year"]].copy()
    tmp["w"] = w
    tmp["wp"] = df["dAvgPos"] * w

    grouped = tmp.groupby(["player_id", "team_id", "season_end_year"], as_index=False).agg(
        w=("w", "sum"), wp=("wp", "sum")
    )
    grouped["avg_pos"] = grouped["wp"] / grouped["w"]
    grouped["posicao"] = grouped["avg_pos"].apply(bucket_position)
    return grouped[cols]


def compute_wpa_acc(player_box: pd.DataFrame, cutoff_date: pd.Timestamp = None,
                     start_date: pd.Timestamp = None) -> pd.DataFrame:
    """WPA_acc per (player_id, team_id, season_end_year), summing tWPA over the
    selected window: [start_date, cutoff_date). Also returns games played (N)."""
    df = player_box
    if start_date is not None:
        df = df[df["game_date"] >= start_date]
    if cutoff_date is not None:
        df = df[df["game_date"] < cutoff_date]
    df = df[df["played"] == True]  # noqa: E712

    cols = ["player_id", "team_id", "season_end_year", "WPA_acc", "games", "minutes"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    agg = df.groupby(["player_id", "team_id", "season_end_year"]).agg(
        WPA_acc=("tWPA", "sum"),
        games=("game_id", "nunique"),
        minutes=("minutes_played", "sum"),
    ).reset_index()
    return agg


def season_date_bounds(engine, season_end_year: int) -> tuple:
    """(first, last) Regular Season game_date for a season - used to restrict the
    trade universe to trades that actually happened DURING the season (excluding
    draft-night/free-agency offseason trades that get bucketed into a
    season_end_year purely by the July-cutoff convention, and preseason moves
    dated before the season's own first game)."""
    with engine.connect() as conn:
        row = pd.read_sql(
            text("""
                SELECT MIN(game_date) AS lo, MAX(game_date) AS hi
                FROM silver.dim_games
                WHERE game_type = 'Regular Season' AND season_end_year = :s
            """),
            con=conn,
            params={"s": int(season_end_year)},
        ).iloc[0]
    return pd.Timestamp(row["lo"]), pd.Timestamp(row["hi"])


def team_games_asof(engine, season_end_year: int, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Per-team win_pct and games played, using Regular Season games strictly
    before cutoff_date, for a single season. Restricted to the 30 real NBA teams."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT tg.team_id, tg.wl, g.game_date
                FROM silver.fact_team_gamelogs tg
                JOIN silver.dim_games g ON tg.game_id = g.game_id
                WHERE g.game_type = 'Regular Season'
                  AND g.season_end_year = :season
                  AND g.game_date < :cutoff
                  AND tg.team_id BETWEEN 1610612737 AND 1610612766
            """),
            con=conn,
            params={"season": int(season_end_year), "cutoff": cutoff_date.date()},
        )
    if df.empty:
        return pd.DataFrame(columns=["team_id", "games", "win_pct"])
    out = df.groupby("team_id").agg(
        games=("wl", "size"),
        wins=("wl", lambda s: (s == "W").sum()),
    ).reset_index()
    out["win_pct"] = out["wins"] / out["games"]
    return out[["team_id", "games", "win_pct"]]


def classify_objective(team_id: int, season_end_year: int, cutoff_date: pd.Timestamp,
                        standings: pd.DataFrame) -> dict:
    """Classify a team's seasonal objective (Contender / Intermediate / Tanking) as of
    cutoff_date, based on its conference-rank-by-win_pct among the other 14 conference
    teams (all snapshotted at the same cutoff_date to avoid look-ahead bias).
    Tiers: rank 1-6 = Contender, 7-10 = Intermediate, 11-15 = Tanking.
    Returns None fields if fewer than MIN_GAMES_FOR_SNAPSHOT games have been played."""
    conf = TEAM_CONFERENCE.get(team_id)
    conf_teams = [tid for tid, c in TEAM_CONFERENCE.items() if c == conf]
    snap = standings[standings["team_id"].isin(conf_teams)].copy()
    snap = snap[snap["games"] >= MIN_GAMES_FOR_SNAPSHOT]
    if team_id not in snap["team_id"].values or len(snap) < 8:
        return {"conference": conf, "conf_rank": None, "win_pct_asof": None, "objetivo_sazonal": None}

    snap = snap.sort_values("win_pct", ascending=False).reset_index(drop=True)
    snap["conf_rank"] = snap["win_pct"].rank(ascending=False, method="min").astype(int)
    row = snap[snap["team_id"] == team_id].iloc[0]
    rank = int(row["conf_rank"])

    if rank <= 6:
        objetivo = "Contender"
    elif rank <= 10:
        objetivo = "Intermediario"
    else:
        objetivo = "Tanking"

    return {
        "conference": conf,
        "conf_rank": rank,
        "win_pct_asof": float(row["win_pct"]),
        "objetivo_sazonal": objetivo,
    }
