#!/usr/bin/env python3
"""
Trade-level WPA gap-classification (H1) and seasonal-objective impact analysis (H2).
Seasons: 2019-2021 (the only years with real ESPN oWPA/dWPA/tWPA data - see wpa_common.py).

For every player acquired in a trade:

  H1 - Preenche Lacuna vs Redundante
    deltaP for the ACQUIRING team is computed using ONLY games strictly before the
    trade date that season (both the team's own RCP_pos and the league benchmark
    muP, snapshotted for all 30 teams at that same date) - a trade cannot be
    diagnosed as "filling a gap" using data generated after the trade itself
    already happened. The acquired player's position comes from the trade log's
    own labeled position (bronze.raw_trades.player_position), falling back to the
    player's minutes-weighted dAvgPos bucket WITH the acquiring team when that's
    blank (in practice this fallback never fires - see wpa_common.py comments -
    but it's scoped to fail closed rather than leak post-trade data).
    Classification: player's position group has deltaP < 0 (below league average,
    i.e. a real structural deficiency) -> "Preenche Lacuna"; deltaP >= 0 -> "Redundante".

  H2 - Objetivo Sazonal (Contender / Intermediario / Tanking)
    The acquiring team's conference standing rank is computed at the same
    pre-trade cutoff (win_pct over games strictly before trade_date).
    Tiers: rank 1-6 = Contender, 7-10 = Intermediario, 11-15 = Tanking.

  Impact measure (delta_WPA_time)
    Team-level WPA rate before vs. after the trade date, using ALL of that
    season's games in each sub-period (not a fixed-size window):
      WPA_rate_pre  = WPA_acc(team, games < trade_date)  / games_pre
      WPA_rate_post = WPA_acc(team, games >= trade_date) / games_post
      delta_WPA_time = WPA_rate_post - WPA_rate_pre

Outputs:
  data/wpa_trade_classification.csv  - one row per (trade_id, team_id, acquired player)
  data/wpa_trade_impact_analysis.csv - one row per (team_id, trade_date) EVENT (comma-joined
                                        trade_ids column when >1 trade_id shares a team+date),
                                        deduplicated so delta_WPA_time - a function of team+date
                                        alone - is never duplicated across rows that could carry
                                        different gap-fill classifications (pseudo-replication)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import (  # noqa: E402
    START_YEAR, END_YEAR, MIN_GAMES_FOR_SNAPSHOT, MIN_WPA_ACC_FOR_STABLE_RATIO, PROJECT_DIR,
    get_engine, load_espn_player_box, assign_positions, compute_wpa_acc,
    team_games_asof, classify_objective, bucket_position, season_date_bounds,
)

CLASSIFICATION_PATH = PROJECT_DIR / "data" / "wpa_trade_classification.csv"
IMPACT_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis.csv"

RAW_TRADE_POS_TO_GROUP = {
    "PG": "Guards", "SG": "Guards", "G": "Guards",
    "SF": "Forwards", "PF": "Forwards", "F": "Forwards",
    "C": "Centers",
}


def load_incoming_players(engine, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT trade_id, date AS trade_date, season_end_year, team_id,
                       player_id, player_name, player_position
                FROM bronze.raw_trades
                WHERE direction = 'Incoming'
                  AND asset_type = 'Player'
                  AND player_id IS NOT NULL
                  AND team_id BETWEEN 1610612737 AND 1610612766
                  AND season_end_year BETWEEN :start_year AND :end_year
            """),
            con=conn,
            params={"start_year": start_year, "end_year": end_year},
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["player_id"] = df["player_id"].astype("int64")
    df["team_id"] = df["team_id"].astype("int64")
    return df.drop_duplicates(subset=["trade_id", "team_id", "player_id"])


def team_meta(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            "SELECT team_id, team_abbreviation FROM silver.dim_teams "
            "WHERE team_id BETWEEN 1610612737 AND 1610612766",
            con=conn,
        )


def season_start_date(engine, season_end_year: int) -> pd.Timestamp:
    with engine.connect() as conn:
        d = pd.read_sql(
            text("""
                SELECT MIN(game_date) AS d FROM silver.dim_games
                WHERE season_end_year = :s AND game_type = 'Regular Season'
            """),
            con=conn,
            params={"s": int(season_end_year)},
        )["d"].iloc[0]
    return pd.Timestamp(d)


def league_window_snapshot(player_box: pd.DataFrame, season_end_year: int,
                            window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """RCP_bar_pos for all 30 teams, positions and WPA both computed using ONLY games
    in [window_start, window_end) of that season. Teams with an unstable (near-zero)
    total WPA_acc denominator are dropped before averaging into muP. Follows the
    paper's exact chain: RCP_P = WPA_P/WPA_acc(time), RCP_bar_P = RCP_P/N (N = team's
    own games played in this window). Pass [season_start, trade_date) for a PRE-trade
    snapshot, or [trade_date, season_end) for a POST-trade snapshot (descriptive)."""
    season_pb = player_box[player_box["season_end_year"] == season_end_year]
    positions = assign_positions(season_pb, start_date=window_start, cutoff_date=window_end)
    wpa_acc = compute_wpa_acc(season_pb, start_date=window_start, cutoff_date=window_end)

    stints = wpa_acc.merge(positions, on=["player_id", "team_id", "season_end_year"], how="left")
    stints = stints.dropna(subset=["posicao"])

    window_pb = season_pb[(season_pb["game_date"] >= window_start) & (season_pb["game_date"] < window_end)
                           & (season_pb["played"] == True)]  # noqa: E712
    team_games = window_pb.groupby("team_id")["game_id"].nunique().rename("N_games").reset_index()

    team_total = stints.groupby("team_id")["WPA_acc"].sum().rename("WPA_time_team").reset_index()
    pos_totals = stints.groupby(["team_id", "posicao"])["WPA_acc"].sum().rename("WPA_pos").reset_index()
    pos_totals = pos_totals.merge(team_total, on="team_id", how="left")
    pos_totals = pos_totals.merge(team_games, on="team_id", how="left")

    pos_totals["stable"] = pos_totals["WPA_time_team"].abs() >= MIN_WPA_ACC_FOR_STABLE_RATIO
    pos_totals["RCP_pos"] = pos_totals["WPA_pos"] / pos_totals["WPA_time_team"]
    pos_totals["RCP_pos_bar"] = pos_totals["RCP_pos"] / pos_totals["N_games"]
    return pos_totals


def asof_league_snapshot(player_box: pd.DataFrame, season_end_year: int,
                          cutoff_date: pd.Timestamp, season_start: pd.Timestamp) -> pd.DataFrame:
    """Backwards-compatible alias: PRE-trade snapshot, [season_start, cutoff_date)."""
    return league_window_snapshot(player_box, season_end_year, season_start, cutoff_date)


def run_analysis(engine, player_box: pd.DataFrame, season_range: range,
                  classification_path: Path, impact_path: Path,
                  pretrade_clusters_path: Path, label: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core H1/H2 pipeline, parametrized over the (player_box, season_range, output
    paths) that differ between the original 2019-2021 ESPN-only run (main(), below)
    and the 2013-2025 extended run (build_wpa_trade_analysis_extended.py). The
    classification/impact logic itself is untouched from the original script."""
    print("=" * 70)
    print(f"  WPA Trade Gap-Classification & Objective Impact Analysis {label}")
    print("=" * 70)

    start_year, end_year = min(season_range), max(season_range)
    incoming = load_incoming_players(engine, start_year, end_year)
    print(f"Loaded {len(incoming):,} player-acquisition rows "
          f"({incoming['trade_id'].nunique()} distinct trades, "
          f"{incoming.groupby(['team_id', 'trade_date']).ngroups} team-date events) "
          f"before restricting to genuine in-season trades.")

    # Restrict to trades that actually happened DURING that season's Regular Season
    # window - this excludes draft-night/free-agency offseason trades (June-Sept)
    # that the source data buckets into a season_end_year purely by a July cutoff
    # convention, and preseason moves. This is the "mid-season trade" universe the
    # hypotheses are actually about.
    bounds = {s: season_date_bounds(engine, s) for s in season_range}
    keep_mask = incoming.apply(
        lambda r: bounds[r["season_end_year"]][0] <= r["trade_date"] <= bounds[r["season_end_year"]][1],
        axis=1,
    )
    n_dropped = (~keep_mask).sum()
    incoming = incoming[keep_mask]
    print(f"Dropped {n_dropped} rows outside the Regular Season date window (offseason/preseason trades).")
    print(f"In-season universe: {len(incoming):,} player-acquisition rows "
          f"({incoming['trade_id'].nunique()} distinct trades, "
          f"{incoming.groupby(['team_id', 'trade_date']).ngroups} team-date events).")

    tmeta = team_meta(engine)
    season_positions_full = assign_positions(player_box, cutoff_date=None)  # fallback source

    season_starts = {s: season_start_date(engine, s) for s in season_range}

    classification_rows = []
    impact_rows = []

    season_ends_excl = {
        s: season_date_bounds(engine, s)[1] + pd.Timedelta(days=1) for s in season_range
    }

    unique_snapshots = incoming[["season_end_year", "trade_date"]].drop_duplicates()
    print(f"Computing {len(unique_snapshots)} as-of-date league snapshots for muP_asof (pre AND post-trade)...")

    snapshot_cache = {}
    post_snapshot_cache = {}
    for _, row in unique_snapshots.iterrows():
        key = (int(row["season_end_year"]), row["trade_date"])
        snapshot_cache[key] = asof_league_snapshot(
            player_box, key[0], key[1], season_starts[key[0]]
        )
        # POST-trade snapshot: descriptive only (did the diagnosed gap actually close?),
        # never used to decide the Preenche Lacuna/Redundante classification itself.
        post_snapshot_cache[key] = league_window_snapshot(
            player_box, key[0], key[1], season_ends_excl[key[0]]
        )

    excluded_no_snapshot = 0
    excluded_unstable_team = 0
    excluded_no_min_games = 0

    # Group by (team_id, trade_date) - NOT by trade_id - because delta_WPA_time is a
    # function of the team's pre/post-date split alone. If a team executes two separate
    # trade_ids on the same date (real, happens on trade-deadline day), they share one
    # outcome window; keeping trade_id in the grouping key would duplicate that single
    # outcome across multiple rows, and those rows could even carry opposite gap-fill
    # classifications - pseudo-replication that directly contaminates the H1 test.
    for (team_id, trade_date), grp in incoming.groupby(["team_id", "trade_date"]):
        season = int(grp["season_end_year"].iloc[0])
        trade_ids = sorted(grp["trade_id"].unique().tolist())
        snap = snapshot_cache[(season, trade_date)]

        team_snap = snap[snap["team_id"] == team_id]
        if team_snap.empty or not team_snap["stable"].all():
            excluded_unstable_team += 1
            continue

        # muP_asof = (1/K) * sum(RCP_bar_k) across all K=30 stable teams at this same
        # snapshot (paper section III-D; literally includes the team itself).
        muP_asof = (
            snap[snap["stable"]]
            .groupby("posicao")["RCP_pos_bar"].mean()
            .to_dict()
        )
        team_rcp = team_snap.set_index("posicao")["RCP_pos_bar"].to_dict()
        deltaP_team = {p: team_rcp.get(p, np.nan) - muP_asof.get(p, np.nan) for p in muP_asof}

        # POST-trade deltaP (descriptive only): did the diagnosed gap at this position
        # actually close after the trade? Same formula, just using the post-trade
        # window for both the team and the league benchmark. NaN if the post-trade
        # window is too unstable/short to trust (season ending soon after the trade).
        post_snap = post_snapshot_cache[(season, trade_date)]
        post_team_snap = post_snap[post_snap["team_id"] == team_id]
        if not post_team_snap.empty and post_team_snap["stable"].all():
            muP_post = post_snap[post_snap["stable"]].groupby("posicao")["RCP_pos_bar"].mean().to_dict()
            team_rcp_post = post_team_snap.set_index("posicao")["RCP_pos_bar"].to_dict()
            deltaP_team_post = {p: team_rcp_post.get(p, np.nan) - muP_post.get(p, np.nan) for p in muP_post}
        else:
            team_rcp_post = {}
            deltaP_team_post = {}

        # pre/post team WPA rate for the whole trade event (season-long sub-periods)
        season_pb = player_box[player_box["season_end_year"] == season]
        team_pb = season_pb[season_pb["team_id"] == team_id]
        pre = team_pb[team_pb["game_date"] < trade_date]
        post = team_pb[team_pb["game_date"] >= trade_date]
        games_pre = pre["game_id"].nunique()
        games_post = post["game_id"].nunique()

        if games_pre < MIN_GAMES_FOR_SNAPSHOT or games_post < MIN_GAMES_FOR_SNAPSHOT:
            excluded_no_min_games += 1
            continue

        wpa_time_pre = pre["tWPA"].sum()
        wpa_time_post = post["tWPA"].sum()
        wpa_rate_pre = wpa_time_pre / games_pre
        wpa_rate_post = wpa_time_post / games_post
        delta_wpa_time = wpa_rate_post - wpa_rate_pre

        standings = team_games_asof(engine, season, trade_date)
        objective = classify_objective(team_id, season, trade_date, standings)
        if objective["objetivo_sazonal"] is None:
            excluded_no_snapshot += 1
            continue

        team_abbr = tmeta.set_index("team_id").loc[team_id, "team_abbreviation"]

        any_gap_filled = False
        player_rows_for_event = []
        for _, prow in grp.iterrows():
            raw_pos = (prow["player_position"] or "").strip()
            posicao = RAW_TRADE_POS_TO_GROUP.get(raw_pos)
            if posicao is None:
                fb = season_positions_full[
                    (season_positions_full["player_id"] == prow["player_id"])
                    & (season_positions_full["team_id"] == team_id)
                    & (season_positions_full["season_end_year"] == season)
                ]
                posicao = fb["posicao"].iloc[0] if not fb.empty else None
            if posicao is None or posicao not in deltaP_team:
                continue

            dP = deltaP_team[posicao]
            preenche_lacuna = "Preenche Lacuna" if dP < 0 else "Redundante"
            any_gap_filled = any_gap_filled or (dP < 0)

            dP_post = deltaP_team_post.get(posicao, np.nan)
            if pd.isna(dP_post):
                gap_fechada = "Sem dado pos-troca"
            elif dP < 0:
                gap_fechada = "Sim" if dP_post >= 0 else ("Melhorou" if dP_post > dP else "Nao")
            else:
                gap_fechada = "N/A (nao era lacuna)"

            player_rows_for_event.append({
                "trade_id": prow["trade_id"],
                "team_id": team_id,
                "team_abbreviation": team_abbr,
                "season_end_year": season,
                "trade_date": trade_date.date(),
                "player_id": prow["player_id"],
                "player_name": prow["player_name"],
                "posicao": posicao,
                "RCP_pos_bar_pre_trade": team_rcp.get(posicao, np.nan),
                "muP_asof": muP_asof.get(posicao, np.nan),
                "deltaP": dP,
                "preenche_lacuna": preenche_lacuna,
                "RCP_pos_bar_post_trade": team_rcp_post.get(posicao, np.nan),
                "deltaP_post_trade": dP_post,
                "gap_fechada": gap_fechada,
            })

        if not player_rows_for_event:
            continue

        classification_rows.extend(player_rows_for_event)
        deltaP_values = [r["deltaP"] for r in player_rows_for_event]
        impact_rows.append({
            "trade_ids": ",".join(str(t) for t in trade_ids),
            "team_id": team_id,
            "team_abbreviation": team_abbr,
            "season_end_year": season,
            "trade_date": trade_date.date(),
            "n_players_acquired": len(player_rows_for_event),
            "trade_classification": "Preenche Lacuna" if any_gap_filled else "Redundante",
            "min_deltaP_acquired": float(min(deltaP_values)),
            "games_pre": int(games_pre),
            "games_post": int(games_post),
            "WPA_time_pre": float(wpa_time_pre),
            "WPA_time_post": float(wpa_time_post),
            "WPA_rate_pre": float(wpa_rate_pre),
            "WPA_rate_post": float(wpa_rate_post),
            "delta_WPA_time": float(delta_wpa_time),
            "conference": objective["conference"],
            "conf_rank_pre_trade": objective["conf_rank"],
            "win_pct_pre_trade": objective["win_pct_asof"],
            "objetivo_sazonal": objective["objetivo_sazonal"],
        })

    df_class = pd.DataFrame(classification_rows)
    df_impact = pd.DataFrame(impact_rows)

    # Pre-trade archetype (NOT the full-season data/team_season_clusters.csv, which mixes
    # in post-trade games and would make cluster circular with the outcome it's used to
    # explain - see src/wpa_pretrade_clusters.py). Run build_wpa_pretrade_clusters.py first.
    # NOTE: as of the 2013-2025 extension, wpa_pretrade_clusters.csv itself still only
    # covers 2019-2021 -- rows outside that range simply get cluster/cluster_name = NaN
    # via this left-join (a known, separate follow-up: extending that script too).
    clusters = pd.read_csv(pretrade_clusters_path, parse_dates=["trade_date"])
    clusters = clusters.rename(columns={"cluster_pre_trade": "cluster",
                                         "cluster_name_pre_trade": "cluster_name"})
    df_class["trade_date"] = pd.to_datetime(df_class["trade_date"])
    df_impact["trade_date"] = pd.to_datetime(df_impact["trade_date"])
    df_class = df_class.merge(clusters, on=["team_id", "season_end_year", "trade_date"], how="left")
    df_impact = df_impact.merge(clusters, on=["team_id", "season_end_year", "trade_date"], how="left")
    df_class["trade_date"] = df_class["trade_date"].dt.date
    df_impact["trade_date"] = df_impact["trade_date"].dt.date

    classification_path.parent.mkdir(parents=True, exist_ok=True)
    df_class.to_csv(classification_path, index=False)
    df_impact.to_csv(impact_path, index=False)

    print(f"\nExcluded (acquiring team's pre-trade WPA_acc unstable/near-zero): {excluded_unstable_team}")
    print(f"Excluded (< {MIN_GAMES_FOR_SNAPSHOT} team games pre or post trade date): {excluded_no_min_games}")
    print(f"Excluded (insufficient games for conference-rank snapshot): {excluded_no_snapshot}")
    print(f"\nSaved {len(df_class)} player-level rows to {classification_path}")
    print(f"Saved {len(df_impact)} trade-event rows to {impact_path}")
    print(f"\nTrade classification counts (event-level):")
    print(df_impact["trade_classification"].value_counts().to_string())
    print(f"\nObjetivo sazonal counts (event-level):")
    print(df_impact["objetivo_sazonal"].value_counts().to_string())

    return df_class, df_impact


def main():
    engine = get_engine()
    player_box = load_espn_player_box(engine)
    pretrade_clusters_path = PROJECT_DIR / "data" / "wpa_pretrade_clusters.csv"
    run_analysis(
        engine, player_box, range(START_YEAR, END_YEAR + 1),
        CLASSIFICATION_PATH, IMPACT_PATH, pretrade_clusters_path,
        label=f"({START_YEAR}-{END_YEAR})",
    )


if __name__ == "__main__":
    main()
