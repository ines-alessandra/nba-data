#!/usr/bin/env python3
"""
Runner: fits the reference archetype space on full-season features, then scores
every team at every trade-date snapshot needed by build_wpa_trade_analysis.py
(the same universe of trade events), producing data/wpa_pretrade_clusters.csv with:

  cluster_pre_trade / cluster_name_pre_trade   - games BEFORE trade_date only.
    Used to diagnose/classify the trade (H1/H2) - see wpa_pretrade_clusters.py for
    why this must never use post-trade data.

  cluster_post_trade / cluster_name_post_trade - games FROM trade_date onward only.
    DESCRIPTIVE ONLY (an outcome measure, symmetric to delta_WPA_time, expressed as
    an archetype instead of a WPA rate). Never used to classify a trade - only to
    narrate "team X was Cluster A before this trade, Cluster B after".

SEASON RANGE: 2013-2025 (matches build_wpa_trade_analysis_extended.py's extended
WPA pipeline). Originally hardcoded to wpa_common.START_YEAR/END_YEAR (2019-2021,
an ESPN-oWPA/dWPA/tWPA-only constraint that does NOT apply here -- this module's
features come from silver.fact_team_gamelogs box-score columns, available for the
full 2013-2025 range). Extending this was the "not-yet-done follow-up" flagged in
build_wpa_trade_analysis_extended.py's docstring. Local SEASON_RANGE below, not
wpa_common's constants, so this script's range can't silently drift back to
2019-2021 if wpa_common's legacy constant changes for unrelated reasons.

Must run BEFORE build_wpa_trade_analysis.py / build_wpa_trade_analysis_extended.py.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR, get_engine, season_date_bounds  # noqa: E402
from wpa_pretrade_clusters import (  # noqa: E402
    fetch_raw_gamelogs, compute_team_features, fit_reference_pipeline, score_snapshot,
)
from build_wpa_trade_analysis import load_incoming_players  # noqa: E402

OUT_PATH = PROJECT_DIR / "data" / "wpa_pretrade_clusters.csv"
SEASON_RANGE = range(2013, 2026)
START_YEAR, END_YEAR = min(SEASON_RANGE), max(SEASON_RANGE)


def main():
    print("=" * 70)
    print("  Pre-Trade Team Archetype Clustering (fixes full-season circularity)")
    print(f"  Season range: {START_YEAR}-{END_YEAR}")
    print("=" * 70)

    engine = get_engine()
    raw = fetch_raw_gamelogs(engine, START_YEAR, END_YEAR)
    print(f"Loaded {len(raw):,} team-game rows for {START_YEAR}-{END_YEAR}.")

    # 1. Reference space: full-season features across all 90 team-seasons studied
    full_season_features = compute_team_features(raw, cutoff_date=None)
    print(f"Built full-season feature matrix: {full_season_features.shape}")
    pipeline = fit_reference_pipeline(full_season_features)
    print("\nCluster archetypes (by centroid net/off/def rating, pace):")
    profile = pipeline["centroid_profile"].copy()
    profile["name"] = profile.index.map(pipeline["cluster_names"])
    print(profile.to_string())

    # 2. Trade universe: same (season, trade_date) snapshots build_wpa_trade_analysis.py needs
    incoming = load_incoming_players(engine, START_YEAR, END_YEAR)
    bounds = {s: season_date_bounds(engine, s) for s in range(START_YEAR, END_YEAR + 1)}
    keep_mask = incoming.apply(
        lambda r: bounds[r["season_end_year"]][0] <= r["trade_date"] <= bounds[r["season_end_year"]][1],
        axis=1,
    )
    incoming = incoming[keep_mask]
    unique_snapshots = incoming[["season_end_year", "trade_date"]].drop_duplicates()
    print(f"\nScoring {len(unique_snapshots)} trade-date snapshots x 30 teams (pre AND post)...")

    all_pre, all_post = [], []
    for _, row in unique_snapshots.iterrows():
        season, cutoff = int(row["season_end_year"]), row["trade_date"]

        pre_features = compute_team_features(raw, cutoff_date=cutoff, season_end_year=season)
        if not pre_features.empty:
            scored_pre = score_snapshot(pipeline, pre_features)
            scored_pre["trade_date"] = cutoff.date()
            all_pre.append(scored_pre)

        post_features = compute_team_features(raw, start_date=cutoff, season_end_year=season)
        if not post_features.empty:
            scored_post = score_snapshot(pipeline, post_features)
            scored_post = scored_post.rename(columns={
                "cluster_pre_trade": "cluster_post_trade",
                "cluster_name_pre_trade": "cluster_name_post_trade",
            })
            scored_post["trade_date"] = cutoff.date()
            all_post.append(scored_post)

    df_pre = pd.concat(all_pre, ignore_index=True)
    df_post = pd.concat(all_post, ignore_index=True)
    df_out = df_pre.merge(df_post, on=["team_id", "season_end_year", "trade_date"], how="left")

    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} team-snapshot cluster rows to {OUT_PATH}")
    print("Pre-trade archetype counts:")
    print(df_out["cluster_name_pre_trade"].value_counts().to_string())
    print("Post-trade archetype counts (descriptive only, not used in H1/H2 tests):")
    print(df_out["cluster_name_post_trade"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
