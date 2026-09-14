#!/usr/bin/env python3
"""
Pre-trade-only team archetype clustering.

The original `data/team_season_clusters.csv` (from src/fit_clusters.py) labels each
team-SEASON using FULL-SEASON stats (avg pts, ratings, pace, etc. over all 82 games,
including games played AFTER any mid-season trade). Using that label to explain or
stratify a trade's OWN outcome is circular: part of what defines the label (the
post-trade portion of the season) is downstream of the very trade being evaluated.

This module fixes that by re-scoring each team at the SAME pre-trade cutoff date
already used for the deltaP/muP computation (see wpa_common.py / build_wpa_trade_analysis.py):

  1. A REFERENCE archetype space (imputer -> season-relative z-score -> PCA(0.95) ->
     KMeans k=4) is fit ONCE on full-season features across every team-season in the
     study population (this defines "what an NBA team archetype generally looks
     like" from the real study population - fitting the reference space on aggregate
     league patterns is standard practice, not circular in the same way). Originally
     fit on the 90 team-seasons of 2019-2021 (ESPN-only window); extended to all 13
     seasons (2013-2025, 390 team-seasons) alongside the rest of the WPA pipeline --
     z-scoring is done PER SEASON, not pooled, because pace/3PA volume/etc. have a
     real secular trend over that span and pooled standardization would let KMeans
     partly cluster on era instead of archetype (same fix as team_clustering_common.py
     / fit_style_archetypes.py).
  2. Each team is then SCORED (not re-fit) at every pre-trade cutoff date needed by
     the trade analysis, using ONLY games strictly before that date - the same
     temporal discipline already applied to WPA/RCP/deltaP. A snapshot is scored
     against ITS OWN season's mean/std (learned from the full-season reference fit),
     never against the snapshot's own partial-season sample.

Feature formulas mirror src/build_team_features.py exactly, parameterized by an
optional cutoff_date.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import get_engine, MIN_GAMES_FOR_SNAPSHOT  # noqa: E402
from team_clustering_common import ARCHETYPE_INFO, season_relative_zscore  # noqa: E402

NUMERIC_METRICS = [
    'pts', 'pts_allowed', 'possessions', 'off_rating', 'def_rating', 'net_rating',
    'minutes', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta',
    'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'pf', 'plus_minus'
]

CLUSTER_K = 4
RANDOM_STATE = 42


def fetch_raw_gamelogs(engine, start_year: int, end_year: int) -> pd.DataFrame:
    query = """
        SELECT
            f.game_id, f.team_id, f.opponent_team_id, f.is_home, f.wl,
            f.pts, f.pts_allowed, f.possessions, f.off_rating, f.def_rating, f.net_rating,
            f.minutes, f.fgm, f.fga, f.fg_pct, f.fg3m, f.fg3a, f.fg3_pct,
            f.ftm, f.fta, f.ft_pct, f.oreb, f.dreb, f.reb, f.ast, f.tov, f.stl, f.blk,
            f.pf, f.plus_minus,
            g.game_date, g.season_year, g.season_end_year,
            t.team_abbreviation, t.team_name,
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
        df = pd.read_sql(text(query), con=conn, params={"start_year": start_year, "end_year": end_year})
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def compute_team_features(raw_df: pd.DataFrame, cutoff_date: pd.Timestamp = None,
                           start_date: pd.Timestamp = None,
                           team_id: int = None, season_end_year: int = None) -> pd.DataFrame:
    """Same feature formulas as build_team_features.py, restricted to games in
    [start_date, cutoff_date) when given, and optionally to a single (team, season).
    Pass cutoff_date alone for a PRE-trade snapshot (games < trade_date); pass
    start_date alone for a POST-trade snapshot (games >= trade_date) - the latter
    is descriptive only (an outcome measure, like delta_WPA_time), never used to
    diagnose/classify a trade (that would reintroduce the circularity this module
    exists to avoid)."""
    df = raw_df
    if season_end_year is not None:
        df = df[df["season_end_year"] == season_end_year]
    if team_id is not None:
        df = df[df["team_id"] == team_id]
    if start_date is not None:
        df = df[df["game_date"] >= start_date]
    if cutoff_date is not None:
        df = df[df["game_date"] < cutoff_date]

    min_srs_by_season = raw_df.groupby('season_end_year')['opp_srs_rating'].min().to_dict()
    df_sorted = df.sort_values(by=['team_id', 'season_end_year', 'game_date']).copy()

    records = []
    for (tid, syear), group in df_sorted.groupby(['team_id', 'season_end_year']):
        if len(group) < MIN_GAMES_FOR_SNAPSHOT:
            continue
        rec = {
            'team_id': int(tid), 'season_end_year': int(syear),
            'team_abbreviation': group['team_abbreviation'].iloc[0],
            'games_played': len(group),
        }
        for col in NUMERIC_METRICS:
            rec[f'avg_{col}'] = float(group[col].mean())
            rec[f'std_{col}'] = float(group[col].std()) if len(group) > 1 else 0.0

        group_chrono = group.sort_values('game_date')
        if len(group_chrono) >= 10:
            avg_first10 = float(group_chrono.head(10)['net_rating'].mean())
            avg_last10 = float(group_chrono.tail(10)['net_rating'].mean())
        else:
            avg_first10 = avg_last10 = float(group_chrono['net_rating'].mean())
        rec['avg_net_rating_first10'] = avg_first10
        rec['avg_net_rating_last10'] = avg_last10
        rec['trend_net_rating'] = avg_last10 - avg_first10

        clutch = group_chrono[group_chrono['plus_minus'].abs() <= 5]
        rec['clutch_win_pct'] = float((clutch['wl'] == 'W').mean()) if len(clutch) else 0.0

        vs_pos = group_chrono[group_chrono['opp_srs_rating'] > 0]
        rec['avg_net_rating_vs_pos_srs'] = float(vs_pos['net_rating'].mean()) if len(vs_pos) else 0.0

        min_srs = min_srs_by_season.get(syear, 0.0)
        opp_srs = group_chrono['opp_srs_rating'].fillna(0.0)
        weights = opp_srs - min_srs + 1.0
        net_ratings = group_chrono['net_rating'].fillna(0.0)
        rec['adjusted_net_rating'] = (
            float(np.average(net_ratings, weights=weights)) if weights.sum() > 0 else float(net_ratings.mean())
        )
        records.append(rec)

    return pd.DataFrame(records)


def fit_reference_pipeline(features_df: pd.DataFrame):
    """Fit imputer/season-relative-zscore/PCA/KMeans on full-season features (the
    study population -- 2013-2025, 390 team-seasons). Returns the fitted objects
    plus a cluster-name mapping derived from centroid profiles."""
    meta_cols = ['team_id', 'team_abbreviation', 'season_end_year', 'games_played']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(features_df[feature_cols]),
                          columns=feature_cols, index=features_df.index)

    # Season-relative z-score, NOT a pooled StandardScaler: see module docstring.
    # A snapshot scored later is standardized against ITS OWN season's mean/std,
    # so these per-season stats must be kept (not just the fitted scaler).
    season_means = X_imp.groupby(features_df['season_end_year']).mean()
    season_stds = X_imp.groupby(features_df['season_end_year']).std().replace(0, 1.0)
    X_scaled = season_relative_zscore(X_imp, features_df['season_end_year'])

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled.values)

    kmeans = KMeans(n_clusters=CLUSTER_K, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    profile_df = features_df[feature_cols].copy()
    profile_df['cluster'] = labels
    means = profile_df.groupby('cluster')[['avg_net_rating', 'avg_off_rating', 'avg_def_rating',
                                            'avg_possessions']].mean()

    # Name clusters from their centroid profile (net rating quality + pace), instead of
    # assuming a fixed KMeans label ordering (arbitrary per fit). Names come from
    # team_clustering_common.ARCHETYPE_INFO -- the single source of truth for archetype
    # names/colors across the whole project (Reconstrução/Defensivo/Ofensivo/Elite) --
    # not a locally-redefined set of labels, even though this is a separate clustering
    # fit (pre-trade-only feature space, see module docstring) from the main one.
    ordered_by_net = means['avg_net_rating'].sort_values()
    names = {}
    names[ordered_by_net.index[0]] = ARCHETYPE_INFO['rebuilding']['full']
    names[ordered_by_net.index[-1]] = ARCHETYPE_INFO['elite']['full']
    middle_two = ordered_by_net.index[1:-1]
    pace_rank = means.loc[middle_two, 'avg_possessions'].sort_values()
    names[pace_rank.index[0]] = ARCHETYPE_INFO['defensive']['full']
    names[pace_rank.index[-1]] = ARCHETYPE_INFO['offensive']['full']

    return {
        "imputer": imputer, "season_means": season_means, "season_stds": season_stds,
        "pca": pca, "kmeans": kmeans,
        "feature_cols": feature_cols, "cluster_names": names,
        "fit_labels": labels, "fit_team_season": features_df[['team_id', 'season_end_year']],
        "centroid_profile": means,
    }


def score_snapshot(pipeline: dict, snapshot_features: pd.DataFrame) -> pd.DataFrame:
    """Transform + predict (NOT re-fit) a new set of pre-trade feature rows through the
    already-fitted reference pipeline. Standardized against the snapshot's OWN
    season's mean/std (learned during the reference fit), never against the
    snapshot sample itself -- keeps a partial-season window from being
    standardized against its own small, biased sample."""
    X = snapshot_features[pipeline["feature_cols"]]
    X_imp = pd.DataFrame(pipeline["imputer"].transform(X), columns=pipeline["feature_cols"], index=X.index)

    seasons = snapshot_features['season_end_year']
    means = pipeline["season_means"].reindex(seasons).values
    stds = pipeline["season_stds"].reindex(seasons).values
    X_scaled = (X_imp.values - means) / stds

    X_pca = pipeline["pca"].transform(X_scaled)
    labels = pipeline["kmeans"].predict(X_pca)

    out = snapshot_features[['team_id', 'season_end_year']].copy()
    out['cluster_pre_trade'] = labels
    out['cluster_name_pre_trade'] = out['cluster_pre_trade'].map(pipeline["cluster_names"])
    return out
