#!/usr/bin/env python3
"""
NBA Team-Season Clustering: Shared Preprocessing
Author: Data Scientist

Every script that clusters team-seasons (fit_clusters.py, validate_clusters.py,
cluster_sankey_migration.py, sankey_additional_plots.py,
pooled_clustering_analysis.py) re-fits its own KMeans on the same
team_season_features.csv. That duplication is itself a risk -- it's exactly
how the season-relative-scaling fix below could silently get applied to some
of those scripts and not others. Centralizing the preprocessing here means
every script produces bit-identical clusters from the same inputs.

Season-relative standardization (the actual fix): the feature set spans
2013-2025, and several raw stats (pace, 3PA volume, points per game) have a
strong secular trend over that span -- the league genuinely plays faster and
shoots more threes now than in 2013. Standardizing each feature against the
POOLED 13-year mean/std (the original approach, fine for the narrower
2019-2025 window) made "fast for 2014" register as slow compared to "fast for
2024", so KMeans partly clustered on ERA instead of style: cluster
membership showed an almost clean 2013-2018 vs. 2019-2025 split instead of 4
archetypes present in every season. Standardizing each feature against its
OWN SEASON's mean/std instead (z-score relative to that year's league)
removes the secular trend: a feature value of +1.0 always means "one SD
above that season's league average", in any year.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

METADATA_COLS = ['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year']


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns and any remaining non-numeric columns."""
    existing_metadata = [c for c in METADATA_COLS if c in df.columns]
    df_features = df.drop(columns=existing_metadata)
    non_numeric = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df_features = df_features.drop(columns=non_numeric)
    return df_features


def season_relative_zscore(df_features: pd.DataFrame, season_series: pd.Series) -> pd.DataFrame:
    """Z-score each feature within its own season instead of pooled across all
    years (see module docstring)."""
    season_values = pd.Series(season_series.values, index=df_features.index)
    grouped = df_features.groupby(season_values)
    means = grouped.transform('mean')
    stds = grouped.transform('std').replace(0, 1.0)
    return (df_features - means) / stds


def fit_pooled_clusters(df_features_raw: pd.DataFrame, season_series: pd.Series, k: int = 4, random_state: int = 42):
    """Impute -> season-relative z-score -> PCA (95% variance) -> KMeans(k).
    Returns (labels, features_pca, pca, feature_names) so callers can reuse
    the same PCA space (e.g. for Agglomerative comparison or ANOVA)."""
    feature_names = df_features_raw.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    imputed = pd.DataFrame(imputer.fit_transform(df_features_raw), columns=feature_names, index=df_features_raw.index)

    scaled = season_relative_zscore(imputed, season_series)

    pca = PCA(n_components=0.95, random_state=random_state)
    features_pca = pca.fit_transform(scaled.values)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_pca)

    return labels, features_pca, pca, feature_names, imputed.values


# Canonical archetype identities. Kept separate from cluster integer IDs on
# purpose (see derive_cluster_archetypes docstring): color and name should
# always follow "which archetype is this" (e.g. red = Reconstrução), never
# "which integer did KMeans happen to assign this run".
#
# ONE name per archetype (single Portuguese word/term, no "X / Y" compounds --
# e.g. never "Rebuilding / Lottery") so every figure, slide and legend in this
# project can cite a single unambiguous label. 'full' and 'short' are kept as
# separate dict keys for backward compatibility with existing call sites, but
# both now hold the exact same single name.
ARCHETYPE_INFO = {
    'rebuilding': {'full': 'Reconstrução', 'short': 'Reconstrução', 'color': '#d62728'},
    'defensive':  {'full': 'Defensivo', 'short': 'Defensivo', 'color': '#2ca02c'},
    'offensive':  {'full': 'Ofensivo', 'short': 'Ofensivo', 'color': '#ff7f0e'},
    'elite':      {'full': 'Elite', 'short': 'Elite', 'color': '#1f77b4'},
}


def derive_cluster_archetypes(df_clustered: pd.DataFrame, cluster_col: str = 'cluster') -> dict:
    """Map each of the 4 cluster IDs to an archetype key ('rebuilding',
    'defensive', 'offensive', 'elite') from its own centroid profile (net
    rating quality + pace), instead of assuming a fixed integer -> name
    mapping.

    KMeans label order is arbitrary per fit -- it depends on centroid
    initialization, not on any inherent "0 = worst" ordering. Several scripts
    used to hardcode {0: 'Rebuilding', 1: 'Defensive', 2: 'Offensive', 3:
    'Elite'}, which happened to match by coincidence when every script fit
    the same data the same way. The season-relative-scaling fix changed which
    integer label each archetype gets, and every one of those hardcoded
    dicts went stale silently (3 of 4 labels ended up pointing at the wrong
    archetype). Deriving the mapping from the data every time removes that
    failure mode. Mirrors the pattern already used in wpa_pretrade_clusters.py.
    """
    means = df_clustered.groupby(cluster_col)[['avg_net_rating', 'avg_possessions']].mean()
    ordered_by_net = means['avg_net_rating'].sort_values()

    mapping = {}
    mapping[ordered_by_net.index[0]] = 'rebuilding'
    mapping[ordered_by_net.index[-1]] = 'elite'
    middle_two = ordered_by_net.index[1:-1]
    pace_rank = means.loc[middle_two, 'avg_possessions'].sort_values()
    mapping[pace_rank.index[0]] = 'defensive'
    mapping[pace_rank.index[-1]] = 'offensive'
    return mapping


def cluster_full_names(df_clustered: pd.DataFrame, cluster_col: str = 'cluster') -> dict:
    """{cluster_id: '2: Rebuilding / Lottery'}"""
    arch = derive_cluster_archetypes(df_clustered, cluster_col)
    return {c: f"{c}: {ARCHETYPE_INFO[key]['full']}" for c, key in arch.items()}


# Back-compat alias (same thing, name used by earlier callers).
derive_cluster_names = cluster_full_names


def cluster_short_names(df_clustered: pd.DataFrame, cluster_col: str = 'cluster') -> dict:
    """{cluster_id: 'Rebuilding'}"""
    arch = derive_cluster_archetypes(df_clustered, cluster_col)
    return {c: ARCHETYPE_INFO[key]['short'] for c, key in arch.items()}


def cluster_colors(df_clustered: pd.DataFrame, cluster_col: str = 'cluster') -> dict:
    """{cluster_id: '#d62728'} -- color follows archetype identity, not cluster_id."""
    arch = derive_cluster_archetypes(df_clustered, cluster_col)
    return {c: ARCHETYPE_INFO[key]['color'] for c, key in arch.items()}
