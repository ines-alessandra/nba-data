#!/usr/bin/env python3
"""
NBA Team-Season Style-Only Archetype Clustering (Complementary Analysis)
Author: Data Scientist

Context: the primary clustering (fit_clusters.py, k=4) mixes two conceptually
different axes in a single feature matrix -- "how good is this team" (net rating
and its ~7 near-collinear derivatives, which dominate the ANOVA ranking) and
"how does this team play" (pace, shot selection, rebounding, ball movement).
That clustering is statistically valid, but its public framing as "playstyle
clusters" overstates how much of it is actually style. This script isolates the
style axis on its own, so the two dimensions (competitive tier vs. playing
style) can be examined independently and then cross-tabulated.

This script:
1. Loads data/team_season_features.csv (raw features, no prior cluster label).
2. Restricts the feature set to variables that describe HOW a team plays,
   excluding anything that is fundamentally an outcome/quality proxy (net
   rating and its derivatives, off/def rating, points scored/allowed, shooting
   percentages, win rates). Shot volume (attempts) is kept as a style signal;
   shot accuracy (makes, percentages) is dropped because it blends style with
   execution quality. See STYLE_FEATURES / EXCLUDED-AS-QUALITY comment below.
3. Runs the same elbow / silhouette / Davies-Bouldin validation sweep (k=2..10)
   used for the primary clustering, to check whether k=4 is a reasonable
   choice on this feature set too (it is not assumed -- it is checked).
4. Fits the final K-Means (k=4) on the style-only feature set and runs a
   univariate ANOVA per feature to see which style traits actually
   differentiate the archetypes.
5. Auto-labels each archetype from its most extreme "signature" feature
   (pace, 3PA volume, rebounding, assists, turnovers, steals, blocks) instead
   of hardcoding a name -> cluster_id dictionary, since K-Means label order is
   arbitrary and this repository's data is refreshed periodically.
6. Cross-tabulates the style archetype against the existing performance-tier
   cluster (data/team_season_clusters.csv) and runs a chi-square test of
   independence + Cramer's V, to quantify how much (if at all) playing style
   and competitive tier overlap.
7. Saves outputs to data/team_style_archetypes.csv,
   data/style_archetype_anova_results.csv, and three plots under viz/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

from team_clustering_common import season_relative_zscore, cluster_full_names

# Features that describe HOW a team plays (pace, shot-selection volume,
# rebounding emphasis, ball movement, turnovers, defensive pressure/physicality)
# as opposed to features that describe HOW WELL it plays.
#
# Excluded as quality/outcome proxies (NOT used here): avg/std_net_rating,
# avg/std_plus_minus, adjusted_net_rating, avg_net_rating_vs_pos_srs,
# avg_net_rating_first10/last10, trend_net_rating, clutch_win_pct,
# avg/std_off_rating, avg/std_def_rating, avg/std_pts, avg/std_pts_allowed,
# and all shooting *_pct / *_m (makes) columns -- accuracy and scoring output
# are execution/quality signals, not style choices. Only attempt volume
# (fga, fg3a, fta) is kept, since "how many/what kind of shots a team takes"
# is a strategic choice independent of whether those shots go in.
STYLE_FEATURES = [
    'avg_possessions', 'std_possessions',
    'avg_fga', 'std_fga', 'avg_fg3a', 'std_fg3a', 'avg_fta', 'std_fta',
    'avg_oreb', 'std_oreb', 'avg_dreb', 'std_dreb', 'avg_reb', 'std_reb',
    'avg_ast', 'std_ast', 'avg_tov', 'std_tov',
    'avg_stl', 'std_stl', 'avg_blk', 'std_blk',
    'avg_pf', 'std_pf',
]

# Signature features used only to auto-generate a short, human-readable tag per
# archetype (whichever feature is most extreme relative to the other clusters).
# This is a labeling convenience, not a claim that a cluster is defined by a
# single stat -- the full profile is in the saved heatmap / ANOVA table.
SIGNATURE_FEATURES = {
    'avg_possessions': ('Ritmo Acelerado', 'Ritmo Lento'),
    'avg_fg3a': ('Ataque Perimetral (alto volume de 3s)', 'Ataque no Garrafão (baixo volume de 3s)'),
    'avg_reb': ('Domínio no Rebote', 'Fragilidade no Rebote'),
    'avg_ast': ('Bola em Movimento (alta assistência)', 'Jogo Individual (baixa assistência)'),
    'avg_tov': ('Alto Turnover', 'Time Cuidadoso (baixo turnover)'),
    'avg_stl': ('Defesa de Pressão (muitos roubos)', 'Defesa Passiva (poucos roubos)'),
    'avg_blk': ('Proteção de Aro', 'Aro Desguarnecido'),
}

FINAL_K = 4


def run_validation(features_pca: np.ndarray, output_plot_path: Path, k_range=range(2, 11)) -> pd.DataFrame:
    """Elbow / Silhouette / Davies-Bouldin sweep on the style-only feature space."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        results.append({
            'k': k,
            'Inertia (WCSS)': kmeans.inertia_,
            'Silhouette Score': silhouette_score(features_pca, labels, random_state=42),
            'Davies-Bouldin Index': davies_bouldin_score(features_pca, labels),
        })
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("     STYLE-ONLY CLUSTERING VALIDATION METRICS (k=2..10)   ")
    print("=" * 60)
    print(results_df.to_string(index=False, formatters={
        'Inertia (WCSS)': '{:,.2f}'.format,
        'Silhouette Score': '{:.6f}'.format,
        'Davies-Bouldin Index': '{:.6f}'.format,
    }))
    print("=" * 60)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#9467bd', '#8c564b', '#17becf']

    sns.lineplot(x='k', y='Inertia (WCSS)', data=results_df, marker='o', ax=axes[0], color=colors[0], linewidth=2.5, markersize=8)
    axes[0].set_title('Elbow Curve — Estilo (WCSS)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xticks(list(k_range))

    sns.lineplot(x='k', y='Silhouette Score', data=results_df, marker='o', ax=axes[1], color=colors[1], linewidth=2.5, markersize=8)
    axes[1].set_title('Silhouette Score — Estilo', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xticks(list(k_range))

    sns.lineplot(x='k', y='Davies-Bouldin Index', data=results_df, marker='o', ax=axes[2], color=colors[2], linewidth=2.5, markersize=8)
    axes[2].set_title('Davies-Bouldin Index — Estilo', fontsize=14, fontweight='bold', pad=12)
    axes[2].set_xticks(list(k_range))

    plt.suptitle('Validação da Clusterização de Estilo Puro (features de qualidade excluídas)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved style validation plot to: {output_plot_path}")

    return results_df


def label_clusters(cluster_means_z: pd.DataFrame) -> dict:
    """Auto-generate a short descriptive tag per cluster from its most extreme signature feature."""
    labels = {}
    for cluster in cluster_means_z.index:
        best_feat, best_z, best_abs = None, 0.0, -1.0
        for feat in SIGNATURE_FEATURES:
            z = cluster_means_z.loc[cluster, feat]
            if abs(z) > best_abs:
                best_abs, best_feat, best_z = abs(z), feat, z
        pos_name, neg_name = SIGNATURE_FEATURES[best_feat]
        labels[cluster] = pos_name if best_z > 0 else neg_name
    return labels


def main():
    np.random.seed(42)
    project_dir = Path(__file__).resolve().parent.parent
    features_csv = project_dir / "data" / "team_season_features.csv"
    performance_csv = project_dir / "data" / "team_season_clusters.csv"

    if not features_csv.exists():
        print(f"Error: {features_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} team-seasons from {features_csv}")

    missing = [c for c in STYLE_FEATURES if c not in df.columns]
    if missing:
        print(f"Error: expected style feature columns missing: {missing}")
        sys.exit(1)

    # 1. Preprocess (style-only feature subset). Standardized PER SEASON, not
    # pooled across the full 2013-2025 span -- pace and 3PA volume both have a
    # real secular trend over that range, and pooled standardization would let
    # KMeans partly cluster on era instead of style (see
    # team_clustering_common.py docstring for the full explanation; the same
    # issue was found and fixed for the primary competitive-archetype
    # clustering in fit_clusters.py).
    X = df[STYLE_FEATURES]
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=STYLE_FEATURES, index=X.index)
    X_scaled = season_relative_zscore(X_imputed, df['season_end_year'])
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled.values)
    print(f"Style-only feature matrix: {len(STYLE_FEATURES)} features -> {pca.n_components_} PCA components (95% variance)")

    # 2. Validate k for the style-only space
    validation_plot = project_dir / "viz" / "style_archetype_validation.png"
    run_validation(X_pca, validation_plot)

    # 3. Fit final K-Means (k=4, mirrors the primary clustering's granularity so the
    #    two typologies can be cross-tabulated on comparable terms)
    kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
    style_labels = kmeans.fit_predict(X_pca)
    df['style_cluster'] = style_labels

    # 4. ANOVA F-test per style feature across the k style archetypes
    anova_results = []
    X_imputed_arr = X_imputed.values
    groups = [X_imputed_arr[style_labels == i] for i in range(FINAL_K)]
    for idx, feat in enumerate(STYLE_FEATURES):
        f_stat, p_val = stats.f_oneway(*[g[:, idx] for g in groups])
        anova_results.append({'Feature': feat, 'F-Statistic': f_stat, 'p-value': p_val})
    anova_df = pd.DataFrame(anova_results).sort_values(by='F-Statistic', ascending=False)

    print("\n" + "=" * 60)
    print("      STYLE FEATURES ANOVA F-TEST (ALL FEATURES)          ")
    print("=" * 60)
    print(anova_df.to_string(index=False, formatters={'F-Statistic': '{:,.4f}'.format, 'p-value': '{:.2e}'.format}))
    print("=" * 60)

    anova_out = project_dir / "data" / "style_archetype_anova_results.csv"
    anova_df.to_csv(anova_out, index=False)
    print(f"Saved ANOVA results to: {anova_out}")

    # 5. Cluster profiling (z-score heatmap) + auto-labels
    cluster_means = df.groupby('style_cluster')[STYLE_FEATURES].mean()
    row_means = cluster_means.mean(axis=0)
    row_stds = cluster_means.std(axis=0).replace(0, 1.0)
    cluster_means_z = (cluster_means - row_means) / row_stds

    style_names = label_clusters(cluster_means_z)
    df['style_label'] = df['style_cluster'].map(style_names)

    print("\n" + "=" * 60)
    print("      AUTO-GENERATED STYLE ARCHETYPE LABELS                ")
    print("=" * 60)
    for c in sorted(style_names):
        n = (df['style_cluster'] == c).sum()
        print(f"  Style {c}: {style_names[c]}  (n={n} team-seasons)")
    print("=" * 60)

    cluster_means_z_t = cluster_means_z.T.reindex(anova_df['Feature'].tolist())
    cluster_means_z_t.columns = [f"S{c}: {style_names[c]}" for c in cluster_means_z_t.columns]

    plt.figure(figsize=(10, 10))
    sns.heatmap(cluster_means_z_t, cmap='PuOr', center=0, annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={'label': 'Z-Score (relativo entre arquétipos de estilo)'})
    plt.title('Perfil dos Arquétipos de Estilo Puro\n(Features ordenadas por poder de diferenciação - ANOVA F)',
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Arquétipo de Estilo', fontsize=11)
    plt.ylabel('Features de Estilo', fontsize=11)
    plt.tight_layout()
    heatmap_path = project_dir / "viz" / "style_archetype_heatmap.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved style archetype heatmap to: {heatmap_path}")

    # 6. Cross-tabulate against the existing performance-tier clustering
    if not performance_csv.exists():
        print(f"\nWarning: {performance_csv} not found -- skipping style x performance crosstab.")
        print("(Run src/fit_clusters.py first to generate it.)")
        df_out = df[['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year',
                     'style_cluster', 'style_label']]
    else:
        df_perf_full = pd.read_csv(performance_csv)
        # Derived from centroid profile -- KMeans label order is arbitrary per
        # fit, see team_clustering_common.py. Must run on the full CSV (which
        # still has avg_net_rating/avg_possessions), before subsetting columns.
        performance_names = cluster_full_names(df_perf_full, cluster_col='cluster')
        df_perf = df_perf_full[['team_id', 'season_end_year', 'cluster']]
        df_perf = df_perf.rename(columns={'cluster': 'performance_cluster'})
        df_merged = pd.merge(
            df[['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year',
                'style_cluster', 'style_label']],
            df_perf, on=['team_id', 'season_end_year'], how='inner'
        )
        df_merged['performance_label'] = df_merged['performance_cluster'].map(performance_names)

        crosstab = pd.crosstab(df_merged['style_cluster'], df_merged['performance_cluster'])
        crosstab.index = [f"S{c}: {style_names[c]}" for c in crosstab.index]
        crosstab.columns = [performance_names[c] for c in crosstab.columns]

        chi2, p_val, dof, _ = stats.chi2_contingency(crosstab)
        n = crosstab.values.sum()
        min_dim = min(crosstab.shape) - 1
        cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else float('nan')

        print("\n" + "=" * 80)
        print("   ESTILO x DESEMPENHO: TABELA DE CONTINGÊNCIA E TESTE DE INDEPENDÊNCIA  ")
        print("=" * 80)
        print(crosstab.to_string())
        print("-" * 80)
        print(f"Qui-quadrado de independência: chi2={chi2:.4f}, dof={dof}, p-value={p_val:.4e}")
        print(f"Cramer's V (força da associação, 0=independente, 1=totalmente dependente): {cramers_v:.4f}")
        if p_val < 0.05:
            strength = "fraca" if cramers_v < 0.2 else ("moderada" if cramers_v < 0.4 else "forte")
            print(f"Interpretação: associação estatisticamente significativa (p<0.05), de força {strength}.")
            print("Isso é compatível com 'estilo' e 'nível competitivo' sendo dimensões distintas,")
            print("porém não totalmente independentes na prática da NBA.")
        else:
            print("Interpretação: não há evidência de associação entre estilo e nível competitivo (p>=0.05).")
        print("=" * 80)

        plt.figure(figsize=(9, 7))
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        sns.heatmap(crosstab_pct, annot=crosstab.values, fmt='d', cmap='BuPu', cbar_kws={'label': '% dentro do arquétipo de estilo (cor)'})
        plt.title(f"Estilo de Jogo x Nível Competitivo\n(Qui-quadrado p={p_val:.2e}, Cramér's V={cramers_v:.3f})",
                  fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Arquétipo de Desempenho (clusterização original)', fontsize=10)
        plt.ylabel('Arquétipo de Estilo (clusterização complementar)', fontsize=10)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        crosstab_path = project_dir / "viz" / "style_vs_performance_crosstab.png"
        plt.savefig(crosstab_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved style x performance crosstab heatmap to: {crosstab_path}")

        df_out = df_merged

    # 7. Save final dataset
    output_csv = project_dir / "data" / "team_style_archetypes.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved final style archetype dataset to: {output_csv}")


if __name__ == '__main__':
    main()
