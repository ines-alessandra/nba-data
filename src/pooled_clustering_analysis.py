#!/usr/bin/env python3
"""
NBA Team-Season Pooled Clustering and Non-Parametric Salary Inequality Analysis
Author: Sports Data Scientist

This script:
1. Reloads the team_season_features dataset for all seasons (2019-2025).
2. Performs a single global K-Means clustering (k=4) on the PCA-reduced space (retaining 95% variance) of the pooled dataset.
3. Assigns fixed cluster labels (cluster_fixed) to all team-seasons.
4. Recomputes full-season Gini weighted by minutes directly from the Silver layer
   (load_full_season_gini) using the same validated formula as salary_inequality_metrics.py,
   instead of reading the orphaned/stale gold.team_inequality_metrics table (see that
   function's docstring for why it was replaced).
5. Computes temporal trends, boxplots (global and annual panel), Kruskal-Wallis, and within-cluster Spearman correlations.
6. Tracks cluster and salary inequality trajectories for iconic teams (BOS, GSW, LAL, OKT).
7. Saves all generated plots to the viz/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sqlalchemy import create_engine
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from salary_inequality_metrics import calculate_gini_weighted_minutes
from team_clustering_common import cluster_full_names, cluster_colors


def load_full_season_gini(engine) -> pd.DataFrame:
    """
    Recomputes Gini weighted by minutes for the FULL season (no pre/post split),
    using the same validated formula as salary_inequality_metrics.py.

    Replaces a previous read from gold.team_inequality_metrics, a table with no
    generating script anywhere in this repository. Its gini_weighted_minutes
    values were all negative (min -0.78, max -0.13) -- inconsistent with the
    correct formula (verified range 0.26-0.76 in data/nba_trade_deadline_inequality.csv)
    and not recoverable by a simple sign flip (correlation of only 0.55 against
    the correctly recomputed values), so it was treated as a stale/orphaned
    artifact rather than reconciled.
    """
    sql_query = """
    SELECT
        pg.team_id,
        g.season_end_year,
        pg.player_id,
        s.salary_inflation_adjusted,
        SUM(pg.minutes) AS minutes_played
    FROM silver.fact_player_gamelogs pg
    JOIN silver.dim_games g ON pg.game_id = g.game_id
    JOIN silver.fact_player_salaries s
      ON pg.player_id = s.player_id
     AND g.season_end_year = s.season_end_year
    WHERE g.game_type = 'Regular Season'
      AND g.season_end_year BETWEEN 2013 AND 2025
      AND s.salary_inflation_adjusted IS NOT NULL
      AND s.salary_inflation_adjusted > 0
    GROUP BY pg.team_id, g.season_end_year, pg.player_id, s.salary_inflation_adjusted
    HAVING SUM(pg.minutes) >= 1.0;
    """
    df_snapshots = pd.read_sql(sql_query, engine)

    records = []
    grouped = df_snapshots.groupby(['team_id', 'season_end_year'])
    for (team_id, season), group in grouped:
        S = group['salary_inflation_adjusted'].values
        M = group['minutes_played'].values
        if len(S) < 5:  # same minimum roster size validation as salary_inequality_metrics.py
            continue
        records.append({
            'team_id': team_id,
            'season_end_year': season,
            'gini_weighted_minutes': calculate_gini_weighted_minutes(S, M),
        })
    return pd.DataFrame(records)


def run_pooled_analysis(db_url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    np.random.seed(42) # Replicabilidade
    
    # 1. Connect to DB and load tables
    try:
        engine = create_engine(db_url)
        df_features = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "team_season_features.csv")
        df_clusters = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "team_season_clusters.csv")
        df_metrics = load_full_season_gini(engine)
        df_sos = pd.read_sql("SELECT * FROM gold.team_sos;", con=engine)
    except Exception as e:
        print(f"Error loading tables: {e}")
        sys.exit(1)

    print(f"Loaded {len(df_features)} feature records, {len(df_metrics)} metric records, and {len(df_sos)} SOS records.")

    # Translate season_year in SOS to season_end_year
    df_sos['season_end_year'] = df_sos['season_year'].apply(lambda x: int(x.split('-')[0]) + 1)

    metadata_cols = ['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year']
    existing_metadata = [col for col in metadata_cols if col in df_features.columns]

    # Reuse the cluster labels already computed by fit_clusters.py instead of
    # re-fitting KMeans here (see cluster_sankey_migration.py for why an
    # independent refit is a recurring source of bugs in this codebase).
    df_features = pd.merge(df_features, df_clusters[['team_id', 'season_end_year', 'cluster']],
                            on=['team_id', 'season_end_year'], how='left')
    df_features['cluster_fixed'] = df_features['cluster']

    # Cluster identities derived from centroid profile -- KMeans label order is
    # arbitrary per fit, see team_clustering_common.py. Must be computed here,
    # from df_features (which still has the raw avg_net_rating/avg_possessions
    # columns), not after the merge below drops everything but cluster_fixed.
    cluster_names = cluster_full_names(df_features, cluster_col='cluster_fixed')
    cluster_colors_map = cluster_colors(df_features, cluster_col='cluster_fixed')

    # Merge cluster labels, inequality metrics, and performance data
    df_merged = pd.merge(df_features[existing_metadata + ['cluster_fixed']], df_metrics, on=['team_id', 'season_end_year'])
    # Merge with SOS using all common keys to prevent name conflict suffixes (_x, _y)
    df_merged = pd.merge(df_merged, df_sos, on=['team_id', 'season_end_year', 'team_abbreviation', 'season_year'])

    # Clean missing values
    df_analysis = df_merged.dropna(subset=['gini_weighted_minutes', 'srs_rating', 'cluster_fixed']).copy()
    _sey_min, _sey_max = int(df_analysis['season_end_year'].min()), int(df_analysis['season_end_year'].max())
    print(f"Merged analysis dataset contains {len(df_analysis)} records (seasons {_sey_min}-{_sey_max}).")

    df_analysis['cluster_desc'] = df_analysis['cluster_fixed'].map(cluster_names)
    
    # --- TAREFA 2.A: EVOLUÇÃO TEMPORAL ---
    # Group by season and cluster to get average Gini
    cluster_annual_avg = df_analysis.groupby(['season_end_year', 'cluster_fixed'])['gini_weighted_minutes'].mean().unstack()
    league_annual_avg = df_analysis.groupby('season_end_year')['gini_weighted_minutes'].mean()
    
    print("\n" + "="*70)
    print("      ANNUAL LEAGUE-WIDE AND CLUSTER-WISE INEQUALITY AVERAGE (POOLED) ")
    print("="*70)
    print(f"{'Season':<8} | {'C0':<9} | {'C1':<9} | {'C2':<9} | {'C3':<9} | {'League Avg':<10}")
    print("-" * 70)
    for season in sorted(df_analysis['season_end_year'].unique()):
        c0 = cluster_annual_avg.loc[season, 0] if 0 in cluster_annual_avg.columns and not np.isnan(cluster_annual_avg.loc[season, 0]) else None
        c1 = cluster_annual_avg.loc[season, 1] if 1 in cluster_annual_avg.columns and not np.isnan(cluster_annual_avg.loc[season, 1]) else None
        c2 = cluster_annual_avg.loc[season, 2] if 2 in cluster_annual_avg.columns and not np.isnan(cluster_annual_avg.loc[season, 2]) else None
        c3 = cluster_annual_avg.loc[season, 3] if 3 in cluster_annual_avg.columns and not np.isnan(cluster_annual_avg.loc[season, 3]) else None
        la = league_annual_avg.loc[season]
        
        c0_str = f"{c0:.4f}" if c0 is not None else "  -  "
        c1_str = f"{c1:.4f}" if c1 is not None else "  -  "
        c2_str = f"{c2:.4f}" if c2 is not None else "  -  "
        c3_str = f"{c3:.4f}" if c3 is not None else "  -  "
        
        print(f"{season:<8} | {c0_str:<9} | {c1_str:<9} | {c2_str:<9} | {c3_str:<9} | {la:<10.4f}")
    print("="*70)

    # Plot 1: Line plot of annual averages
    plt.figure(figsize=(10, 6.5))
    colors = [cluster_colors_map[c] for c in range(4)]  # color follows archetype identity, not cluster_fixed integer
    
    # Plot clusters
    for c in range(4):
        # Only plot if we have non-null values for this cluster
        if c in cluster_annual_avg.columns:
            series = cluster_annual_avg[c].dropna()
            plt.plot(
                series.index, 
                series.values, 
                marker='o', 
                linewidth=2.5, 
                markersize=8, 
                color=colors[c], 
                label=cluster_names[c]
            )
            
    # Plot League Average
    plt.plot(
        league_annual_avg.index, 
        league_annual_avg.values, 
        marker='s', 
        linestyle='--', 
        linewidth=2.5, 
        markersize=8, 
        color='black', 
        label='League Average'
    )
    plt.title('Evolução do Gini Ponderado por Minutos por Cluster Fixo (POOLED K-Means)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Temporada (Ano Fim)', fontsize=12)
    plt.ylabel('Gini Ponderado por Minutos (Médias)', fontsize=12)
    plt.xticks(sorted(df_analysis['season_end_year'].unique()))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_temporal_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pooled_temporal_evolution.png")

    # --- TAREFA 2.B: BOXPLOTS ---
    # 1. Global Boxplot
    plt.figure(figsize=(9, 6))
    ordered_clusters = [cluster_names[i] for i in range(4)]
    sns.boxplot(
        x='cluster_desc', 
        y='gini_weighted_minutes', 
        data=df_analysis,
        order=ordered_clusters,
        palette=colors,
        width=0.55
    )
    sns.stripplot(
        x='cluster_desc', 
        y='gini_weighted_minutes', 
        data=df_analysis,
        order=ordered_clusters,
        color='black', 
        alpha=0.3, 
        size=4.5, 
        jitter=0.2
    )
    plt.title(f'Distribuição Global do Gini Ponderado por Minutos por Cluster Fixo ({_sey_min}-{_sey_max})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Arquétipo Competitivo (Fixo)', fontsize=12)
    plt.ylabel('Gini Ponderado por Minutos', fontsize=12)
    plt.xticks(range(4), [c.split(':')[0] for c in ordered_clusters])
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_boxplot_aggregate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pooled_boxplot_aggregate.png")

    # 2. Panel of annual boxplots
    seasons = sorted(df_analysis['season_end_year'].unique())
    ncols = 4
    nrows = int(np.ceil(len(seasons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    for idx, season in enumerate(seasons):
        df_season = df_analysis[df_analysis['season_end_year'] == season]
        sns.boxplot(
            x='cluster_fixed', 
            y='gini_weighted_minutes', 
            data=df_season, 
            palette=colors,
            order=range(4),
            ax=axes[idx],
            width=0.6
        )
        sns.stripplot(
            x='cluster_fixed', 
            y='gini_weighted_minutes', 
            data=df_season, 
            color='black',
            order=range(4),
            alpha=0.4,
            size=4.5,
            jitter=0.15,
            ax=axes[idx]
        )
        axes[idx].set_title(f"Temporada {season-1}-{str(season)[2:]}", fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Arquétipos Competitivos', fontsize=10)
        axes[idx].set_ylabel('Gini Ponderado por Minutos', fontsize=10)
        axes[idx].set_xticklabels(['C0', 'C1', 'C2', 'C3'])
        
    for idx in range(len(seasons), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Comparativo da Desigualdade por Cluster Fixo em Cada Temporada ({seasons[0]}-{seasons[-1]})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_annual_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pooled_annual_boxplots.png")

    # --- TAREFA 2.C: KRUSKAL-WALLIS TEST BY YEAR ---
    print("\n" + "="*80)
    print("      KRUSKAL-WALLIS H-TESTS BY SEASON (POOLED CLUSTERS)")
    print("="*80)
    print(f"{'Season':<8} | {'H-Statistic':<12} | {'p-value':<12} | {'Significativo?':<15}")
    print("-" * 65)
    
    for season in seasons:
        df_season = df_analysis[df_analysis['season_end_year'] == season]
        
        # Collect non-empty groups (with at least 2 observations)
        groups = []
        group_names = []
        for c in range(4):
            vals = df_season[df_season['cluster_fixed'] == c]['gini_weighted_minutes'].values
            if len(vals) >= 2:
                groups.append(vals)
                group_names.append(c)
                
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            is_sig = p_val < 0.05
            print(f"{season:<8} | {h_stat:>12.4f} | {p_val:>12.2e} | {str(is_sig):<15} (Presente: {group_names})")
            
            # Post-hoc pairwise Mann-Whitney U test with Bonferroni correction
            if is_sig:
                pairs = []
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        pairs.append((group_names[i], group_names[j]))
                        
                if len(pairs) > 0:
                    alpha_adjusted = 0.05 / len(pairs)
                    print(f"    Post-hoc U tests (Bonferroni alpha_adj = {alpha_adjusted:.4f}):")
                    for c1, c2 in pairs:
                        g1 = df_season[df_season['cluster_fixed'] == c1]['gini_weighted_minutes'].values
                        g2 = df_season[df_season['cluster_fixed'] == c2]['gini_weighted_minutes'].values
                        u_stat, p_val_pair = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                        if p_val_pair < alpha_adjusted:
                            print(f"      * Cluster {c1} vs. {c2} (p = {p_val_pair:.2e}) - Significativo")
        else:
            print(f"{season:<8} | {'Insuf. Clusters':^12} | {'-':^12} | {'False':<15}")
        print("-" * 65)
    print("="*80)

    # --- TAREFA 2.D: WITHIN-CLUSTER SPEARMAN CORRELATIONS ---
    print("\n" + "="*80)
    print("       SPEARMAN CORRELATIONS WITHIN FIXED CLUSTERS (ALL YEARS COMBINED) ")
    print("="*80)
    print(f"{'Cluster':<25} | {'Spearman (rho)':<15} | {'p-value':<12} | {'N':<6} | {'Significativo (5%)?':<15}")
    print("-" * 80)
    
    cluster_correlations = []
    
    for c in range(4):
        df_c = df_analysis[df_analysis['cluster_fixed'] == c]
        x_c = df_c['gini_weighted_minutes'].values
        y_c = df_c['srs_rating'].values
        
        rho, p_val = stats.spearmanr(x_c, y_c)
        is_sig = p_val < 0.05
        cluster_correlations.append({
            'Cluster': c,
            'Label': cluster_names[c],
            'rho': rho,
            'p_val': p_val,
            'N': len(df_c)
        })
        print(f"{cluster_names[c]:<25} | {rho:>15.5f} | {p_val:>12.2e} | {len(df_c):<6} | {str(is_sig):<15}")
    print("="*80)

    # Plot 3: LOESS Scatter plots by cluster
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for c in range(4):
        df_c = df_analysis[df_analysis['cluster_fixed'] == c]
        sns.regplot(
            x='gini_weighted_minutes', 
            y='srs_rating', 
            data=df_c, 
            ax=axes[c],
            lowess=True,
            color=colors[c],
            scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 's': 45},
            line_kws={'color': 'black', 'linewidth': 2}
        )
        
        # Get stats
        stats_row = next(item for item in cluster_correlations if item['Cluster'] == c)
        rho_val = stats_row['rho']
        p_val = stats_row['p_val']
        
        axes[c].set_title(f"{cluster_names[c]}\n(N = {len(df_c)})", fontsize=13, fontweight='bold', pad=8)
        axes[c].set_xlabel('Gini Ponderado por Minutos', fontsize=11)
        axes[c].set_ylabel('Performance (SRS Rating)', fontsize=11)
        
        # Add text box
        text_str = f"Spearman \u03c1 = {rho_val:.4f}\np = {p_val:.2e}"
        axes[c].text(
            0.05, 0.95, text_str, 
            transform=axes[c].transAxes, 
            fontsize=10, 
            fontweight='semibold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray')
        )
        
    # Monografia, não slide: sem título geral embutido na imagem -- a
    # legenda da figura no capítulo já identifica o gráfico. Os títulos de
    # cada painel (arquétipo + N) e as caixas de rho/p por painel continuam,
    # pois são parte da leitura do gráfico, não narrativa de slide.
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_cluster_scatter_loess.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pooled_cluster_scatter_loess.png")

    # --- TAREFA 2.E: TRACKING SPECIFIC TEAMS (BONUS) ---
    # Track BOS, GSW, LAL, OKC (Oklahoma City Thunder - check abbreviation: 'OKC')
    # Let's verify OKC abbreviation in df_analysis
    all_teams = df_analysis['team_abbreviation'].unique()
    okc_abbr = 'OKC' if 'OKC' in all_teams else ('OKL' if 'OKL' in all_teams else 'OKC')
    # If not OKC, let's use LAC or MIA
    if okc_abbr not in all_teams:
        okc_abbr = 'LAC' # LA Clippers
        
    target_teams = ['BOS', 'GSW', 'LAL', okc_abbr]
    df_targets = df_analysis[df_analysis['team_abbreviation'].isin(target_teams)].copy()
    
    # Sort
    df_targets = df_targets.sort_values(by=['team_abbreviation', 'season_end_year'])
    
    plt.figure(figsize=(12, 7.5))
    team_colors = {'BOS': '#007A33', 'GSW': '#1D428A', 'LAL': '#552583', okc_abbr: '#007AC1'}
    # Distinct diagonal offset per team so C-labels fan out instead of stacking on top of
    # each other whenever two teams land on a similar Gini value in the same season.
    label_offsets = {'BOS': (-16, 12), 'GSW': (16, 12), 'LAL': (-16, -16), okc_abbr: (16, -16)}

    for team in target_teams:
        team_data = df_targets[df_targets['team_abbreviation'] == team]
        plt.plot(
            team_data['season_end_year'],
            team_data['gini_weighted_minutes'],
            marker='o',
            linewidth=2.5,
            markersize=9,
            color=team_colors[team],
            label=f"{team} ({df_analysis[df_analysis['team_abbreviation']==team]['team_name'].iloc[0]})"
        )

        # Annotate each point with the cluster fixed number
        for idx, row in team_data.iterrows():
            plt.annotate(
                f"C{row['cluster_fixed']}",
                (row['season_end_year'], row['gini_weighted_minutes']),
                textcoords="offset points",
                xytext=label_offsets[team],
                ha='center',
                va='center',
                fontsize=10.5,
                fontweight='bold',
                color=team_colors[team],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75)
            )

    plt.title('Evolução do Gini Ponderado e Rastreamento de Transição de Clusters (BOS, GSW, LAL, ' + okc_abbr + ')', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Temporada (Ano Fim)', fontsize=12)
    plt.ylabel('Gini Ponderado por Minutos', fontsize=12)
    plt.xticks(sorted(df_analysis['season_end_year'].unique()))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        facecolor='white',
        framealpha=0.9
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_team_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pooled_team_trajectories.png")
    
    # 8. Save updated clusters table to DB just in case
    # gold.team_season_clustering_results already exists, but we can save our new classifications
    try:
        with engine.begin() as conn:
            df_analysis[['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year', 'cluster_fixed', 'gini_weighted_minutes', 'srs_rating']].to_sql(
                name="team_season_clustering_results_fixed",
                con=conn,
                schema="gold",
                if_exists="replace",
                index=False
            )
            print("Successfully saved updated classifications to: gold.team_season_clustering_results_fixed")
    except Exception as e:
        print(f"Warning: Failed to save updated classifications to DB. Error: {e}")

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    viz_folder = project_dir / "viz"
    
    # Retrieve DB URL
    DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
    DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    
    # Check if OKC or another exists
    run_pooled_analysis(DB_URL, viz_folder)
