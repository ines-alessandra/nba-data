#!/usr/bin/env python3
"""
NBA Team-Season Temporal Cluster Evolution Analysis
Author: Data Scientist

This script:
1. Loads the team_season_clusters.csv dataset.
2. Group by season_end_year and cluster to get counts of teams in each profile.
3. Generates a multi-panel plot showing the absolute/proportional evolution of clusters over the years.
4. Generates a second plot showing the trajectory of specific teams across clusters.
5. Discusses trend patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from team_clustering_common import derive_cluster_names, cluster_short_names, cluster_colors

def run_temporal_analysis(csv_path: str, output_evolution_img: str, output_trajectory_img: str):
    # Load dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Group by season_end_year and cluster
    # Get count of teams in each cluster per season
    temporal_counts = df.groupby(['season_end_year', 'cluster']).size().unstack(fill_value=0)
    print("\n" + "="*50)
    print("      NUMBER OF TEAMS PER CLUSTER ACROSS SEASONS   ")
    print("="*50)
    print(temporal_counts.to_string())
    print("="*50)
    
    # Also get percentages
    temporal_pct = temporal_counts.div(temporal_counts.sum(axis=1), axis=0) * 100
    print("\nPROPORTION (%) OF TEAMS PER CLUSTER ACROSS SEASONS:")
    print(temporal_pct.to_string(float_format=lambda x: f"{x:.1f}%"))
    print("="*50)
    
    # Define names of clusters for labels (derived from centroid profile --
    # KMeans label order is arbitrary per fit, see team_clustering_common.py)
    cluster_names = derive_cluster_names(df)
    short_names = cluster_short_names(df)

    # Plot 1: Stacked Bar Chart & Cluster Trends
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Colors follow ARCHETYPE identity (team_clustering_common.ARCHETYPE_INFO), not
    # cluster integer position -- keeps this figure's palette identical to
    # legenda_arquetipos_cluster.png / sankey_additional_plots.png / sankey_cluster_migration.png,
    # regardless of which integer KMeans happened to assign each archetype in this fit.
    color_map = cluster_colors(df)
    colors = [color_map[i] for i in range(4)]
    
    # Stacked bar chart
    temporal_counts.plot(
        kind='bar', 
        stacked=True, 
        ax=axes[0], 
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8
    )
    axes[0].set_title('Distribuição de Clusters por Temporada', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel('Ano de Fim da Temporada', fontsize=11)
    axes[0].set_ylabel('Quantidade de Times (Total = 30)', fontsize=11)
    axes[0].set_xticklabels(temporal_counts.index, rotation=0)
    # Legend placed BELOW the axes, not inside it -- with value labels now drawn on the
    # bars, an inside legend (previously upper-left) sat right on top of the tallest early-
    # year Elite segments (e.g. 10 in 2015) and hid them.
    axes[0].legend([cluster_names[i] for i in range(4)], loc='upper center', bbox_to_anchor=(0.5, -0.14),
                    ncol=4, frameon=True, fontsize=9.5)

    # Value labels centered in each stacked segment (white, bold -- readable against all
    # 4 archetype colors, which are all mid-to-dark saturation).
    for container in axes[0].containers:
        axes[0].bar_label(container, label_type='center', fontsize=8.5, fontweight='bold', color='white')
    axes[0].set_ylim(0, 32)
    
    # Line chart showing individual cluster trends
    for i in range(4):
        axes[1].plot(
            temporal_counts.index, 
            temporal_counts[i], 
            marker='o', 
            linewidth=2.5, 
            markersize=8,
            color=colors[i],
            label=cluster_names[i]
        )
    axes[1].set_title('Evolução Temporal das Quantidades de Clusters', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel('Ano de Fim da Temporada', fontsize=11)
    axes[1].set_ylabel('Quantidade de Times', fontsize=11)
    axes[1].set_xticks(temporal_counts.index)
    axes[1].legend(loc='best')
    
    plt.suptitle(f'Evolução Temporal dos Clusters de Times da NBA ({temporal_counts.index.min()}-{temporal_counts.index.max()})', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_evolution_img, dpi=300, bbox_inches='tight')
    print(f"Saved temporal evolution plot to: {output_evolution_img}")
    plt.close()
    
    # 2. Track trajectories of specific teams
    # Let's select a few teams with interesting paths:
    # BOS (Boston Celtics) - went from 1 (defensive) to 2 (pace/offensive) to 3 (elite)
    # GSW (Golden State Warriors) - dynasty changes, rebuild in 2020
    # CLE (Cleveland Cavaliers) - post-LeBron rebuild (0) to playoff team (1) to contender (3)
    # LAL (Los Angeles Lakers) - championship team, pace changes
    target_teams = ['BOS', 'GSW', 'CLE', 'LAL']
    df_targets = df[df['team_abbreviation'].isin(target_teams)].copy()
    
    plt.figure(figsize=(12, 8))
    # We can plot each team's cluster trajectory over time
    # To plot it cleanly, we can map cluster numbers to y-axis with custom labels
    team_colors = {'BOS': '#007A33', 'GSW': '#1D428A', 'CLE': '#860038', 'LAL': '#552583'}
    
    # Sort by team and year
    df_targets = df_targets.sort_values(by=['team_abbreviation', 'season_end_year'])
    
    for team_idx, team in enumerate(target_teams):
        team_data = df_targets[df_targets['team_abbreviation'] == team]
        plt.plot(
            team_data['season_end_year'],
            team_data['cluster'],
            marker='o',
            linewidth=3,
            markersize=10,
            color=team_colors[team],
            label=f"{team} ({df[df['team_abbreviation']==team]['team_name'].iloc[0]})"
        )

        # Label only TRANSITIONS (first season, or any season whose cluster differs from
        # the team's own previous season) -- not every single point. Words like
        # "Reconstrução" are wider than the 1-year gap between points, so labeling every
        # point made consecutive same-archetype seasons (and different teams landing on the
        # same cluster in the same year) overlap into unreadable smears. A small per-team
        # vertical stagger (team_idx) further separates labels when two teams DO transition
        # into the same cluster in the same season.
        prev_cluster = None
        for idx, row in team_data.iterrows():
            if row['cluster'] == prev_cluster:
                prev_cluster = row['cluster']
                continue
            prev_cluster = row['cluster']
            plt.text(
                row['season_end_year'],
                row['cluster'] + 0.08 + team_idx * 0.075,
                short_names[row['cluster']],
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color=team_colors[team]
            )
            
    plt.title(f'Trajetória de Clusters para Franquias Selecionadas ({temporal_counts.index.min()}-{temporal_counts.index.max()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Temporada (Ano Fim)', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.yticks(range(4), [cluster_names[i] for i in range(4)], fontsize=10)
    plt.xticks(temporal_counts.index)
    plt.ylim(-0.5, 3.5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        facecolor='white',
        framealpha=0.9
    )
    plt.tight_layout()
    plt.savefig(output_trajectory_img, dpi=300, bbox_inches='tight')
    print(f"Saved team trajectories plot to: {output_trajectory_img}")
    plt.close()

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    csv_file = project_dir / "data" / "team_season_clusters.csv"
    out_evol = project_dir / "viz" / "temporal_evolution.png"
    out_traj = project_dir / "viz" / "team_trajectories.png"
    
    run_temporal_analysis(str(csv_file), str(out_evol), str(out_traj))
