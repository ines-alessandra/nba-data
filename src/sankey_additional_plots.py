#!/usr/bin/env python3
"""
NBA Competitive Archetype Migration: Additional Plots Panel (2x2)

NOTE ON TERMINOLOGY: these clusters mix a competitive-quality axis (net rating
and its derivatives dominate the ANOVA ranking in fit_clusters.py) with a
playing-style axis (pace, shot volume). They are referred to as "competitive
archetypes" rather than "playstyle clusters" for that reason. A style-only
clustering that isolates the style axis is available in
src/fit_style_archetypes.py.
Author: Sports Visualization Specialist & Data Scientist

This script:
1. Reloads the team_season_features dataset for all 210 records (2019-2025).
2. Performs the global Pooled K-Means model (k=4) to assign cluster_fixed.
3. Computes Year-over-Year consecutive transition counts.
4. Generates a 2x2 subplot panel:
   - (0, 0): Pie chart of competitive archetype persistence vs. migration.
   - (0, 1): Bar chart of destinations when exiting Rebuilding (C0).
   - (1, 0): Bar chart of origins when entering Elite Contenders (C3).
   - (1, 1): Bar chart of persistence (same cluster) by competitive archetype.
5. Saves the panel to viz/sankey_additional_plots.png at 300 DPI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from team_clustering_common import cluster_full_names, cluster_short_names, cluster_colors, derive_cluster_archetypes

def generate_additional_plots(clusters_csv_path: str, output_png_path: str):
    # 1. Load data
    if not os.path.exists(clusters_csv_path):
        print(f"Error: CSV not found at {clusters_csv_path}")
        sys.exit(1)

    df_features = pd.read_csv(clusters_csv_path)

    # 2. Reuse the cluster labels already computed by fit_clusters.py (see
    # cluster_sankey_migration.py for why this replaced an independent refit).
    df_features['cluster_fixed'] = df_features['cluster']
    
    # Sort
    df_features = df_features.sort_values(by=['team_id', 'season_end_year'])
    seasons = sorted(df_features['season_end_year'].unique())
    
    # 3. Calculate transitions
    transitions = []
    for i in range(len(seasons) - 1):
        year_s = seasons[i]
        year_t = seasons[i+1]
        
        df_s = df_features[df_features['season_end_year'] == year_s][['team_id', 'cluster_fixed']].rename(columns={'cluster_fixed': 'c_source'})
        df_t = df_features[df_features['season_end_year'] == year_t][['team_id', 'cluster_fixed']].rename(columns={'cluster_fixed': 'c_target'})
        
        df_trans = pd.merge(df_s, df_t, on='team_id')
        df_trans['year_source'] = year_s
        df_trans['year_target'] = year_t
        transitions.append(df_trans)
        
    df_all_trans = pd.concat(transitions, ignore_index=True)
    
    # --- STATISTICS ---
    total_trans = len(df_all_trans)
    persistence_df = df_all_trans[df_all_trans['c_source'] == df_all_trans['c_target']]
    n_persistence = len(persistence_df)
    n_migration = total_trans - n_persistence

    # Cluster identities derived from centroid profile -- KMeans label order is
    # arbitrary per fit, see team_clustering_common.py.
    full_names = cluster_full_names(df_features, cluster_col='cluster_fixed')
    short_names = cluster_short_names(df_features, cluster_col='cluster_fixed')
    colors = cluster_colors(df_features, cluster_col='cluster_fixed')
    all_ids = sorted(full_names.keys())
    # Looked up by archetype KEY ('rebuilding'/'elite'), not by substring-
    # matching the display name -- see cluster_sankey_migration.py for why.
    archetype_of = derive_cluster_archetypes(df_features, cluster_col='cluster_fixed')
    id_of_archetype = {key: c for c, key in archetype_of.items()}
    rebuilding_id = id_of_archetype['rebuilding']
    elite_id = id_of_archetype['elite']
    other_ids = [c for c in all_ids if c not in (rebuilding_id, elite_id)]

    # Exits from Rebuilding
    df_c0_out = df_all_trans[(df_all_trans['c_source'] == rebuilding_id) & (df_all_trans['c_target'] != rebuilding_id)]
    dest_order = other_ids + [elite_id]
    c0_dest_counts = df_c0_out['c_target'].value_counts().reindex(dest_order, fill_value=0)

    # Entrances into Elite
    df_c3_in = df_all_trans[(df_all_trans['c_target'] == elite_id) & (df_all_trans['c_source'] != elite_id)]
    orig_order = [rebuilding_id] + other_ids
    c3_orig_counts = df_c3_in['c_source'].value_counts().reindex(orig_order, fill_value=0)

    # Persistence counts per cluster
    persist_counts = persistence_df['c_source'].value_counts().reindex(all_ids, fill_value=0)
    
    # --- PLOTTING 2x2 PANEL ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Pie chart (Persistence vs. Migration)
    labels = ['Persistência\n(Mesmo Arquétipo)', 'Mudança\n(Novo Arquétipo)']
    sizes = [n_persistence, n_migration]
    pie_colors = ['#aec7e8', '#ffbb78'] # Soft blue, soft orange
    
    axes[0, 0].pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=pie_colors, 
        textprops={'fontsize': 11.5, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'w', 'linewidth': 1.5}
    )
    axes[0, 0].set_title(f'Estabilidade de Arquétipo Competitivo ({seasons[0]}-{seasons[-1]})\n(Total: {total_trans} transições time-temporada)', fontsize=13, fontweight='bold', pad=10)

    # Subplot 2: Destinations when exiting Rebuilding
    c0_dest_labels = [f'C{c}: {short_names[c]}' for c in dest_order]
    c0_dest_vals = c0_dest_counts.values
    c0_dest_colors = [colors[c] for c in dest_order]
    c0_out_total = len(df_c0_out)

    bars1 = axes[0, 1].bar(c0_dest_labels, c0_dest_vals, color=c0_dest_colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title(f'Destino dos Times ao Sair da Reconstrução (C{rebuilding_id})\n(Total: {c0_out_total} saídas observadas)', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].set_ylabel('Quantidade de Times', fontsize=11)
    axes[0, 1].set_ylim(0, max(c0_dest_vals) + 2)
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)} ({yval/sum(c0_dest_vals)*100:.1f}%)", ha='center', va='bottom', fontweight='bold', fontsize=10.5)

    # Subplot 3: Origins of Elite Contenders
    c3_orig_labels = [f'C{c}: {short_names[c]}' for c in orig_order]
    c3_orig_vals = c3_orig_counts.values
    c3_orig_colors = [colors[c] for c in orig_order]
    c3_in_total = len(df_c3_in)

    bars2 = axes[1, 0].bar(c3_orig_labels, c3_orig_vals, color=c3_orig_colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title(f'Origem dos Times que Acederam à Elite (C{elite_id})\n(Total: {c3_in_total} acessos observados)', fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].set_ylabel('Quantidade de Times', fontsize=11)
    axes[1, 0].set_ylim(0, max(c3_orig_vals) + 2)
    for bar in bars2:
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)} ({yval/sum(c3_orig_vals)*100:.1f}%)", ha='center', va='bottom', fontweight='bold', fontsize=10.5)

    # Subplot 4: Persistence counts by Cluster
    persist_labels = [f'C{c}: {short_names[c]}' for c in all_ids]
    persist_vals = persist_counts.values
    persist_colors = [colors[c] for c in all_ids]

    bars3 = axes[1, 1].bar(persist_labels, persist_vals, color=persist_colors, width=0.55, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title(f'Persistências por Cluster (Estabilidade de Cada Arquétipo)\n(Total: {n_persistence} manutenções consecutivas)', fontsize=13, fontweight='bold', pad=10)
    axes[1, 1].set_ylabel('Quantidade de Manutenções', fontsize=11)
    axes[1, 1].set_ylim(0, max(persist_vals) + 3)
    for bar in bars3:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{int(yval)} times", ha='center', va='bottom', fontweight='bold', fontsize=10.5)
        
    plt.suptitle(f'Análise Detalhada das Transições e Estabilidade de Arquétipos Competitivos na NBA ({seasons[0]}-{seasons[-1]})\n(Abordagem Pooled)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save panel in high resolution
    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved additional plots panel to: {output_png_path}")

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "team_season_clusters.csv"
    png_output = project_dir / "viz" / "sankey_additional_plots.png"
    
    generate_additional_plots(str(csv_path), str(png_output))
