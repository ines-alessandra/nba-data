#!/usr/bin/env python3
"""
NBA Competitive Archetype Migration Sankey Diagram

NOTE ON TERMINOLOGY: these clusters mix a competitive-quality axis (net rating
and its derivatives dominate the ANOVA ranking in fit_clusters.py) with a
playing-style axis (pace, shot volume). They are referred to as "competitive
archetypes" rather than "playstyle clusters" for that reason. A style-only
clustering that isolates the style axis is available in
src/fit_style_archetypes.py.
Author: Sports Visualization Specialist & Data Scientist

This script:
1. Reloads the team_season_features dataset for all 210 records (2019-2025).
2. Standardizes, runs PCA (retaining 95% variance), and fits the global Pooled K-Means model (k=4).
3. Assigns cluster_fixed to all 210 team-seasons (ensuring continuous flows for all 30 franchises).
4. Formats data for a Plotly Sankey diagram (nodes and links) representing annual transitions.
5. Saves the interactive flow visualization to viz/sankey_cluster_migration.html.
6. Computes and prints migration statistics (persistence, top pathways, "bridge" clusters).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

from team_clustering_common import cluster_full_names, cluster_short_names, cluster_colors, derive_cluster_archetypes


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

def generate_sankey_migration(clusters_csv_path: str, output_html_path: str, output_png_path: str):
    # 1. Load team features
    if not os.path.exists(clusters_csv_path):
        print(f"Error: CSV file not found at {clusters_csv_path}")
        sys.exit(1)

    df_features = pd.read_csv(clusters_csv_path)
    print(f"Loaded {len(df_features)} records for Sankey migration analysis.")

    # 2. Reuse the cluster labels already computed by fit_clusters.py, instead of
    # re-fitting KMeans here. clusters_csv_path (team_season_clusters.csv) already
    # carries them in the 'cluster' column. Re-fitting independently was a past
    # source of bugs in this file (the previous refit briefly leaked 'cluster'
    # itself back in as an input feature) and risked silently diverging from the
    # canonical clustering if this script's preprocessing wasn't kept in perfect
    # lockstep with fit_clusters.py's (e.g. the season-relative standardization
    # fix -- see team_clustering_common.py -- had to be applied consistently
    # everywhere or clusters would disagree across scripts).
    df_features['cluster_fixed'] = df_features['cluster']
    
    # Define cluster descriptions (derived from centroid profile -- KMeans
    # label order is arbitrary per fit, see team_clustering_common.py)
    cluster_names = cluster_full_names(df_features, cluster_col='cluster_fixed')
    cluster_short = cluster_short_names(df_features, cluster_col='cluster_fixed')

    # Colors follow archetype identity, not the arbitrary cluster_fixed integer
    # (e.g. red always means Rebuilding, whichever integer KMeans gave it)
    cluster_colors_hex = cluster_colors(df_features, cluster_col='cluster_fixed')
    cluster_colors_rgba = {c: _hex_to_rgba(hexval, 0.35) for c, hexval in cluster_colors_hex.items()}
    
    # Sort dataset by team and year
    df_features = df_features.sort_values(by=['team_id', 'season_end_year'])
    seasons = sorted(df_features['season_end_year'].unique())
    
    # 3. Compute year-over-year transitions
    transitions = []
    
    for i in range(len(seasons) - 1):
        year_s = seasons[i]
        year_t = seasons[i+1]
        
        # Get active team clusters for year_s and year_t
        df_s = df_features[df_features['season_end_year'] == year_s][['team_id', 'cluster_fixed']].rename(columns={'cluster_fixed': 'c_source'})
        df_t = df_features[df_features['season_end_year'] == year_t][['team_id', 'cluster_fixed']].rename(columns={'cluster_fixed': 'c_target'})
        
        df_trans = pd.merge(df_s, df_t, on='team_id')
        df_trans['year_source'] = year_s
        df_trans['year_target'] = year_t
        
        transitions.append(df_trans)
        
    df_all_trans = pd.concat(transitions, ignore_index=True)
    
    # Count transitions
    df_counts = df_all_trans.groupby(['year_source', 'year_target', 'c_source', 'c_target']).size().reset_index(name='count')
    
    # --- SANKEY STRUCTURE SETUP ---
    # Nodes: N years * 4 clusters. Node mapping: index = (year - base_year) * 4 + cluster,
    # where base_year is derived from the data instead of a hardcoded season.
    base_year = seasons[0]
    node_labels = []
    node_colors = []
    
    for year in seasons:
        for c in range(4):
            node_labels.append(f"{year-1}-{str(year)[2:]}<br>C{c}: {cluster_short[c]}")
            node_colors.append(cluster_colors_hex[c])
            
    # Links
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    
    for idx, row in df_counts.iterrows():
        y_s = row['year_source']
        y_t = row['year_target']
        c_s = row['c_source']
        c_t = row['c_target']
        val = row['count']
        
        # Node indices
        src_idx = (y_s - base_year) * 4 + c_s
        tgt_idx = (y_t - base_year) * 4 + c_t
        
        link_sources.append(src_idx)
        link_targets.append(tgt_idx)
        link_values.append(val)
        link_colors.append(cluster_colors_rgba[c_s]) # Flow takes the color of the source cluster
        
    # Create the Plotly Figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 12,
            thickness = 16,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = node_colors
        ),
        link = dict(
            source = link_sources,
            target = link_targets,
            value = link_values,
            color = link_colors,
            hovertemplate = 'De %{source.label} para %{target.label}<br>Quantidade: <b>%{value} times</b><extra></extra>'
        )
    )])
    
    # Layout adjustments (width scales with number of seasons so nodes stay legible)
    fig.update_layout(
        title_text = f"<b>NBA Team Migration Flow between Competitive Archetypes ({seasons[0]}-{seasons[-1]})</b>",
        font_size = 10,
        width = max(1300, 190 * len(seasons)),
        height = 780,
        title_x = 0.5,
        title_font = dict(size=18, family="Arial")
    )
    
    # Save as HTML
    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html_path)
    print(f"Saved interactive Sankey diagram to: {output_html_path}")
    
    # Try saving static PNG using Kaleido
    try:
        fig.write_image(output_png_path, scale=2)
        print(f"Saved high-res static PNG to: {output_png_path}")
    except Exception as e:
        print(f"Warning: Static image export failed (Kaleido might not be installed). Error: {e}")
        
    # --- 4. MIGRATION STATISTICS ANALYSIS ---
    total_transitions = len(df_all_trans)
    persistence = df_all_trans[df_all_trans['c_source'] == df_all_trans['c_target']]
    pct_persistence = (len(persistence) / total_transitions) * 100
    
    print("\n" + "="*80)
    print("             NBA CLUSTER MIGRATION STATISTICAL ANALYSIS        ")
    print("="*80)
    print(f"Total de transições observadas (time-temporada consecutivas): {total_transitions}")
    print(f"Times que permaneceram no mesmo cluster: {len(persistence)} ({pct_persistence:.2f}%)")
    print(f"Times que mudaram de arquétipo competitivo: {total_transitions - len(persistence)} ({100 - pct_persistence:.2f}%)")
    print("-" * 80)
    
    # top 5 pathways (aggregated across all seasons)
    df_pathways = df_all_trans.groupby(['c_source', 'c_target']).size().reset_index(name='count')
    df_pathways = df_pathways.sort_values(by='count', ascending=False)
    
    print("\nTop 8 Caminhos de Transição Agregados:")
    print(f"{'Origem':<22} -> {'Destino':<22} | {'Contagem':<10} | {'Proporção (%)':<15}")
    print("-" * 80)
    for idx, row in df_pathways.head(8).iterrows():
        src_lbl = cluster_names[row['c_source']]
        tgt_lbl = cluster_names[row['c_target']]
        c_pct = (row['count'] / total_transitions) * 100
        print(f"{src_lbl:<22} -> {tgt_lbl:<22} | {row['count']:<10} | {c_pct:<15.2f}%")
    print("-" * 80)
    
    # Which cluster IDs are "Reconstrução" and "Elite" this run (not necessarily
    # 0 and 3 -- see team_clustering_common.py). Looked up by archetype KEY
    # ('rebuilding'/'elite'), not by substring-matching the display name --
    # the display name is just a single Portuguese word now and has no fixed
    # substring to match on.
    archetype_of = derive_cluster_archetypes(df_features, cluster_col='cluster_fixed')
    id_of_archetype = {key: c for c, key in archetype_of.items()}
    rebuilding_id = id_of_archetype['rebuilding']
    elite_id = id_of_archetype['elite']
    other_ids = [c for c in cluster_names if c not in (rebuilding_id, elite_id)]

    # Analyze transitions out of Rebuilding to see if they go to Elite directly or pass through the other two
    df_c0_out = df_all_trans[(df_all_trans['c_source'] == rebuilding_id) & (df_all_trans['c_target'] != rebuilding_id)]
    c0_out_total = len(df_c0_out)

    print(f"\nDestinos de times saindo de {cluster_names[rebuilding_id]} [Total de saídas = {c0_out_total}]:")
    for dest in other_ids + [elite_id]:
        count = len(df_c0_out[df_c0_out['c_target'] == dest])
        pct = (count / c0_out_total) * 100 if c0_out_total > 0 else 0
        print(f"  -> Para {cluster_names[dest]:<25}: {count} times ({pct:.2f}%)")

    # Analyze transitions into Elite to see where they came from
    df_c3_in = df_all_trans[(df_all_trans['c_target'] == elite_id) & (df_all_trans['c_source'] != elite_id)]
    c3_in_total = len(df_c3_in)

    print(f"\nOrigem dos times que ascenderam a {cluster_names[elite_id]} [Total de acessos = {c3_in_total}]:")
    for src in [rebuilding_id] + other_ids:
        count = len(df_c3_in[df_c3_in['c_source'] == src])
        pct = (count / c3_in_total) * 100 if c3_in_total > 0 else 0
        print(f"  <- Vindo de {cluster_names[src]:<25}: {count} times ({pct:.2f}%)")
    print("="*80)

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "team_season_clusters.csv"
    
    html_output = project_dir / "viz" / "sankey_cluster_migration.html"
    png_output = project_dir / "viz" / "sankey_cluster_migration.png"
    
    generate_sankey_migration(str(csv_path), str(html_output), str(png_output))
