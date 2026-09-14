#!/usr/bin/env python3
"""
NBA Team-Season Clustering: Archetype Legend / Reference Card
Author: Data Scientist

A single reference figure -- one card per cluster (0..3) -- meant to be the
"key" every other cluster figure in this project should be read against
(cluster_heatmap.png, temporal_evolution.png, team_trajectories.png,
sankey_cluster_migration.png, sankey_additional_plots.png,
pooled_team_trajectories.png, pooled_temporal_evolution.png all already derive
their name/color from team_clustering_common.derive_cluster_archetypes, so
consistency across figures is structural, not just visual convention -- this
card just makes that mapping explicit and puts real numbers on it).

Each archetype has exactly ONE name (single Portuguese word: Reconstrução,
Defensivo, Ofensivo, Elite -- never a compound like "Rebuilding / Lottery"),
defined once in team_clustering_common.ARCHETYPE_INFO and reused everywhere.

Each card shows:
  - color + cluster id + archetype name (from team_clustering_common)
  - the 3 features with the highest ANOVA F-statistic for that cluster's mean
    (cluster_anova_results.csv), so "why this name" is traceable to numbers
  - n (team-seasons) and season range
  - 3 representative team-seasons: the ones whose avg_net_rating sits closest
    to the cluster's own mean (typical members, not extremes/outliers)

A banner above the cards explains, in plain language, why these groups are
called "arquétipos" and not just "clusters" or "grupos".
"""

import textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from team_clustering_common import derive_cluster_archetypes, ARCHETYPE_INFO, extract_feature_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLUSTERS_CSV = PROJECT_DIR / "data" / "team_season_clusters.csv"
ANOVA_CSV = PROJECT_DIR / "data" / "cluster_anova_results.csv"
OUT_PNG = PROJECT_DIR / "viz" / "legenda_arquetipos_cluster.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
BANNER_BG = "#eef2f7"
BANNER_EDGE = "#94a3b8"

FEATURE_LABELS = {
    'avg_plus_minus': 'Plus/Minus médio', 'avg_net_rating': 'Net Rating médio',
    'adjusted_net_rating': 'Net Rating ajust. (SoS)', 'avg_net_rating_vs_pos_srs': 'Net Rating vs. times fortes',
    'avg_net_rating_last10': 'Net Rating (últimos 10j.)', 'avg_net_rating_first10': 'Net Rating (10j. iniciais)',
    'avg_fg3_pct': '% de 3 pontos', 'avg_off_rating': 'Rating Ofensivo', 'avg_fg_pct': '% de arremessos de quadra',
    'std_possessions': 'Variação de posses (ritmo)', 'avg_dreb': 'Rebotes defensivos', 'avg_pts': 'Pontos por jogo',
    'avg_reb': 'Rebotes totais', 'avg_fgm': 'Cestas convertidas', 'avg_def_rating': 'Rating Defensivo',
    'clutch_games_count': 'Jogos decididos (clutch)', 'avg_tov': 'Turnovers', 'clutch_win_pct': '% vitórias no clutch',
    'avg_possessions': 'Posses por jogo (ritmo)',
}

BANNER_TEXT = (
    "O que é um “arquétipo” aqui?  Cada um dos 4 grupos é um PADRÃO TÍPICO e RECORRENTE de perfil "
    "competitivo — um “modelo” que se repete ao longo de 13 temporadas e 30 franquias diferentes, "
    "não uma categoria fixa de um time específico. Chamamos de arquétipo (e não só “cluster” ou "
    "“grupo”) porque o rótulo descreve um estilo competitivo generalizável — várias equipes, em "
    "anos e contextos diferentes, caem repetidamente no mesmo padrão estatístico de Net Rating e ritmo de "
    "jogo. O mesmo time passa por arquétipos diferentes ao longo do tempo (ver team_trajectories.png) — o "
    "arquétipo descreve a TEMPORADA, não a identidade permanente da franquia."
)


def fmt_feature(feat):
    return FEATURE_LABELS.get(feat, feat.replace('_', ' '))


def build():
    df = pd.read_csv(CLUSTERS_CSV)
    anova = pd.read_csv(ANOVA_CSV)
    archetypes = derive_cluster_archetypes(df, cluster_col='cluster')

    n_seasons_min, n_seasons_max = int(df['season_end_year'].min()), int(df['season_end_year'].max())

    cluster_stats = df.groupby('cluster').agg(
        n=('team_id', 'size'),
        avg_net_rating=('avg_net_rating', 'mean'),
        avg_possessions=('avg_possessions', 'mean'),
        avg_off_rating=('avg_off_rating', 'mean'),
        avg_def_rating=('avg_def_rating', 'mean'),
    )

    # Global top-3 differentiating features (ANOVA F is a property of the whole
    # partition, not of one cluster) -- what IS cluster-specific is each
    # cluster's own z-score on those same 3 features, so that's what each card
    # shows (same row-wise z-score already used for cluster_heatmap.png).
    top3_global = anova['Feature'].head(3).tolist()
    df_feat_only = extract_feature_matrix(df)
    df_feat_only['cluster'] = df['cluster']
    cluster_means_raw = df_feat_only.groupby('cluster').mean()
    row_mean = cluster_means_raw.mean(axis=0)
    row_std = cluster_means_raw.std(axis=0).replace(0, 1.0)
    cluster_z = (cluster_means_raw - row_mean) / row_std

    top_feats_by_cluster = {}
    for c in range(4):
        top_feats_by_cluster[c] = [(feat, cluster_z.loc[c, feat]) for feat in top3_global]

    examples = {}
    for c in range(4):
        sub = df[df['cluster'] == c].copy()
        mean_nr = cluster_stats.loc[c, 'avg_net_rating']
        sub['dist'] = (sub['avg_net_rating'] - mean_nr).abs()
        sub = sub.sort_values('dist')
        picked, seen_teams = [], set()
        for _, row in sub.iterrows():
            if row['team_abbreviation'] in seen_teams:
                continue
            picked.append(f"{row['team_abbreviation']} {int(row['season_end_year'])-1}-{str(int(row['season_end_year']))[2:]}")
            seen_teams.add(row['team_abbreviation'])
            if len(picked) == 3:
                break
        examples[c] = picked

    order = ['rebuilding', 'defensive', 'offensive', 'elite']
    cluster_by_key = {v: k for k, v in archetypes.items()}

    fig = plt.figure(figsize=(19, 7.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 5.6], hspace=0.1, wspace=0.08)

    # --- Banner: why "arquétipo" ---
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.axis('off')
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    banner_box = patches.FancyBboxPatch((0.005, 0.06), 0.99, 0.88, boxstyle="round,pad=0.015,rounding_size=0.03",
                                         linewidth=1.4, edgecolor=BANNER_EDGE, facecolor=BANNER_BG, zorder=1)
    ax_banner.add_patch(banner_box)
    wrapped_banner = "\n".join(textwrap.wrap(BANNER_TEXT, width=168))
    ax_banner.text(0.014, 0.5, wrapped_banner, ha='left', va='center', fontsize=10.6, color=INK_PRIMARY,
                   linespacing=1.6)

    # --- Cards ---
    axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for ax, key in zip(axes, order):
        c = cluster_by_key[key]
        info = ARCHETYPE_INFO[key]
        color = info['color']
        st = cluster_stats.loc[c]

        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(SURFACE)

        card = patches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02,rounding_size=0.04",
                                       linewidth=2.2, edgecolor=color, facecolor=color, alpha=0.06, zorder=1)
        ax.add_patch(card)
        border = patches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02,rounding_size=0.04",
                                         linewidth=2.2, edgecolor=color, facecolor='none', zorder=2)
        ax.add_patch(border)

        header = patches.FancyBboxPatch((0.08, 0.83), 0.84, 0.11, boxstyle="round,pad=0.01,rounding_size=0.025",
                                         linewidth=0, facecolor=color, zorder=3)
        ax.add_patch(header)
        ax.text(0.5, 0.885, f"Cluster {c}", ha='center', va='center', fontsize=13, fontweight='bold',
                color='white', zorder=4)

        ax.text(0.5, 0.755, f"Arquétipo: {info['full']}", ha='center', va='center', fontsize=15,
                fontweight='bold', color=INK_PRIMARY, zorder=4)

        ax.text(0.5, 0.695, f"n = {int(st['n'])} times-temporada  ({n_seasons_min}-{n_seasons_max})",
                ha='center', va='center', fontsize=9.5, color=INK_SECONDARY, style='italic', zorder=4)

        stats_lines = [
            f"Net Rating médio: {st['avg_net_rating']:+.2f}",
            f"Rating Ofensivo: {st['avg_off_rating']:.1f}",
            f"Rating Defensivo: {st['avg_def_rating']:.1f}",
            f"Posses/jogo (ritmo): {st['avg_possessions']:.1f}",
        ]
        y0 = 0.615
        for i, line in enumerate(stats_lines):
            ax.text(0.5, y0 - i * 0.052, line, ha='center', va='center', fontsize=10.3,
                    color=INK_PRIMARY, zorder=4, family='monospace')

        ax.plot([0.12, 0.88], [0.375, 0.375], color=color, linewidth=1.2, alpha=0.5, zorder=3)

        ax.text(0.5, 0.335, "Z-score neste cluster (top-3 ANOVA F global)", ha='center', va='center', fontsize=9,
                fontweight='bold', color=INK_SECONDARY, zorder=4)
        for i, (feat, z) in enumerate(top_feats_by_cluster[c]):
            ax.text(0.5, 0.29 - i * 0.045, f"{fmt_feature(feat)}: {z:+.2f}", ha='center', va='center', fontsize=8.6,
                    color=INK_PRIMARY, zorder=4)

        ax.text(0.5, 0.125, "Exemplos típicos", ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=INK_SECONDARY, zorder=4)
        ax.text(0.5, 0.075, "  •  ".join(examples[c]), ha='center', va='center', fontsize=8.6,
                color=INK_PRIMARY, zorder=4)

    fig.suptitle("Legenda de Referência: os 4 Arquétipos Competitivos da Clusterização NBA",
                 fontsize=17, fontweight='bold', y=1.01, color=INK_PRIMARY)
    fig.text(0.5, -0.015,
              "Nomes e cores derivados do centróide de cada cluster (avg_net_rating + avg_possessions), não de um "
              "índice fixo do KMeans — ver src/team_clustering_common.py. Use esta legenda para ler qualquer outro "
              "gráfico de cluster deste projeto.",
              ha='center', fontsize=9.3, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")
    for c in range(4):
        print(f"Cluster {c}: {examples[c]}")


if __name__ == "__main__":
    build()
