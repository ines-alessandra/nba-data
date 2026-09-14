#!/usr/bin/env python3
"""
Slide 3 of the clustering presentation: the preprocessing/clustering pipeline
(imputação -> padronização por temporada -> PCA -> KMeans), drawn as an
illustrated 5-panel strip instead of the plain box-and-arrow style already
used for salary_methodology_flowchart.png. Each panel shows the data actually
being transformed (a real or realistically-shaped mini-visualization), not
just a text label, so the "why" of each step is visible instead of asserted.

Panels 3 (season-relative standardization) and 4 (PCA) use REAL numbers from
team_season_features.csv / the fitted PCA (league-average pace by season,
explained-variance curve) -- not illustrative fake data -- because those are
exactly the two steps a reader is most likely to ask "why do you need that?"
about. Panel 5 (KMeans) projects the REAL 31-dimensional PCA space onto its
first 2 components (PC1 x PC2) and colors every real team-season by its
actual fitted archetype, for visualization only -- standard practice, not a
literal claim that PC1 x PC2 alone separates the 4 archetypes cleanly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from team_clustering_common import (
    extract_feature_matrix, season_relative_zscore, ARCHETYPE_INFO, derive_cluster_archetypes,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_DIR / "data" / "team_season_features.csv"
CLUSTERS_CSV = PROJECT_DIR / "data" / "team_season_clusters.csv"
OUT_PNG = PROJECT_DIR / "viz" / "fluxograma_clusterizacao.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
MISSING_COLOR = "#e34948"
FILL_COLOR = "#eda100"
ACCENT = "#2a78d6"


def panel_titles_and_captions():
    return [
        ("1. Dados Brutos", "390 times-temporada × 59 variáveis,\ncom valores faltantes (vermelho)"),
        ("2. Imputação", "Faltantes preenchidos pela\nmediana da própria variável"),
        ("3. Padronização por Temporada", "Cada valor vira um z-score\nrelativo à SUA temporada"),
        ("4. Redução (PCA)", "59 variáveis → 31 componentes\n(95% da variância original)"),
        ("5. Agrupamento (KMeans, k=4)", "Times-temporada agrupados em\n4 arquétipos competitivos"),
    ]


def build():
    df = pd.read_csv(FEATURES_CSV)
    df_features = extract_feature_matrix(df)
    df_clusters = pd.read_csv(CLUSTERS_CSV)

    # --- Panel 1/2 illustrative matrix (real shape, synthetic values for legibility -- a
    # 390x59 heatmap would be unreadable at this size, so a representative 14x10 sample
    # is used, with the SAME imputation logic as the real pipeline) ---
    rng = np.random.default_rng(7)
    mat = rng.normal(0, 1, size=(14, 10))
    mask = rng.random((14, 10)) < 0.07
    col_median = np.median(np.where(mask, np.nan, mat), axis=0)
    mat_filled = mat.copy()
    for j in range(mat.shape[1]):
        mat_filled[mask[:, j], j] = np.nanmedian(np.where(mask[:, j], np.nan, mat[:, j]))
    mat_raw_display = np.ma.masked_array(mat, mask=mask)

    # --- Panel 3 real data: league-average pace by season, + one illustrative team at
    # the SAME raw pace (97 poss/game) in two different seasons, standardized ---
    pace_by_season = df.groupby('season_end_year')['avg_possessions'].mean()
    example_val = 97.0
    s1, s2 = 2014, 2024
    mean1, std1 = df[df.season_end_year == s1]['avg_possessions'].mean(), df[df.season_end_year == s1]['avg_possessions'].std()
    mean2, std2 = df[df.season_end_year == s2]['avg_possessions'].mean(), df[df.season_end_year == s2]['avg_possessions'].std()
    z1, z2 = (example_val - mean1) / std1, (example_val - mean2) / std2

    # --- Panel 4 real data: cumulative explained variance ---
    imputer = SimpleImputer(strategy='median')
    imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns, index=df_features.index)
    scaled = season_relative_zscore(imputed, df['season_end_year'])
    pca_full = PCA(random_state=42).fit(scaled.values)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = int(np.argmax(cum_var >= 0.95)) + 1

    # --- Panel 5 real data: PC1 x PC2, colored by real fitted archetype ---
    pca_95 = PCA(n_components=0.95, random_state=42)
    features_pca = pca_95.fit_transform(scaled.values)
    df_clusters = df_clusters.copy()
    archetype_of = derive_cluster_archetypes(df_clusters, cluster_col='cluster')
    colors_by_cluster = {c: ARCHETYPE_INFO[key]['color'] for c, key in archetype_of.items()}
    point_colors = df_clusters['cluster'].map(colors_by_cluster).values

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(24, 6.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    titles_captions = panel_titles_and_captions()

    # Panel 1
    ax = axes[0]
    cmap1 = plt.cm.Blues.copy()
    cmap1.set_bad(color=MISSING_COLOR)
    ax.imshow(mat_raw_display, cmap=cmap1, aspect='auto', vmin=-2.5, vmax=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel 2
    ax = axes[1]
    ax.imshow(mat_filled, cmap='Blues', aspect='auto', vmin=-2.5, vmax=2.5)
    for (i, j) in zip(*np.where(mask)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=FILL_COLOR, linewidth=2.2))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel 3: two stacked mini-axes (raw trend vs. standardized)
    ax = axes[2]
    ax.axis('off')
    sub_raw = ax.inset_axes([0.02, 0.56, 0.96, 0.42])
    sub_z = ax.inset_axes([0.02, 0.0, 0.96, 0.42])

    sub_raw.plot(pace_by_season.index, pace_by_season.values, color=INK_SECONDARY, linewidth=2, zorder=2)
    sub_raw.scatter([s1, s2], [example_val, example_val], color=ACCENT, s=55, zorder=4, edgecolor='black', linewidth=0.8)
    sub_raw.axhline(example_val, color=ACCENT, linestyle=':', linewidth=1.2, alpha=0.7, zorder=1)
    sub_raw.annotate(f"{s1}: {example_val:.0f} poss.", (s1, example_val), textcoords="offset points",
                      xytext=(0, 8), fontsize=7.3, ha='center', fontweight='bold', color=ACCENT)
    sub_raw.annotate(f"{s2}: {example_val:.0f} poss. (mesmo!)", (s2, example_val), textcoords="offset points",
                      xytext=(-8, 8), fontsize=7.3, ha='right', fontweight='bold', color=ACCENT)
    sub_raw.set_title("Ritmo médio da liga por temporada (bruto)", fontsize=8.6, fontweight='bold')
    sub_raw.set_ylabel("Posses/jogo", fontsize=7.5)
    sub_raw.tick_params(labelsize=6.5)

    sub_z.axhline(0, color='black', linewidth=0.8, zorder=1)
    sub_z.scatter([s1, s2], [z1, z2], color=[ACCENT, MISSING_COLOR], s=70, zorder=3, edgecolor='black', linewidth=0.8)
    sub_z.annotate(f"z = {z1:+.2f}\n(acima da média\nda SUA temporada)", (s1, z1), textcoords="offset points",
                    xytext=(10, 0), fontsize=6.8, ha='left', va='center', color=ACCENT, fontweight='bold')
    sub_z.annotate(f"z = {z2:+.2f}\n(abaixo da média\nda SUA temporada)", (s2, z2), textcoords="offset points",
                    xytext=(-10, 0), fontsize=6.8, ha='right', va='center', color=MISSING_COLOR, fontweight='bold')
    sub_z.set_title("Mesmo valor, padronizado por temporada", fontsize=8.6, fontweight='bold')
    sub_z.set_ylabel("z-score", fontsize=7.5)
    sub_z.set_xlim(sub_raw.get_xlim())
    sub_z.set_ylim(-2.6, 2.6)
    sub_z.tick_params(labelsize=6.5)

    # Panel 4: cumulative explained variance
    ax = axes[3]
    xs = np.arange(1, len(cum_var) + 1)
    ax.plot(xs, cum_var, color=ACCENT, linewidth=2, zorder=2)
    ax.axhline(0.95, color=INK_SECONDARY, linestyle='--', linewidth=1.1, zorder=1)
    ax.axvline(n_components_95, color=MISSING_COLOR, linestyle='--', linewidth=1.3, zorder=1)
    ax.scatter([n_components_95], [cum_var[n_components_95 - 1]], color=MISSING_COLOR, s=60, zorder=4, edgecolor='black')
    ax.annotate(f"PC{n_components_95}\n({cum_var[n_components_95-1]*100:.1f}%)",
                (n_components_95, cum_var[n_components_95 - 1]), textcoords="offset points",
                xytext=(10, -22), fontsize=8.5, fontweight='bold', color=MISSING_COLOR)
    ax.set_xlim(0, len(cum_var) + 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Componentes principais", fontsize=9)
    ax.set_ylabel("Variância acumulada explicada", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 5: PC1 x PC2 scatter colored by real archetype
    ax = axes[4]
    ax.scatter(features_pca[:, 0], features_pca[:, 1], c=point_colors, s=22, alpha=0.75,
               edgecolor='white', linewidth=0.3, zorder=3)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.6, alpha=0.4, zorder=1)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.6, alpha=0.4, zorder=1)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.tick_params(labelsize=7.5)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=ARCHETYPE_INFO[key]['color'],
                           label=ARCHETYPE_INFO[key]['full'], markersize=7)
               for key in ['rebuilding', 'defensive', 'offensive', 'elite']]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.5, frameon=False)

    # Titles + captions + arrows (figure-fraction coordinates, computed after layout)
    fig.subplots_adjust(top=0.72, bottom=0.28, left=0.03, right=0.98, wspace=0.35)
    for ax, (title, caption) in zip(axes, titles_captions):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.86, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color=INK_PRIMARY)
        fig.text(cx, 0.10, caption, ha='center', va='top', fontsize=9, color=INK_SECONDARY, linespacing=1.4)

    for i in range(len(axes) - 1):
        pos_a = axes[i].get_position()
        pos_b = axes[i + 1].get_position()
        y = (pos_a.y0 + pos_a.y1) / 2 + 0.03
        arrow = FancyArrowPatch((pos_a.x1 + 0.003, y), (pos_b.x0 - 0.003, y),
                                 transform=fig.transFigure, arrowstyle='-|>', mutation_scale=22,
                                 color=INK_SECONDARY, linewidth=2, zorder=5)
        fig.add_artist(arrow)

    fig.text(0.5, 0.965, "Pipeline de Pré-Processamento e Clusterização",
              ha='center', fontsize=18, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.925, "Como 390 times-temporada com 59 variáveis viram 4 arquétipos competitivos",
              ha='center', fontsize=11.5, color=INK_SECONDARY, fontstyle='italic')
    fig.text(0.5, 0.025,
              "Painéis 3 e 4 usam números reais deste projeto (ritmo médio real por temporada; curva de variância "
              "explicada real). Painel 5 projeta o espaço real de 31 componentes em PC1×PC2 apenas para visualização.",
              ha='center', fontsize=8, color=INK_SECONDARY, fontstyle='italic')

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")
    print(f"n_components_95 = {n_components_95}, cum_var = {cum_var[n_components_95-1]:.4f}")
    print(f"Panel 3 example: {example_val} poss. -> {s1} z={z1:+.3f} | {s2} z={z2:+.3f}")


if __name__ == "__main__":
    build()
