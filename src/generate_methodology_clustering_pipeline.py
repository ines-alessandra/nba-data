#!/usr/bin/env python3
"""
Figura 3.5 do Capítulo 3 (Metodologia) do TCC.

Pipeline de clusterização usado para derivar os arquétipos competitivos
(K-Means, k=4) que funcionam como variável moderadora da Hipótese 1 --
imputação, padronização por temporada, PCA e K-Means, com a validação
cruzada por Agglomerative Clustering (Ward) ao final. Implementado em
src/team_clustering_common.py e src/fit_clusters.py.

Segue o mesmo sistema de tokens monocromático e a fonte Nimbus Sans das
demais figuras do trabalho.

Segunda versão: aumenta o tamanho de fonte e das caixas (o orientando
relatou dificuldade para ler as figuras da metodologia).

Output: viz/metodologia_pipeline_clusterizacao.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "metodologia_pipeline_clusterizacao.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"

TOP_Y = 5.70
BW, BH = 3.20, 2.30
GAP = 0.35

STEPS = [
    dict(title="Dados brutos", sub="~59 variáveis por\ntime-temporada\n(2012-13 a 2024-25)"),
    dict(title="Imputação", sub="valores faltantes\npela mediana"),
    dict(title="Padronização\npor temporada", sub="z-score dentro\nde cada temporada"),
    dict(title="PCA", sub="retém 95%\nda variância"),
    dict(title="K-Means\n(k = 4)", sub="random_state\nfixo (42)"),
]


def draw_box(ax, x, y_center, w, h, title, sub, title_size=16.5, sub_size=14.0,
             fill=INPUT_FILL, edge_lw=1.4):
    y0 = y_center - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=edge_lw, zorder=3))
    ax.plot([x + 0.18, x + w - 0.18], [y_center + h * 0.20, y_center + h * 0.20],
            color=LINE, lw=0.7, alpha=0.35, zorder=4)
    ax.text(x + w / 2, y_center + h * 0.36, title, ha="center", va="center",
             fontsize=title_size, color=INK, fontweight="bold", zorder=5, linespacing=1.2)
    ax.text(x + w / 2, y_center - h * 0.22, sub, ha="center", va="center",
             fontsize=sub_size, color=INK_MUTED, zorder=5, linespacing=1.5)


def arrow(ax, p0, p1, lw=1.5, mutation_scale=15):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(13.4, 6.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.2, 19.0)
    ax.set_ylim(0.7, 7.3)
    ax.axis("off")

    xs = [0.30 + i * (BW + GAP) for i in range(5)]
    for x, step in zip(xs, STEPS):
        draw_box(ax, x, TOP_Y, BW, BH, step["title"], step["sub"])
    for i in range(4):
        arrow(ax, (xs[i] + BW + 0.05, TOP_Y), (xs[i + 1] - 0.05, TOP_Y))

    # Caixa de validação, alinhada com o K-Means, ligada por uma seta lateral
    # (é um cross-check do resultado do K-Means, não mais um passo em série).
    val_w, val_h = 4.20, 2.30
    val_x = xs[4] + BW / 2 - val_w / 2
    val_y = TOP_Y - BH - 1.30
    draw_box(ax, val_x, val_y, val_w, val_h, "Validação",
             "Agglomerative (Ward)\n+ Adjusted Rand Index",
             title_size=16.5, sub_size=14.0, fill=OUTPUT_FILL, edge_lw=1.9)
    arrow(ax, (xs[4] + BW / 2, TOP_Y - BH / 2 - 0.05),
          (val_x + val_w / 2, val_y + val_h / 2 + 0.05))

    # Monografia, não slide: a explicação de por que a padronização é feita
    # por temporada já está no texto do capítulo -- não repetida dentro da
    # figura. Fica só o pipeline em si.

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
