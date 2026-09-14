#!/usr/bin/env python3
"""
Figura 3.4 do Capítulo 3 (Metodologia) do TCC.

Mostra, para as 13 temporadas usadas nas Hipóteses 2 a 4 (2013-2025), qual
fonte supre o WPA por jogo e a posição de cada jogador em cada uma das três
eras de disponibilidade de dado -- transparência necessária porque nem
ESPN nem inpredictable.com cobrem as 13 temporadas por igual (ver
src/wpa_extended_common.py e src/build_wpa_positional_analysis_extended.py).

Segue o mesmo sistema de tokens monocromático e a fonte Nimbus Sans das
demais figuras do trabalho.

Output: viz/metodologia_fontes_wpa_por_era.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "metodologia_fontes_wpa_por_era.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"

ERAS = [
    dict(
        period="2012-13 a 2017-18",
        wpa="WPA: inpredictable.com",
        pos="Posição: rótulo estático\ndo inpredictable.com\n(Guards/Forwards/Centers)",
        note="mais fraca das três: não\ndetecta mudança de posição\nno meio da temporada",
        fill=INPUT_FILL,
    ),
    dict(
        period="2018-19 a 2020-21",
        wpa="WPA: ESPN (nativo)",
        pos="Posição: ESPN (dAvgPos),\npor jogo, ponderada\npor minutos",
        note="fonte original de Sá (2025);\nsinal contínuo e por jogo\nnas duas variáveis",
        fill=OUTPUT_FILL,
    ),
    dict(
        period="2021-22 a 2024-25",
        wpa="WPA: inpredictable.com",
        pos="Posição: ESPN (dAvgPos),\npor jogo, ponderada\npor minutos",
        note="híbrida: ESPN manteve o\ndAvgPos após zerar o WPA\npróprio nessas temporadas",
        fill=INPUT_FILL,
    ),
]

COL_W, COL_H = 4.55, 5.50
GAP = 0.65


def draw_col(ax, x, y0, era):
    ax.add_patch(Rectangle((x, y0), COL_W, COL_H, facecolor=era["fill"],
                            edgecolor=LINE, linewidth=1.4, zorder=3))
    top = y0 + COL_H
    cx = x + COL_W / 2
    ax.text(cx, top - 0.51, era["period"], ha="center", va="center",
             fontsize=19.0, color=INK, fontweight="bold", zorder=5)
    ax.plot([x + 0.24, x + COL_W - 0.24], [top - 0.90, top - 0.90],
            color=LINE, lw=0.7, alpha=0.35, zorder=4)
    ax.text(cx, top - 1.55, era["wpa"], ha="center", va="center",
             fontsize=15.0, color=INK, fontweight="bold", zorder=5, linespacing=1.4)
    ax.text(cx, top - 2.60, era["pos"], ha="center", va="center",
             fontsize=13.5, color=INK, zorder=5, linespacing=1.55)
    ax.plot([x + 0.24, x + COL_W - 0.24], [y0 + 1.63, y0 + 1.63],
            color=LINE, lw=0.6, alpha=0.25, zorder=4)
    ax.text(cx, y0 + 0.94, era["note"], ha="center", va="center",
             fontsize=12.5, color=INK_MUTED, style="italic", zorder=5, linespacing=1.55)


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    total_w = 3 * COL_W + 2 * GAP
    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(0.05, 7.35)
    ax.axis("off")

    y0 = 0.40
    xs = [i * (COL_W + GAP) for i in range(3)]
    for x, era in zip(xs, ERAS):
        draw_col(ax, x, y0, era)

    # Linha do tempo conectando as três eras, acima das caixas.
    line_y = y0 + COL_H + 0.65
    ax.add_patch(FancyArrowPatch(
        (xs[0] + COL_W / 2, line_y), (xs[2] + COL_W / 2, line_y),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=LINE,
        zorder=2, shrinkA=0, shrinkB=0))
    for x in xs:
        ax.plot([x + COL_W / 2, x + COL_W / 2], [line_y - 0.12, line_y + 0.12],
                color=LINE, lw=1.4, zorder=3)
    ax.text((xs[0] + xs[2] + COL_W) / 2, line_y + 0.46,
             "13 temporadas — amostra das Hipóteses 2 a 4 (2012-13 a 2024-25)",
             ha="center", va="center", fontsize=13.0, family=SANS, color=INK_MUTED)

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
