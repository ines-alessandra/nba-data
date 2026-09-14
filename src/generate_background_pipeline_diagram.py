#!/usr/bin/env python3
"""
Figura 2.2 do Capítulo 2 (Fundamentação Teórica) do TCC.

Décima terceira versão: aumenta bastante o tamanho de fonte de todos os
elementos (títulos de camada, legendas, nota de infraestrutura, fontes de
dados). A versão anterior usava um canvas grande (9.2 pol.) incluído em
.85\\textwidth, o que encolhia o texto a um tamanho ilegível na página
impressa. Para caber com folga nas caixas no novo tamanho de fonte, as
legendas de cada camada e a nota de infraestrutura, antes em uma única
linha longa, foram quebradas em duas linhas.

Estrutura (mantida da versão anterior): três camadas empilhadas (Bronze
na base, Ouro no topo, "refinamento crescente" indicado pela seta
lateral), cada uma com uma miniatura de tabela (grade de linhas x
colunas, número de linhas caindo de uma camada para a outra).

Output: viz/fundamentacao_arquitetura_dados.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "fundamentacao_arquitetura_dados.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"

X_MAX, Y_MAX = 12.2, 10.6

LAYERS = [
    dict(title="Ouro", sub="métricas agregadas por\ntime-temporada", rows=1),
    dict(title="Prata", sub="limpeza e variáveis\nderivadas", rows=3),
    dict(title="Bronze", sub="dado bruto, sem\ntransformação", rows=6),
]  # topo -> base

BOX_X, BOX_W, BOX_H = 3.55, 4.3, 2.6
GAP = 0.35
TOP_Y = 10.05

TABLE_COLS = 3
TABLE_CELL_W = 0.34
TABLE_CELL_H = 0.175

SOURCES_TEXT = "API oficial da NBA (nba_api) · ESPN\n· inpredictable.com"


def draw_box_frame(ax, x, y0, w, h, fill, edge_lw):
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=edge_lw, zorder=3))


def draw_mini_table(ax, cx, cy, rows, cols=TABLE_COLS, cell_w=TABLE_CELL_W,
                     cell_h=TABLE_CELL_H):
    w, h = cols * cell_w, rows * cell_h
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=SURFACE, edgecolor=LINE,
                            linewidth=1.0, zorder=5))
    for r in range(1, rows):
        yy = y0 + r * cell_h
        ax.plot([x0, x0 + w], [yy, yy], color=LINE, lw=0.6, zorder=5)
    for c in range(1, cols):
        xx = x0 + c * cell_w
        ax.plot([xx, xx], [y0, y0 + h], color=LINE, lw=0.6, zorder=5)


def arrow(ax, p0, p1, lw=1.3, mutation_scale=12):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle="arc3,rad=0", zorder=2,
        shrinkA=0, shrinkB=0))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(7.6, 6.9), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.6, Y_MAX)
    ax.axis("off")

    cx = BOX_X + BOX_W / 2
    layer_ys = [TOP_Y - BOX_H, TOP_Y - 2 * BOX_H - GAP, TOP_Y - 3 * BOX_H - 2 * GAP]

    for layer, y0 in zip(LAYERS, layer_ys):
        draw_box_frame(ax, BOX_X, y0, BOX_W, BOX_H, INPUT_FILL, 1.3)
        top = y0 + BOX_H
        ax.text(cx, top - 0.40, layer["title"], ha="center", va="center",
                 fontsize=16.5, color=INK, fontweight="bold", zorder=5)
        ax.plot([BOX_X + 0.22, BOX_X + BOX_W - 0.22], [top - 0.70, top - 0.70],
                 color=LINE, lw=0.7, alpha=0.35, zorder=4)
        ax.text(cx, top - 1.02, layer["sub"], ha="center", va="center",
                 fontsize=11.5, color=INK_MUTED, style="italic", zorder=5,
                 linespacing=1.4)
        draw_mini_table(ax, cx, y0 + 0.62, layer["rows"])

    # Setas de ligação apontando de baixo para cima (Bronze -> Prata -> Ouro),
    # acompanhando o sentido do dado bruto para o refinado -- mesma direção
    # da seta lateral "refinamento crescente".
    for y_top, y_bot in zip(layer_ys[:-1], layer_ys[1:]):
        arrow(ax, (cx, y_bot + BOX_H + 0.03), (cx, y_top - 0.03))

    # Seta lateral -- refinamento crescente -- do lado esquerdo das caixas,
    # para não cruzar com a seta horizontal que sai para a caixa de destino.
    side_x = BOX_X - 0.90
    top_edge = layer_ys[0] + BOX_H
    bottom_edge = layer_ys[-1]
    arrow(ax, (side_x, bottom_edge + 0.12), (side_x, top_edge - 0.12), lw=1.2,
          mutation_scale=10.5)
    ax.text(side_x - 0.30, (top_edge + bottom_edge) / 2, "refinamento\ncrescente",
             ha="right", va="center", fontsize=10.5, color=INK_MUTED,
             style="italic", rotation=90, linespacing=1.5)

    # Fontes alimentando a base (Bronze)
    arrow(ax, (cx, layer_ys[-1] - 0.95), (cx, layer_ys[-1] - 0.08), lw=1.2,
          mutation_scale=10.5)
    ax.text(cx, layer_ys[-1] - 1.32, SOURCES_TEXT, ha="center", va="center",
             fontsize=11.0, color=INK, zorder=5, linespacing=1.4)

    # Caixa de destino, à direita
    out_w, out_h = 3.1, BOX_H
    out_x = BOX_X + BOX_W + 1.15
    out_y0 = layer_ys[0] + BOX_H / 2 - out_h / 2
    draw_box_frame(ax, out_x, out_y0, out_w, out_h, OUTPUT_FILL, 1.9)
    out_cx = out_x + out_w / 2
    ax.text(out_cx, out_y0 + out_h / 2, "Testes estatísticos\ne agrupamento",
             ha="center", va="center", fontsize=13.0, color=INK,
             fontweight="bold", zorder=5, linespacing=1.35)

    arrow(ax, (BOX_X + BOX_W + 0.04, layer_ys[0] + BOX_H / 2),
          (out_x - 0.05, layer_ys[0] + BOX_H / 2))

    # Nota de infraestrutura, com folga maior em relação às caixas acima
    ax.text(X_MAX / 2, layer_ys[-1] - 2.55,
             "PostgreSQL local — schemas bronze / silver / gold,\n"
             "populados por scripts Python e orquestrados via Makefile",
             ha="center", va="center", fontsize=10.5, color=INK_MUTED,
             style="italic", linespacing=1.5)

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
