#!/usr/bin/env python3
"""
Figura 1.2 do Capítulo 1 (Introdução) do TCC.

Diagrama conceitual do modelo multidimensional descrito na seção 1.1
(Objetivos): as quatro hipóteses centrais do trabalho -- desigualdade
salarial (H1), ajuste funcional do jogador (H2), experiência dos jogadores
adquiridos (H3) e estrutura da negociação (H4) -- convergindo para o
impacto da troca na performance.

Sexta versão: volta à quarta versão (estilo monocromático de caixas e
setas em linha fina, sem ícones nem cor categórica) -- o orientando
preferiu esse registro ao redesenho com ícones em azulejo tentado para
alinhar com a Figura 2.1. Único ajuste real: a fonte muda de DejaVu Sans
para Nimbus Sans, para acompanhar a família tipográfica adotada nas
demais figuras do trabalho.

Output: viz/introducao_modelo_multidimensional.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "introducao_modelo_multidimensional.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"

BOX_W, BOX_H = 4.05, 1.62

DIMENSIONS = [
    dict(y=8.85, title="Desigualdade salarial",
         sub="Índice de Gini ponderado por minutos jogados"),
    dict(y=6.35, title="Ajuste funcional do jogador",
         sub="Contribuição Relativa, derivada do WPA"),
    dict(y=3.85, title="Experiência dos jogadores\nadquiridos",
         sub="Idade média no momento da troca", title_size=9.4),
    dict(y=1.35, title="Estrutura da negociação",
         sub="Severidade da lacuna e tamanho do pacote"),
]

BOX_X = 0.35
OUT_W, OUT_H = 3.35, 4.6
OUT_X = 6.55


def draw_box(ax, x, y_center, w, h, fill, edge_lw, title, sub,
             title_size=10.0, sub_size=8.2, title_dy=0.30, sub_dy=-0.30,
             divider_dy=-0.02):
    y0 = y_center - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=edge_lw, zorder=3))
    ax.plot([x + 0.22, x + w - 0.22], [y_center + divider_dy, y_center + divider_dy],
            color=LINE, lw=0.6, alpha=0.35, zorder=4)
    ax.text(x + w / 2, y_center + title_dy, title, ha="center", va="center",
             fontsize=title_size, color=INK, fontweight="bold", zorder=5,
             linespacing=1.25)
    ax.text(x + w / 2, y_center + sub_dy, sub, ha="center", va="center",
             fontsize=sub_size, color=INK_MUTED, style="italic", zorder=5,
             linespacing=1.35, wrap=True)


def main() -> None:
    plt.rcParams["font.family"] = "Nimbus Sans"
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    for d in DIMENSIONS:
        two_line_title = "\n" in d["title"]
        draw_box(ax, BOX_X, d["y"], BOX_W, BOX_H, INPUT_FILL, 1.1,
                 d["title"], d["sub"],
                 title_size=d.get("title_size", 10.0),
                 title_dy=0.42 if two_line_title else 0.30,
                 sub_dy=-0.44 if two_line_title else -0.30,
                 divider_dy=0.10 if two_line_title else -0.02)

    draw_box(ax, OUT_X, 5.1, OUT_W, OUT_H, OUTPUT_FILL, 1.7,
              "Impacto da troca\nna performance",
              "Modelo multidimensional\nem nível de jogo",
              title_size=11.2, sub_size=8.6, title_dy=0.65, sub_dy=-0.65,
              divider_dy=0.05)

    start_x = BOX_X + BOX_W
    targets_y = [7.85, 6.15, 4.45, 2.75]
    for d, ty in zip(DIMENSIONS, targets_y):
        ax.add_patch(FancyArrowPatch(
            (start_x + 0.06, d["y"]), (OUT_X - 0.06, ty),
            arrowstyle="-|>", mutation_scale=10.5, linewidth=1.1,
            color=LINE, connectionstyle="arc3,rad=0", zorder=2,
            shrinkA=0, shrinkB=0))

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
