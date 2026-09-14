#!/usr/bin/env python3
"""
Figura 3.3 do Capítulo 3 (Metodologia) do TCC.

Cadeia de cálculo da Contribuição Relativa (ΔP), da métrica individual
(WPA_acc, por jogador) até a classificação binária da troca (Preenche
Lacuna / Redundante), seguindo a metodologia de Sá (2025), Seção III,
implementada em src/wpa_common.py e src/build_wpa_trade_analysis.py.

Segue o mesmo sistema de tokens monocromático e a fonte Nimbus Sans das
demais figuras (1.1, 1.2, 2.1, 2.2, 3.1, 3.2).

Terceira versão: aumenta bastante o tamanho de fonte de todos os elementos
(o orientando relatou dificuldade para ler as figuras da metodologia). As
caixas também crescem, na mesma proporção, para acomodar o texto maior sem
estourar a largura -- só a fonte sozinha estava resultando em texto raspando
as bordas das caixas mais estreitas (WPA_pos).

Output: viz/metodologia_cadeia_wpa.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "metodologia_cadeia_wpa.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"

TOP_Y = 7.95
BW, BH = 3.40, 2.20
GAP = 0.36

STEPS = [
    dict(sym="$WPA_{acc}$", desc="soma do WPA por jogo\nde um jogador,\nno período"),
    dict(sym="$WPA_{pos}$", desc="soma do $WPA_{acc}$ dos\njogadores da mesma\nposição no time"),
    dict(sym="$RCP_{pos}$", desc="$WPA_{pos}\\, /\\, WPA_{time}$\nfração do time vinda\ndaquela posição"),
    dict(sym=r"$\overline{RCP}_{pos}$", desc="$RCP_{pos}\\, /\\, $ nº de jogos\nnormalizado pelo\ntamanho do período"),
    dict(sym="$\\Delta P$", desc="$\\overline{RCP}_{pos} - \\mu_P$\ndesvio em relação\nà média da liga"),
]


def draw_box(ax, x, y_center, w, h, title, sub, title_size=19.5, sub_size=14.5,
             fill=INPUT_FILL, edge_lw=1.4):
    y0 = y_center - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=edge_lw, zorder=3))
    ax.plot([x + 0.18, x + w - 0.18], [y_center + h * 0.20, y_center + h * 0.20],
            color=LINE, lw=0.7, alpha=0.35, zorder=4)
    ax.text(x + w / 2, y_center + h * 0.34, title, ha="center", va="center",
             fontsize=title_size, color=INK, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y_center - h * 0.20, sub, ha="center", va="center",
             fontsize=sub_size, color=INK_MUTED, zorder=5, linespacing=1.5)


def arrow(ax, p0, p1, lw=1.5, mutation_scale=15):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0))


def label_beside_line(p0, p1, t, offset, side=1.0):
    """Ponto a uma fração t ao longo do segmento p0->p1, deslocado por
    `offset` na direção perpendicular à linha -- evita texto sobre a seta."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    px, py = -dy / length, dx / length  # unitário perpendicular
    x = p0[0] + t * dx + side * offset * px
    y = p0[1] + t * dy + side * offset * py
    return x, y


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(15.3, 8.3), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.2, 22.2)
    ax.set_ylim(-0.4, 9.5)
    ax.axis("off")

    xs = [0.30 + i * (BW + GAP) for i in range(5)]
    for x, step in zip(xs, STEPS):
        draw_box(ax, x, TOP_Y, BW, BH, step["sym"], step["desc"])
    for i in range(4):
        x_a = xs[i] + BW + 0.05
        x_b = xs[i + 1] - 0.05
        arrow(ax, (x_a, TOP_Y), (x_b, TOP_Y))

    dp_cx = xs[4] + BW / 2

    # mu_P: computado em paralelo, entre as 30 equipes na mesma data-corte.
    # Fica bem mais à esquerda (sob a 2a caixa da fileira de cima), com uma
    # única seta diagonal longa entrando pela base da caixa de Delta P, à
    # esquerda das duas setas de classificação que também saem dali --
    # assim a ligação fica visualmente clara, em vez de espremida entre
    # RCP_bar_pos e Delta P.
    mu_w, mu_h = BW, 1.90
    mu_x = xs[1] + 0.50
    mu_y = 4.45
    draw_box(ax, mu_x, mu_y, mu_w, mu_h, "$\\mu_P$",
             "média do $\\overline{RCP}_{pos}$\nentre as 30 equipes,\nna mesma data-corte",
             title_size=18.0, sub_size=13.5)
    mu_out = (mu_x + mu_w + 0.05, mu_y + mu_h / 2 * 0.6)
    dp_in = (dp_cx - 1.15, TOP_Y - BH / 2 - 0.05)
    arrow(ax, mu_out, dp_in)
    # (a fórmula já aparece dentro da própria caixa de Delta P -- repeti-la
    # aqui seria redundante, e o rótulo competia por espaço com a seta.)

    # Classificação: ramifica a partir da base da caixa de Delta P.
    class_y = 1.05
    class_w, class_h = 3.90, 1.70
    lacuna_x = dp_cx - class_w - 0.75
    redund_x = dp_cx + 0.75
    draw_box(ax, lacuna_x, class_y, class_w, class_h, "Preenche Lacuna",
             "posição estruturalmente\ndeficiente antes da troca",
             title_size=16.5, sub_size=13.8, fill=OUTPUT_FILL, edge_lw=1.9)
    draw_box(ax, redund_x, class_y, class_w, class_h, "Redundante",
             "posição já contribuía\nna média da liga ou acima",
             title_size=16.5, sub_size=13.8, fill=OUTPUT_FILL, edge_lw=1.9)

    dp_bottom_l = (dp_cx - 0.45, TOP_Y - BH / 2 - 0.05)
    dp_bottom_r = (dp_cx + 0.45, TOP_Y - BH / 2 - 0.05)
    lacuna_top = (lacuna_x + class_w / 2, class_y + class_h / 2 + 0.05)
    redund_top = (redund_x + class_w / 2, class_y + class_h / 2 + 0.05)
    arrow(ax, dp_bottom_l, lacuna_top)
    arrow(ax, dp_bottom_r, redund_top)
    # Rótulos deslocados perpendicularmente à seta (não apenas em x), para
    # nunca ficar sobre o traço -- e, do lado esquerdo, abaixo o bastante
    # para também não tocar a caixa de mu_P.
    lx, ly = label_beside_line(dp_bottom_l, lacuna_top, t=0.80, offset=0.68, side=1.0)
    ax.text(lx, ly, "$\\Delta P < 0$", ha="center", va="center", fontsize=15.0,
             family=SANS, color=INK_MUTED, style="italic")
    rx, ry = label_beside_line(dp_bottom_r, redund_top, t=0.80, offset=0.68, side=-1.0)
    ax.text(rx, ry, "$\\Delta P \\geq 0$", ha="center", va="center", fontsize=15.0,
             family=SANS, color=INK_MUTED, style="italic")

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
