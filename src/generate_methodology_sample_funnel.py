#!/usr/bin/env python3
"""
Figura 3.1 do Capítulo 3 (Metodologia) do TCC.

Funil de construção das duas amostras usadas nas quatro hipóteses: a
amostra em nível de time-temporada (Hipótese 1, desigualdade salarial) e a
amostra em nível de evento de troca (Hipóteses 2 a 4, WPA/Contribuição
Relativa, experiência e estrutura da negociação). Os números vêm direto de
data/team_season_features.csv, data/team_trade_activity.csv e
data/wpa_trade_impact_analysis_extended.csv -- contagens de amostra, não
resultados de hipótese.

Segue o mesmo sistema de tokens monocromático (INK/MUTED/LINE/INPUT_FILL)
e a fonte Nimbus Sans usadas nas Figuras 1.1, 1.2, 2.1 e 2.2.

Output: viz/metodologia_funil_amostra.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "metodologia_funil_amostra.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
INPUT_FILL = "#f2f4f7"
OUTPUT_FILL = "#ffffff"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"

BOX_W, BOX_H = 5.30, 2.05


def draw_box(ax, x, y_center, title, sub, title_size=18.0, sub_size=13.5):
    y0 = y_center - BOX_H / 2
    ax.add_patch(Rectangle((x, y0), BOX_W, BOX_H, facecolor=INPUT_FILL,
                            edgecolor=LINE, linewidth=1.4, zorder=3))
    ax.plot([x + 0.24, x + BOX_W - 0.24], [y_center + 0.32, y_center + 0.32],
            color=LINE, lw=0.7, alpha=0.35, zorder=4)
    ax.text(x + BOX_W / 2, y_center + 0.68, title, ha="center", va="center",
             fontsize=title_size, color=INK, fontweight="bold", zorder=5,
             linespacing=1.25)
    ax.text(x + BOX_W / 2, y_center - 0.36, sub, ha="center", va="center",
             fontsize=sub_size, color=INK_MUTED, style="italic", zorder=5,
             linespacing=1.45)


def arrow(ax, p0, p1, lw=1.4, mutation_scale=13):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0))


def track(ax, y, label, box1_title, box1_sub, filt_label, box2_title, box2_sub):
    ax.text(0.15, y + 1.72, label, ha="left", va="center", fontsize=16.5,
             color=INK, fontweight="bold")
    x1 = 0.15
    x2 = x1 + BOX_W + 3.75
    draw_box(ax, x1, y, box1_title, box1_sub)
    draw_box(ax, x2, y, box2_title, box2_sub)
    arrow(ax, (x1 + BOX_W + 0.08, y), (x2 - 0.08, y))
    ax.text((x1 + BOX_W + x2) / 2, y + 0.48, filt_label, ha="center", va="center",
             fontsize=12.5, color=INK_MUTED, linespacing=1.45)


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(10.6, 6.1), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.2, 14.8)
    ax.set_ylim(0.3, 7.6)
    ax.axis("off")

    track(
        ax, y=5.55,
        label="Amostra da Hipótese 1 — desigualdade salarial",
        box1_title="390 times-temporada",
        box1_sub="2012-13 a 2024-25\nGini ponderado e Performance\nAjustada pré/pós-deadline válidos",
        filt_label="filtro: troca real registrada\nnos 30 dias antes do deadline",
        box2_title="282 times-temporada",
        box2_sub="amostra final da Hipótese 1",
    )

    track(
        ax, y=1.75,
        label="Amostra das Hipóteses 2 a 4 — WPA / Contribuição Relativa",
        box1_title="370 eventos de troca",
        box1_sub="2012-13 a 2024-25 · uma linha\npor equipe-data de troca",
        filt_label="cada evento pode\nenvolver mais de 1 jogador",
        box2_title="663 jogadores adquiridos",
        box2_sub="unidade classificada em\nPreenche Lacuna / Redundante",
    )

    plt.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
