#!/usr/bin/env python3
"""
Illustrated flowchart of the WPA -> RCP -> muP -> deltaP -> classification ->
delta_WPA_time formula chain, threaded through ONE real trade (James Harden ->
Brooklyn Nets, 14/01/2021) -- same case already used in exemplo_passo_a_passo.py,
reusing its gather_numbers() so both figures are guaranteed to agree.

This REPLACES viz/fluxograma_metodologia.png, which (a) only had 9 text-only
boxes with no real numbers -- not clearly showing HOW the formulas are applied
-- and (b) was actively wrong: its last box claimed the pipeline runs
Mann-Whitney/Kruskal-Wallis significance tests, which it deliberately does NOT
do (see generate_wpa_visualizations.py docstring -- only 3 real seasons of
data, not enough power for inferential statistics). This figure shows the
real, current, descriptive-only pipeline instead, in the same illustrated,
"data actually transforming" style as fluxograma_clusterizacao.png.
"""

import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import get_engine, load_espn_player_box  # noqa: E402
from build_wpa_trade_analysis import asof_league_snapshot  # noqa: E402
from generate_wpa_worked_example import gather_numbers, TEAM_ID, SEASON, TRADE_DATE, SEASON_START  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PNG = PROJECT_DIR / "viz" / "fluxograma_formulas_wpa.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
GOLD = "#eda100"


def fmt(x, d=3):
    return f"{x:+.{d}f}".replace(".", ",")


def build():
    n = gather_numbers()

    engine = get_engine()
    pb = load_espn_player_box(engine)
    snap = asof_league_snapshot(pb, SEASON, TRADE_DATE, SEASON_START)
    guards_snap = snap[(snap["posicao"] == "Guards") & snap["stable"]]

    fig, axes = plt.subplots(1, 6, figsize=(29, 7.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.70, bottom=0.30, left=0.02, right=0.99, wspace=0.35)

    titles = [
        "1. WPA_acc por Jogador",
        "2. Agregar por Posição",
        "3. RCP = Razão",
        "4. Comparar com a Liga",
        "5. Classificar",
        "6. Medir o Impacto",
    ]
    formulas = [
        "WPA_acc = Σ tWPA (ESPN)",
        "WPA_pos = Σ WPA_acc(grupo)",
        "RCP_pos = WPA_pos / WPA_acc(time)",
        "δP = RCP_bar_pos − μP",
        "δP < 0  →  Lacuna",
        "ΔWPA_time = taxa pós − taxa pré",
    ]

    # ------------------------------------------------------------------
    # Panel 1: player-level WPA_acc bars (real roster, pre-trade)
    ax = axes[0]
    top5 = n["guards_roster"][:5]
    names = [r["name"] for r in top5][::-1]
    vals = [r["WPA_acc"] for r in top5][::-1]
    colors = [AQUA if v >= 0 else RED for v in vals]
    ax.barh(names, vals, color=colors, edgecolor='black', linewidth=0.7, height=0.55, zorder=3)
    ax.axvline(0, color=INK_PRIMARY, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, fmt(v), va='center',
                ha='left' if v >= 0 else 'right', fontsize=8.5, fontweight='bold')
    ax.set_xlabel("WPA_acc", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.set_title("", fontsize=1)

    # ------------------------------------------------------------------
    # Panel 2: aggregation -- Guards sum vs. whole-team sum
    ax = axes[1]
    labels2 = ["WPA_pos\n(Guards)", "WPA_acc\n(time inteiro)"]
    vals2 = [n["wpa_pos_guards"], n["wpa_time_team"]]
    colors2 = [AQUA if v >= 0 else RED for v in vals2]
    bars = ax.bar(labels2, vals2, color=colors2, edgecolor='black', linewidth=0.8, width=0.5, zorder=3)
    ax.axhline(0, color=INK_PRIMARY, linewidth=1)
    for bar, v in zip(bars, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.06 if v >= 0 else -0.06), fmt(v),
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.set_ylabel("Soma de WPA_acc", fontsize=9.5)

    # ------------------------------------------------------------------
    # Panel 3: the division, shown as a fraction
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.78, fmt(n["wpa_pos_guards"]), ha='center', va='center', fontsize=17, fontweight='bold', color=AQUA)
    ax.plot([0.15, 0.85], [0.62, 0.62], color=INK_PRIMARY, linewidth=2)
    ax.text(0.5, 0.46, fmt(n["wpa_time_team"]), ha='center', va='center', fontsize=17, fontweight='bold', color=RED)
    ax.text(0.5, 0.22, f"= {fmt(n['rcp_pos'])}", ha='center', va='center', fontsize=19, fontweight='bold', color=INK_PRIMARY)
    ax.text(0.5, 0.02, "(negativo aqui = time inteiro estava\nnegativo, não os guards)",
            ha='center', va='center', fontsize=8, color=INK_SECONDARY, fontstyle='italic', linespacing=1.3)

    # ------------------------------------------------------------------
    # Panel 4: league distribution of RCP_bar_pos(Guards) with BKN + muP marked
    ax = axes[3]
    y_jitter = np.random.default_rng(3).uniform(-0.15, 0.15, size=len(guards_snap))
    ax.scatter(guards_snap["RCP_pos_bar"], y_jitter, color=INK_MUTED, alpha=0.5, s=28, zorder=2, label="30 times (14/01/2021)")
    ax.axvline(n["muP"], color=INK_PRIMARY, linestyle='--', linewidth=1.4, zorder=3)
    ax.scatter([n["rcp_pos_bar"]], [0], color=RED, s=140, zorder=5, edgecolor='black', linewidth=1.2, marker='D')
    ax.annotate(f"μP = {fmt(n['muP'], 4)}", (n["muP"], 0.22), fontsize=8.7, ha='center', color=INK_PRIMARY, fontweight='bold')
    ax.annotate(f"BKN = {fmt(n['rcp_pos_bar'], 4)}", (n["rcp_pos_bar"], -0.28), fontsize=8.7, ha='center', color=RED, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlabel("RCP_bar_pos (Guards)", fontsize=9.5)
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Panel 5: classification decision
    ax = axes[4]
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch((0.06, 0.62), 0.88, 0.28, boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.8, edgecolor=RED, facecolor=RED, alpha=0.1, zorder=1)
    ax.add_patch(box)
    ax.text(0.5, 0.83, f"δP = {fmt(n['deltaP'], 4)}", ha='center', fontsize=13, fontweight='bold', color=RED, zorder=2)
    ax.text(0.5, 0.68, "δP < 0  →  LACUNA estrutural\nem Guards, ANTES da troca",
            ha='center', va='center', fontsize=9, color=INK_PRIMARY, zorder=2, linespacing=1.4)
    ax.annotate('', xy=(0.5, 0.52), xytext=(0.5, 0.60), arrowprops=dict(arrowstyle='-|>', color=INK_SECONDARY, lw=2))
    box2 = FancyBboxPatch((0.06, 0.12), 0.88, 0.38, boxstyle="round,pad=0.02,rounding_size=0.04",
                           linewidth=1.8, edgecolor=AQUA, facecolor=AQUA, alpha=0.12, zorder=1)
    ax.add_patch(box2)
    ax.text(0.5, 0.42, "James Harden → Guards", ha='center', fontsize=10.5, fontweight='bold', color=INK_PRIMARY, zorder=2)
    ax.text(0.5, 0.25, "Guards tinha δP < 0\nneste instante", ha='center', fontsize=9, color=INK_SECONDARY, zorder=2, linespacing=1.4)
    ax.text(0.5, 0.155, "\"Preenche Lacuna\"", ha='center', fontsize=12.5, fontweight='bold', color=AQUA, zorder=2)

    # ------------------------------------------------------------------
    # Panel 6: impact, pre vs post rate
    ax = axes[5]
    labels6 = ["Taxa Pré\n(13 jogos)", "Taxa Pós\n(59 jogos)"]
    vals6 = [n["rate_pre"], n["rate_post"]]
    colors6 = [RED, AQUA]
    bars = ax.bar(labels6, vals6, color=colors6, edgecolor='black', linewidth=0.8, width=0.5, zorder=3)
    ax.axhline(0, color=INK_PRIMARY, linewidth=1)
    for bar, v in zip(bars, vals6):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.006 if v >= 0 else -0.006), fmt(v),
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.set_ylabel("WPA_acc / jogos", fontsize=9.5)
    ax.text(0.5, -0.32, f"ΔWPA_time = {fmt(n['delta_wpa_time'])}", ha='center', transform=ax.transAxes,
            fontsize=11.5, fontweight='bold', color=INK_PRIMARY)

    # ------------------------------------------------------------------
    # Titles, formulas, arrows (figure-fraction, computed after layout)
    for ax, title, formula in zip(axes, titles, formulas):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.86, title, ha='center', va='center', fontsize=13, fontweight='bold', color=INK_PRIMARY)
        fig.text(cx, 0.80, formula, ha='center', va='center', fontsize=9.5, color=BLUE, fontfamily='monospace')

    for i in range(len(axes) - 1):
        pos_a = axes[i].get_position()
        pos_b = axes[i + 1].get_position()
        y = (pos_a.y0 + pos_a.y1) / 2 + 0.02
        arrow = FancyArrowPatch((pos_a.x1 + 0.004, y), (pos_b.x0 - 0.004, y),
                                 transform=fig.transFigure, arrowstyle='-|>', mutation_scale=20,
                                 color=INK_SECONDARY, linewidth=2, zorder=5)
        fig.add_artist(arrow)

    fig.text(0.5, 0.965, "Da Fórmula ao Resultado: Como o WPA Diagnostica uma Troca",
              ha='center', fontsize=18.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.925, "Caso real: James Harden → Brooklyn Nets, 14/01/2021 (temporada 2020-21)",
              ha='center', fontsize=11.5, color=INK_SECONDARY, fontstyle='italic')
    fig.text(0.5, 0.02,
              "Todos os números recalculados ao vivo de data/espn/player_box.parquet e do banco (mesmo caso de "
              "viz/exemplo_passo_a_passo.png, aqui em formato visual). Metodologia descritiva -- sem teste de "
              "significância estatística (amostra de 3 temporadas reais é pequena demais para inferência).",
              ha='center', fontsize=8.7, color=INK_SECONDARY, fontstyle='italic')

    fig.savefig(OUT_PNG, dpi=300, facecolor=SURFACE, bbox_inches='tight')
    plt.close()
    print(f"[OK] {OUT_PNG}")


if __name__ == "__main__":
    build()
