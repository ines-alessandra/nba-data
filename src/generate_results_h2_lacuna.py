#!/usr/bin/env python3
"""
Figura 4.2 do Capítulo 4 (Resultados e Discussão) do TCC.

Dois painéis lado a lado para a Hipótese 2 (ajuste funcional do jogador
adquirido): (a) comparação categórica Preenche Lacuna x Redundante (Delta
taxa de WPA da equipe), a forma original da hipótese, não significativa na
amostra agrupada; (b) a mesma relação tratada de forma contínua (Delta P x
Delta WPA), por era de fonte de dado -- significativa apenas na janela
2019-2021, a única com WPA e posição nativamente da ESPN.

Dados: data/wpa_trade_impact_analysis_extended.csv.

Paleta de cor idêntica à já usada em src/generate_wpa_hypothesis_verdicts.py
(os slides apresentados ao orientador): aqua = Preenche Lacuna, vermelho =
Redundante, mesmo par usado para "significativo" x "não significativo" nos
demais vereditos. Fonte Nimbus Sans, igual ao restante do trabalho.

Output: viz/resultados_h2_lacuna_redundante.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis_extended.csv"
OUT_PATH = PROJECT_DIR / "viz" / "resultados_h2_lacuna_redundante.png"

# Mesma paleta de generate_wpa_hypothesis_verdicts.py.
AQUA = "#1baf7a"
RED = "#e34948"
TINT_AQUA = "#dcf3ea"
TINT_RED = "#fbdedd"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_ZONE_FILL = "#e9e8e3"
COLOR_ZONE_EDGE = "#c3c2b7"
SURFACE = "#fcfcfb"
SANS = "Nimbus Sans"


def fmt(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


def main() -> None:
    plt.rcParams["font.family"] = SANS
    df = pd.read_csv(DATA_PATH)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 6.6), dpi=300,
                                    gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.patch.set_facecolor(SURFACE)

    # -------- Painel A: boxplot categórico --------
    g_lac = df.loc[df.trade_classification == "Preenche Lacuna", "delta_WPA_time"].dropna()
    g_red = df.loc[df.trade_classification == "Redundante", "delta_WPA_time"].dropna()
    data = [g_lac.values, g_red.values]

    axA.set_facecolor(SURFACE)
    bp = axA.boxplot(data, positions=[0, 1], widths=0.42, patch_artist=True,
                      showfliers=False, zorder=3,
                      medianprops=dict(color=INK_PRIMARY, linewidth=2.2),
                      whiskerprops=dict(color=INK_PRIMARY, linewidth=1.2),
                      capprops=dict(color=INK_PRIMARY, linewidth=1.2),
                      boxprops=dict(edgecolor=INK_PRIMARY, linewidth=1.4))
    for patch, fill in zip(bp["boxes"], [TINT_AQUA, TINT_RED]):
        patch.set_facecolor(fill)

    rng = np.random.default_rng(42)
    for i, (arr, color) in enumerate(zip(data, [AQUA, RED])):
        jitter = rng.uniform(-0.13, 0.13, size=len(arr))
        axA.scatter(i + jitter, arr, s=15, color=color, alpha=0.55, edgecolors="none", zorder=2)

    axA.axhline(0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    axA.set_xticks([0, 1])
    axA.set_xticklabels([f"Preenche Lacuna\n(n={len(g_lac)})", f"Redundante\n(n={len(g_red)})"],
                         fontsize=13.0)
    axA.set_ylabel("Δ taxa de WPA da equipe (pós − pré-troca)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axA.set_title("(a) Comparação categórica\n(forma original da hipótese)",
                   fontsize=14.5, color=INK_PRIMARY, fontweight="bold", pad=14)
    axA.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axA.spines[spine].set_color(INK_PRIMARY)
        axA.spines[spine].set_linewidth(1.2)

    y_max = max(g_lac.max(), g_red.max())
    bracket_y = y_max + 0.05
    axA.plot([0, 0, 1, 1], [bracket_y - 0.015, bracket_y, bracket_y, bracket_y - 0.015],
              color=INK_PRIMARY, lw=1.4, zorder=4)
    axA.text(0.5, bracket_y + 0.012, "U = 16149,0\np = 0,1165 (não sig.)",
              ha="center", va="bottom", fontsize=11.5, color=INK_SECONDARY, fontweight="bold")
    axA.set_ylim(top=bracket_y + 0.13)

    # -------- Painel B: correlação contínua por era --------
    eras = [
        ("2012-13 a\n2017-18", 2013, 2018),
        ("2018-19 a\n2020-21", 2019, 2021),
        ("2021-22 a\n2024-25", 2022, 2025),
    ]
    rhos, ps, ns = [], [], []
    for _, lo, hi in eras:
        sub = df[df.season_end_year.between(lo, hi)].dropna(subset=["min_deltaP_acquired", "delta_WPA_time"])
        rho, p = stats.spearmanr(sub["min_deltaP_acquired"], sub["delta_WPA_time"])
        rhos.append(rho); ps.append(p); ns.append(len(sub))

    axB.set_facecolor(SURFACE)
    xpos = [0, 1, 2]
    colors_edge = [AQUA if p < 0.05 else COLOR_ZONE_EDGE for p in ps]
    fills = [TINT_AQUA if p < 0.05 else COLOR_ZONE_FILL for p in ps]
    axB.bar(xpos, rhos, width=0.55, color=fills, edgecolor=colors_edge, linewidth=1.8, zorder=3)
    axB.axhline(0, color=INK_PRIMARY, lw=1.1, zorder=2)

    for i, (rho, p, ni) in enumerate(zip(rhos, ps, ns)):
        sig = p < 0.05
        y_txt = rho + (0.018 if rho >= 0 else -0.018)
        va = "bottom" if rho >= 0 else "top"
        axB.text(i, y_txt, f"ρ = {fmt(rho, 3)}\np = {fmt(p, 4)}{'  ***' if sig else ''}",
                  ha="center", va=va, fontsize=11.3, color=(AQUA if sig else INK_SECONDARY),
                  fontweight="bold" if sig else "normal", linespacing=1.4)

    axB.set_xticks(xpos)
    axB.set_xticklabels([f"{lbl}\n(n={ni})" for (lbl, *_), ni in zip(eras, ns)], fontsize=12.0)
    axB.set_ylabel("Spearman ρ (ΔP pré-troca x Δ taxa de WPA)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axB.set_title("(b) Versão contínua, por era de fonte de dado\n(WPA e posição do jogador)",
                   fontsize=14.5, color=INK_PRIMARY, fontweight="bold", pad=14)
    axB.set_ylim(-0.40, 0.15)
    axB.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axB.spines[spine].set_color(INK_PRIMARY)
        axB.spines[spine].set_linewidth(1.2)

    plt.tight_layout(pad=0.6)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
