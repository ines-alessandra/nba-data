#!/usr/bin/env python3
"""
Figura 4.4 do Capítulo 4 (Resultados e Discussão) do TCC.

Dois painéis para a Hipótese 4 (estrutura da negociação): (a) dispersão
completa de Delta P mínimo entre os jogadores adquiridos x número de
jogadores adquiridos no evento, com tendência local; (b) a mesma amostra
agrupada em quartis de severidade da lacuna, com mediana e IC95% bootstrap
do tamanho do pacote por quartil.

Dados: data/wpa_trade_impact_analysis_extended.csv (min_deltaP_acquired,
n_players_acquired).

Paleta de cor idêntica à já usada em src/generate_wpa_hypothesis_verdicts.py
(os slides apresentados ao orientador): aqua para os quartis com lacuna real
(Delta P negativo), vermelho para os quartis redundantes (Delta P >= 0) --
o mesmo par de cores já usado para Preenche Lacuna x Redundante na Figura
4.2. Fonte Nimbus Sans, igual ao restante do trabalho.

Output: viz/resultados_h4_severidade_pacote.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis_extended.csv"
OUT_PATH = PROJECT_DIR / "viz" / "resultados_h4_severidade_pacote.png"

# Mesma paleta de generate_wpa_hypothesis_verdicts.py.
AQUA = "#1baf7a"
RED = "#e34948"
TINT_AQUA = "#dcf3ea"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
SANS = "Nimbus Sans"


def fmt(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


def bootstrap_median_ci(x, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    df = pd.read_csv(DATA_PATH).dropna(subset=["min_deltaP_acquired", "n_players_acquired"]).copy()

    rho, p = stats.spearmanr(df["min_deltaP_acquired"], df["n_players_acquired"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 6.6), dpi=300,
                                    gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # -------- Painel A: dispersão + tendência --------
    axA.set_facecolor(SURFACE)
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.10, 0.10, size=len(df))
    point_colors = np.where(df["min_deltaP_acquired"] < 0, AQUA, RED)
    axA.scatter(df["min_deltaP_acquired"], df["n_players_acquired"] + jitter, s=26,
                color=point_colors, alpha=0.40, edgecolors="none", zorder=3)
    lowess = sm.nonparametric.lowess
    z = lowess(df["n_players_acquired"], df["min_deltaP_acquired"], frac=0.65)
    axA.plot(z[:, 0], z[:, 1], color=INK_PRIMARY, linewidth=2.6, zorder=4)
    axA.axvline(0, color=INK_MUTED, lw=1.0, ls="--", zorder=2)

    axA.set_xlabel("ΔP do jogador adquirido (lacuna pré-troca)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axA.set_ylabel("Nº de jogadores adquiridos no evento", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axA.set_title(f"(a) Dispersão completa (n={len(df)})", fontsize=14.5, color=INK_PRIMARY,
                   fontweight="bold", pad=14)
    axA.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axA.spines[spine].set_color(INK_PRIMARY)
        axA.spines[spine].set_linewidth(1.2)

    axA.text(0.97, 0.97, f"Spearman ρ = {fmt(rho, 3)}\np < 0,0001",
              transform=axA.transAxes, ha="right", va="top", fontsize=13.0, color=AQUA,
              fontweight="bold", linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=TINT_AQUA, edgecolor=AQUA, linewidth=1.2))

    # -------- Painel B: quartis --------
    axB.set_facecolor(SURFACE)
    df["quartil"] = pd.qcut(df["min_deltaP_acquired"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qs = []
    for ql in ["Q1", "Q2", "Q3", "Q4"]:
        s = df.loc[df.quartil == ql, "n_players_acquired"]
        lo, hi = bootstrap_median_ci(s.values)
        qs.append({"q": ql, "n": len(s), "median": s.median(), "lo": lo, "hi": hi})

    x = np.arange(4)
    medians = [d["median"] for d in qs]
    bar_colors = [AQUA, AQUA, RED, RED]  # Q1-Q2 = lacuna real; Q3-Q4 = redundante
    axB.bar(x, medians, width=0.58, color=bar_colors, edgecolor=INK_PRIMARY, linewidth=1.3,
            alpha=0.80, zorder=3)
    for i, d in enumerate(qs):
        axB.plot([i, i], [d["lo"], d["hi"]], color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.plot([i - 0.08, i + 0.08], [d["lo"]] * 2, color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.plot([i - 0.08, i + 0.08], [d["hi"]] * 2, color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.text(i, d["hi"] + 0.05, fmt(d["median"], 1), ha="center", va="bottom",
                  fontsize=12.0, color=INK_PRIMARY, fontweight="bold")

    axB.set_xticks(x)
    axB.set_xticklabels([f"{d['q']}\n(n={d['n']})" for d in qs], fontsize=12.5)
    axB.set_ylabel("Mediana de jogadores no evento (IC95% bootstrap)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axB.set_title("(b) Por quartil de severidade da lacuna\n(Q1 = mais severa; Q4 = mais redundante)",
                   fontsize=14.5, color=INK_PRIMARY, fontweight="bold", pad=14)
    axB.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    y_hi = max(d["hi"] for d in qs)
    axB.set_ylim(0, y_hi * 1.55)
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
