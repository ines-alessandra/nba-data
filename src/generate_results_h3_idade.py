#!/usr/bin/env python3
"""
Figura 4.3 do Capítulo 4 (Resultados e Discussão) do TCC.

Dois painéis para a Hipótese 3 (experiência dos jogadores adquiridos):
(a) dispersão completa de idade média dos jogadores adquiridos x Delta taxa
de WPA da equipe, com uma linha de tendência local; (b) a mesma amostra
agrupada em quartis de idade, com mediana e IC95% bootstrap por quartil.

Idade recalculada ao vivo a partir de bronze.raw_trades.player_age (não
está persistida em nenhum CSV), seguindo a mesma lógica de
compute_idade_media_adquiridos() em generate_wpa_hypothesis_verdicts.py.

Paleta de cor idêntica à já usada em src/generate_wpa_hypothesis_verdicts.py
(os slides apresentados ao orientador): azul para a tendência, aqua para
marcar o resultado significativo. Fonte Nimbus Sans, igual ao restante do
trabalho.

Output: viz/resultados_h3_idade.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sqlalchemy import text

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from wpa_common import get_engine  # noqa: E402

DATA_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis_extended.csv"
OUT_PATH = PROJECT_DIR / "viz" / "resultados_h3_idade.png"

# Mesma paleta de generate_wpa_hypothesis_verdicts.py.
BLUE = "#2a78d6"
AQUA = "#1baf7a"
TINT_AQUA = "#dcf3ea"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
SANS = "Nimbus Sans"


def fmt(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


def compute_idade_media(impact: pd.DataFrame) -> pd.Series:
    engine = get_engine()
    with engine.connect() as conn:
        incoming = pd.read_sql(text("""
            SELECT season_end_year, trade_id, team_id, player_age
            FROM bronze.raw_trades
            WHERE direction = 'Incoming' AND asset_type = 'Player' AND player_age IS NOT NULL
        """), con=conn)

    def avg_age(row):
        s = int(row["season_end_year"])
        tids = set(int(x) for x in str(row["trade_ids"]).split(","))
        sub = incoming[(incoming.season_end_year == s) & (incoming.trade_id.isin(tids)) &
                        (incoming.team_id == row["team_id"])]
        return sub["player_age"].mean() if len(sub) else np.nan

    return impact.apply(avg_age, axis=1)


def bootstrap_median_ci(x, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    df = pd.read_csv(DATA_PATH)
    df["idade_media"] = compute_idade_media(df)
    df = df.dropna(subset=["idade_media", "delta_WPA_time"]).copy()

    rho, p = stats.spearmanr(df["idade_media"], df["delta_WPA_time"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 6.6), dpi=300,
                                    gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # -------- Painel A: dispersão + tendência --------
    axA.set_facecolor(SURFACE)
    axA.scatter(df["idade_media"], df["delta_WPA_time"], s=26, color=INK_SECONDARY, alpha=0.30,
                edgecolors="none", zorder=3)
    lowess = sm.nonparametric.lowess
    z = lowess(df["delta_WPA_time"], df["idade_media"], frac=0.65)
    axA.plot(z[:, 0], z[:, 1], color=BLUE, linewidth=3.0, zorder=4)
    axA.axhline(0, color=INK_MUTED, lw=1.0, ls="--", zorder=2)

    axA.set_xlabel("Idade média dos jogadores adquiridos (anos)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axA.set_ylabel("Δ taxa de WPA da equipe (pós − pré-troca)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axA.set_title(f"(a) Dispersão completa (n={len(df)})", fontsize=14.5, color=INK_PRIMARY,
                   fontweight="bold", pad=14)
    axA.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axA.spines[spine].set_color(INK_PRIMARY)
        axA.spines[spine].set_linewidth(1.2)

    axA.text(0.03, 0.97, f"Spearman ρ = {fmt(rho, 3)}\np = {fmt(p, 4)}",
              transform=axA.transAxes, ha="left", va="top", fontsize=13.0, color=AQUA,
              fontweight="bold", linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=TINT_AQUA, edgecolor=AQUA, linewidth=1.2))

    # -------- Painel B: quartis --------
    axB.set_facecolor(SURFACE)
    df["quartil"] = pd.qcut(df["idade_media"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qs = []
    for ql in ["Q1", "Q2", "Q3", "Q4"]:
        s = df.loc[df.quartil == ql, "delta_WPA_time"]
        lo, hi = bootstrap_median_ci(s.values)
        qs.append({"q": ql, "n": len(s), "median": s.median(), "lo": lo, "hi": hi})

    x = np.arange(4)
    medians = [d["median"] for d in qs]
    bar_colors = [BLUE, BLUE, BLUE, AQUA]  # Q4 destacado -- é onde o salto aparece
    axB.bar(x, medians, width=0.58, color=bar_colors, edgecolor=INK_PRIMARY, linewidth=1.3,
            alpha=0.85, zorder=3)
    for i, d in enumerate(qs):
        axB.plot([i, i], [d["lo"], d["hi"]], color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.plot([i - 0.08, i + 0.08], [d["lo"]] * 2, color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.plot([i - 0.08, i + 0.08], [d["hi"]] * 2, color=INK_PRIMARY, linewidth=1.6, zorder=4)
        axB.text(i, d["hi"] + 0.004, fmt(d["median"], 4), ha="center", va="bottom",
                  fontsize=12.0, color=INK_PRIMARY, fontweight="bold")
    axB.axhline(0, color=INK_MUTED, lw=1.0, zorder=2)

    axB.set_xticks(x)
    axB.set_xticklabels([f"{d['q']}\n(n={d['n']})" for d in qs], fontsize=12.5)
    axB.set_ylabel("Mediana de Δ taxa de WPA (IC95% bootstrap)", fontsize=13.5, color=INK_PRIMARY, labelpad=10)
    axB.set_title("(b) Por quartil de idade média adquirida", fontsize=14.5, color=INK_PRIMARY,
                   fontweight="bold", pad=14)
    axB.tick_params(labelsize=12.0, colors=INK_PRIMARY, length=4.5, width=1.1)
    y_hi = max(d["hi"] for d in qs)
    y_lo = min(d["lo"] for d in qs)
    axB.set_ylim(y_lo - 0.01, y_hi * 1.28)
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
