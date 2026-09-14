#!/usr/bin/env python3
"""
Figura 4.5 do Capítulo 4 (Resultados e Discussão) do TCC.

Distribuição de Delta taxa de WPA da equipe por objetivo sazonal
(contender / intermediário / tanking), a variável de contexto complementar
descrita na Seção 3.7 do Capítulo 3 -- não uma quinta hipótese central.
Anota o p-valor do teste de Wilcoxon de amostra única (H0: mediana=0) por
grupo, evidenciando que a previsão de impacto neutro/negativo para tanking
se inverte.

Dados: data/wpa_trade_impact_analysis_extended.csv (objetivo_sazonal,
delta_WPA_time).

Paleta de cor idêntica à já usada em src/generate_wpa_hypothesis_verdicts.py
(os slides apresentados ao orientador): azul = Contender, laranja =
Intermediário, aqua = Tanking; o destaque vermelho sobre Tanking marca a
inversão da previsão. Fonte Nimbus Sans, igual ao restante do trabalho.

Output: viz/resultados_objetivo_sazonal.png
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
OUT_PATH = PROJECT_DIR / "viz" / "resultados_objetivo_sazonal.png"

# Mesma paleta de generate_wpa_hypothesis_verdicts.py.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
TINT_RED = "#fbdedd"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
SANS = "Nimbus Sans"

OBJ_COLORS = {"Contender": BLUE, "Intermediario": ORANGE, "Tanking": AQUA}


def fmt(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


def main() -> None:
    plt.rcParams["font.family"] = SANS
    df = pd.read_csv(DATA_PATH)

    order = ["Contender", "Intermediario", "Tanking"]
    labels = ["Contender", "Intermediário", "Tanking"]
    predicted = ["previsto: melhora", "sem previsão", "previsto: neutro/negativo"]
    groups = {g: df.loc[df.objetivo_sazonal == g, "delta_WPA_time"].dropna() for g in order}

    fig, ax = plt.subplots(figsize=(10.6, 7.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    data = [groups[g].values for g in order]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.45, patch_artist=True,
                     showfliers=False, zorder=3,
                     medianprops=dict(color=INK_PRIMARY, linewidth=2.2),
                     whiskerprops=dict(color=INK_PRIMARY, linewidth=1.2),
                     capprops=dict(color=INK_PRIMARY, linewidth=1.2),
                     boxprops=dict(edgecolor=INK_PRIMARY, linewidth=1.3))
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(OBJ_COLORS[g])
        patch.set_alpha(0.28)

    rng = np.random.default_rng(3)
    for i, g in enumerate(order):
        arr = groups[g].values
        jitter = rng.uniform(-0.15, 0.15, size=len(arr))
        ax.scatter(i + jitter, arr, s=14, color=OBJ_COLORS[g], alpha=0.55, edgecolors="none", zorder=2)

    ax.axhline(0, color=INK_MUTED, lw=1.1, ls="--", zorder=1)

    y_top = df["delta_WPA_time"].max()
    for i, g in enumerate(order):
        w, pw = stats.wilcoxon(groups[g])
        sig = pw < 0.05
        med = groups[g].median()
        txt_color = RED if (g == "Tanking" and sig) else (OBJ_COLORS[g] if sig else INK_SECONDARY)
        weight = "bold" if sig else "normal"
        ax.text(i, y_top * 1.06, f"mediana = {fmt(med, 4)}", ha="center", va="bottom",
                 fontsize=11.5, color=INK_PRIMARY)
        ax.text(i, y_top * 1.14, f"Wilcoxon p = {fmt(pw, 4)}{'  (sig.)' if sig else ''}",
                 ha="center", va="bottom", fontsize=11.5, color=txt_color, fontweight=weight)

    # Monografia, não slide: o veredito ("inverte a previsão") já está no
    # texto do capítulo -- a figura fica só com os rótulos numéricos por
    # grupo (mediana e p-valor, já em vermelho/negrito quando significativo
    # e no grupo Tanking), que bastam para ler o resultado no gráfico.

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"{lbl}\n(n={len(groups[g])})\n{pred}"
                         for lbl, g, pred in zip(labels, order, predicted)], fontsize=12.5)
    ax.set_ylabel("Δ taxa de WPA da equipe (pós − pré-troca)", fontsize=14.5, color=INK_PRIMARY, labelpad=12)
    ax.set_title("Kruskal-Wallis (3 grupos): H = 1,0698   p = 0,5857 (não significativo)",
                  fontsize=14.0, color=INK_MUTED, pad=16)
    ax.tick_params(labelsize=12.5, colors=INK_PRIMARY, length=4.5, width=1.1)
    ax.set_ylim(top=y_top * 1.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_PRIMARY)
        ax.spines[spine].set_linewidth(1.2)

    plt.tight_layout(pad=0.6)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
