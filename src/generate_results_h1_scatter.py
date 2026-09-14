#!/usr/bin/env python3
"""
Figura 4.1 do Capítulo 4 (Resultados e Discussão) do TCC.

Dispersão completa da amostra da Hipótese 1 (N=282 times-temporada,
2012-13 a 2024-25): Delta Gini ponderado por minutos (eixo x) x Delta
Performance Ajustada (eixo y), com os quatro quadrantes definidos pelo
cruzamento de sinais rotulados pela teoria que favorecem (Equidade vs.
Torneio) e o percentual da amostra em cada um.

Dados: data/salary_inequality_trades_analysis.csv (delta_gini,
delta_adjusted_perf), o mesmo par de variáveis testado por Spearman na
Seção 4.1/Tabela 4.1 do capítulo.

Paleta de cor idêntica à já usada em src/generate_h1_verdict.py (os slides
apresentados ao orientador): azul = polo Equidade, vermelho = polo Torneio.
Fonte Nimbus Sans, igual ao restante do trabalho.

Output: viz/resultados_h1_dispersao.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "salary_inequality_trades_analysis.csv"
OUT_PATH = PROJECT_DIR / "viz" / "resultados_h1_dispersao.png"

# Mesma paleta de generate_h1_verdict.py.
COLOR_EQUIDADE = "#2a78d6"        # polo azul
COLOR_TORNEIO = "#e34948"         # polo vermelho
TINT_EQUIDADE = "#dce9f9"
TINT_TORNEIO = "#fbdedd"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
SANS = "Nimbus Sans"


def main() -> None:
    plt.rcParams["font.family"] = SANS
    df = pd.read_csv(DATA_PATH)
    x, y = df["delta_gini"], df["delta_adjusted_perf"]
    n = len(df)

    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x_pad = (x.max() - x.min()) * 0.08
    y_pad = (y.max() - y.min()) * 0.10
    x_lo, x_hi = x.min() - x_pad, x.max() + x_pad
    y_lo, y_hi = y.min() - y_pad, y.max() + y_pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Metade esquerda (Delta Gini < 0) = território da Teoria da Equidade;
    # metade direita (Delta Gini > 0) = território da Teoria do Torneio --
    # mesma lógica de generate_h1_verdict.py, agora estendida ao plano
    # completo em vez de só à régua de rho.
    ax.axvspan(x_lo, 0, color=TINT_EQUIDADE, alpha=0.55, zorder=0)
    ax.axvspan(0, x_hi, color=TINT_TORNEIO, alpha=0.45, zorder=0)

    ax.axhline(0, color=INK_PRIMARY, lw=1.1, zorder=2)
    ax.axvline(0, color=INK_PRIMARY, lw=1.1, zorder=2)

    quad_masks = {
        "torneio_sup": (x > 0) & (y > 0),   # a favor do Torneio
        "torneio_con": (x > 0) & (y < 0),   # contraria o Torneio
        "equid_sup": (x < 0) & (y > 0),     # a favor da Equidade
        "equid_con": (x < 0) & (y < 0),     # contraria a Equidade
    }
    quad_style = {
        "torneio_sup": dict(color=COLOR_TORNEIO, alpha=0.75),
        "torneio_con": dict(color=COLOR_TORNEIO, alpha=0.30),
        "equid_sup": dict(color=COLOR_EQUIDADE, alpha=0.75),
        "equid_con": dict(color=COLOR_EQUIDADE, alpha=0.30),
    }
    counts = {}
    for key, mask in quad_masks.items():
        counts[key] = int(mask.sum())
        ax.scatter(x[mask], y[mask], s=36, edgecolors="none", zorder=3, **quad_style[key])

    def pct(k):
        return f"{100*counts[k]/n:.1f}".replace(".", ",")

    label_kw = dict(fontsize=13.5, family=SANS, linespacing=1.55, zorder=5)
    ax.text(x_hi * 0.60, y_hi * 0.86,
            f"Torneio\n(a favor)\n{counts['torneio_sup']} times ({pct('torneio_sup')}%)",
            ha="center", va="center", color=COLOR_TORNEIO, fontweight="bold", **label_kw)
    ax.text(x_hi * 0.60, y_lo * 0.86,
            f"Torneio\n(contra)\n{counts['torneio_con']} times ({pct('torneio_con')}%)",
            ha="center", va="center", color=INK_SECONDARY, **label_kw)
    ax.text(x_lo * 0.55, y_hi * 0.86,
            f"Equidade\n(a favor)\n{counts['equid_sup']} times ({pct('equid_sup')}%)",
            ha="center", va="center", color=COLOR_EQUIDADE, fontweight="bold", **label_kw)
    ax.text(x_lo * 0.55, y_lo * 0.86,
            f"Equidade\n(contra)\n{counts['equid_con']} times ({pct('equid_con')}%)",
            ha="center", va="center", color=INK_SECONDARY, **label_kw)

    ax.set_xlabel("Δ Gini ponderado por minutos (pós − pré-deadline)",
                  fontsize=15.0, color=INK_PRIMARY, labelpad=12)
    ax.set_ylabel("Δ Performance Ajustada (pós − pré-deadline)",
                  fontsize=15.0, color=INK_PRIMARY, labelpad=12)
    ax.tick_params(labelsize=12.5, colors=INK_PRIMARY, length=4.5, width=1.1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_PRIMARY)
        ax.spines[spine].set_linewidth(1.2)

    fig.text(0.5, 0.965,
              f"Spearman $\\rho$ = -0,1892   p = 0,0014   N = {n}   (efeito fraco, Cohen 1988)",
              ha="center", va="center", fontsize=15.0, color=INK_PRIMARY, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
