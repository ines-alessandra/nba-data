#!/usr/bin/env python3
"""
Figura 3.2 do Capítulo 3 (Metodologia) do TCC.

Ilustra geometricamente por que este trabalho pondera o Índice de Gini por
minutos jogados em vez de usar a versão simples: dois painéis com a mesma
curva de Lorenz (a mesma construção geométrica usada em
salary_inequality_metrics.calculate_gini_simple/calculate_gini_weighted_minutes
-- área entre a curva e a diagonal de igualdade perfeita), um contando cada
jogador como uma unidade e outro ponderando pelos minutos jogados.

Os salários e minutos usados são ILUSTRATIVOS (um elenco fictício de 8
jogadores, não uma equipe real), construídos deliberadamente com um reserva
bem pago mas com poucos minutos, para deixar visível o efeito da ponderação.
O eixo horizontal do painel (a) é a fração acumulada de JOGADORES; no
painel (b) é a fração acumulada de MINUTOS -- por isso a mesma folha
salarial produz duas curvas diferentes.

Output: viz/metodologia_curva_lorenz.png
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "metodologia_curva_lorenz.png"

INK = "#15181c"
MUTED = "#5a6472"
SHADE = "#dde1e6"
SANS = "Nimbus Sans"

# Elenco fictício (8 jogadores) -- salário em US$ milhões, minutos na temporada.
# O jogador 5 é o caso deliberado: salário alto (8.5), poucos minutos (400).
SALARY = np.array([1.2, 1.8, 2.4, 3.6, 8.5, 6.0, 11.0, 18.0])
MINUTES = np.array([900, 1400, 1100, 1800, 400, 2000, 2300, 2500])


def lorenz_curve(values: np.ndarray, weights: np.ndarray):
    order = np.argsort(values)
    v_sorted, w_sorted = values[order], weights[order]
    w_total = w_sorted.sum()
    wv_total = (v_sorted * w_sorted).sum()
    cum_w = np.concatenate([[0.0], np.cumsum(w_sorted)]) / w_total
    cum_wv = np.concatenate([[0.0], np.cumsum(v_sorted * w_sorted)]) / wv_total
    return cum_w, cum_wv


def gini_from_lorenz(p: np.ndarray, l_: np.ndarray) -> float:
    b = np.sum((l_[1:] + l_[:-1]) / 2.0 * (p[1:] - p[:-1]))
    return 1.0 - 2.0 * b


def panel(ax, p, l_, gini, x_label, panel_letter, panel_title, panel_sub):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.3, ls="--", zorder=2)
    ax.plot(p, l_, color=INK, lw=2.2, zorder=4)
    ax.fill_between(p, l_, p, color=SHADE, zorder=1)
    ax.scatter(p, l_, s=26, color=INK, zorder=5, linewidths=0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(1.2)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=13.5, family=SANS)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=13.5, family=SANS)
    ax.tick_params(length=4.5, colors=INK, width=1.1)
    ax.set_xlabel(x_label, fontsize=14.5, family=SANS, color=INK, labelpad=10)
    ax.set_ylabel("proporção acumulada\ndo salário", fontsize=14.5, family=SANS,
                   color=INK, labelpad=10, linespacing=1.3)

    ax.set_title(panel_letter, fontsize=19.0, family=SANS, color=INK,
                 fontweight="bold", loc="left", pad=12)
    ax.text(0.56, 0.13, f"Gini = {gini:.2f}", fontsize=15.0, family=SANS,
             color=INK, fontweight="bold", transform=ax.transAxes)

    return panel_title, panel_sub


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    p_simple, l_simple = lorenz_curve(SALARY, np.ones_like(SALARY))
    gini_simple = gini_from_lorenz(p_simple, l_simple)
    t0, s0 = panel(axes[0], p_simple, l_simple, gini_simple,
                    "proporção acumulada\nde jogadores", "(a)",
                    "Gini simples", "cada jogador conta como 1 unidade")

    p_weighted, l_weighted = lorenz_curve(SALARY, MINUTES)
    gini_weighted = gini_from_lorenz(p_weighted, l_weighted)
    t1, s1 = panel(axes[1], p_weighted, l_weighted, gini_weighted,
                    "proporção acumulada\nde minutos jogados", "(b)",
                    "Gini ponderado por minutos", "cada jogador pesa pelos minutos jogados")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.72, bottom=0.30, wspace=0.60)

    for ax, title, sub in zip(axes, (t0, t1), (s0, s1)):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.965, title, ha="center", va="center", fontsize=16.5,
                  family=SANS, color=INK, fontweight="bold")
        fig.text(cx, 0.885, sub, ha="center", va="center", fontsize=12.5,
                  family=SANS, color=MUTED, style="italic")

    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    print(f"Gini simples ilustrativo = {gini_simple:.3f}; "
          f"Gini ponderado ilustrativo = {gini_weighted:.3f}")


if __name__ == "__main__":
    main()
