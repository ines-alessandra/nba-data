#!/usr/bin/env python3
"""
Figura 2.1 do Capítulo 2 (Fundamentação Teórica) do TCC.

Sétima versão: remove o texto de título/banner acima dos painéis
("Métricas com distribuição não normal...") -- essa informação já está
no parágrafo que antecede a figura no texto, e a linha ficava redundante
dentro da imagem. Também reduz a largura do canvas e aumenta bastante o
tamanho de fonte de todos os elementos (nomes dos testes, legendas de
condição, rótulos dos eixos): a versão anterior usava uma figura muito
larga (13.6 pol.) incluída em .95\\textwidth, o que encolhia o texto a
um tamanho ilegível na página impressa. Nomes de teste foram encurtados
para "Nome (símbolo)" e legendas de condição mais longas foram quebradas
em duas linhas, para caber com folga no novo tamanho de fonte sem
estourar a largura de cada painel.

Output: viz/fundamentacao_decisao_testes_estatisticos.png
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "fundamentacao_decisao_testes_estatisticos.png"

INK = "#15181c"
MUTED = "#5a6472"
SANS = "Nimbus Sans"


def style_axes(ax, xlabel, ylabel, xticks=None, xticklabels=None):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(1.2)
    ax.set_yticks([])
    if xticks is not None:
        # Grupos categóricos: os rótulos dos ticks já nomeiam o eixo x,
        # então nenhum xlabel adicional é usado (evita duas legendas
        # competindo pelo mesmo espaço vertical abaixo do painel).
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=11.5, family=SANS)
        ax.tick_params(axis="x", length=4, colors=INK, width=1.1)
    else:
        ax.set_xticks([])
        ax.set_xlabel(xlabel, fontsize=12.0, family=SANS, color=INK, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12.0, family=SANS, color=INK, labelpad=7)


def panel_a(ax):
    rng = np.random.default_rng(7)
    x = np.linspace(0.6, 9.4, 11) + rng.uniform(-0.35, 0.35, 11)
    y = 0.85 * x + rng.normal(0, 1.15, 11) + 0.8
    order = np.argsort(x)
    fit = np.poly1d(np.polyfit(x, y, 1))
    ax.plot(x[order], fit(x[order]), color=MUTED, lw=1.5, ls="--", zorder=2)
    ax.scatter(x, y, s=42, color=INK, zorder=3, linewidths=0)
    ax.set_xlim(-0.6, 10.6)
    style_axes(ax, "Variável X", "Variável Y")


def panel_b(ax):
    rng = np.random.default_rng(3)
    g1 = rng.normal(4.1, 0.95, 18)
    g2 = rng.normal(6.6, 0.95, 18)
    ax.boxplot([g1, g2], widths=0.42, patch_artist=False, showfliers=False,
               medianprops=dict(color=INK, lw=1.9),
               boxprops=dict(color=INK, lw=1.3),
               whiskerprops=dict(color=INK, lw=1.3),
               capprops=dict(color=INK, lw=1.3))
    style_axes(ax, "", "Variável de\ninteresse",
               xticks=[1, 2], xticklabels=["Grupo A", "Grupo B"])


def panel_c(ax):
    rng = np.random.default_rng(11)
    centers = [6.6, 5.0, 3.3]
    data = [rng.normal(c, 0.95, 14) for c in centers]
    ax.boxplot(data, widths=0.42, patch_artist=False, showfliers=False,
               medianprops=dict(color=INK, lw=1.9),
               boxprops=dict(color=INK, lw=1.3),
               whiskerprops=dict(color=INK, lw=1.3),
               capprops=dict(color=INK, lw=1.3))
    style_axes(ax, "", "Variável de\ninteresse",
               xticks=[1, 2, 3], xticklabels=["A", "B", "C"])


def panel_d(ax):
    rng = np.random.default_rng(5)
    n = 9
    pre = rng.normal(5.0, 1.0, n)
    post = pre + rng.normal(1.4, 0.7, n)
    for p0, p1 in zip(pre, post):
        ax.plot([0, 1], [p0, p1], color=MUTED, lw=1.2, zorder=2)
    ax.scatter(np.zeros(n), pre, s=38, color=INK, zorder=3, linewidths=0)
    ax.scatter(np.ones(n), post, s=38, color=INK, zorder=3, linewidths=0)
    ax.set_xlim(-0.45, 1.45)
    style_axes(ax, "", "Variável\npareada",
               xticks=[0, 1], xticklabels=["Antes", "Depois"])


PANELS = [
    dict(fn=panel_a, letter="(a)", test="Spearman ($\\rho$)",
         cond="duas variáveis\ncontínuas"),
    dict(fn=panel_b, letter="(b)", test="Mann-Whitney ($U$)",
         cond="dois grupos\nindependentes"),
    dict(fn=panel_c, letter="(c)", test="Kruskal-Wallis ($H$)",
         cond="três grupos\nindependentes"),
    dict(fn=panel_d, letter="(d)", test="Wilcoxon ($W$)",
         cond="amostras pareadas\n(pré/pós)"),
]


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 4.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    for ax, p in zip(axes, PANELS):
        ax.set_facecolor("none")
        p["fn"](ax)
        ax.set_title(p["letter"], fontsize=15.5, family=SANS, color=INK,
                     fontweight="bold", loc="left", pad=10)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.335, wspace=0.75)

    # Legendas de teste + condição, abaixo dos eixos
    for ax, p in zip(axes, PANELS):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.205, p["test"], ha="center", va="center", fontsize=13.5,
                  family=SANS, color=INK, fontweight="bold")
        fig.text(cx, 0.075, p["cond"], ha="center", va="center", fontsize=11.0,
                  family=SANS, color=MUTED, linespacing=1.35)

    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor="#ffffff")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
