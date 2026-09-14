#!/usr/bin/env python3
"""
Figura 1.1 do Capítulo 1 (Introdução) do TCC.

Gráfico de barras com o total de jogadores movimentados por temporada, na
janela de 30 dias antes do trade deadline -- o recorte temporal adotado
nesta pesquisa (ver Capítulo 3), a partir de data/team_trade_activity.csv.

Quarta versão: troca a fonte serifada/DejaVu Sans para Nimbus Sans, e
introduz um acento azul no lugar do cinza monocromático.

Quinta versão: remove o acento azul -- o orientando preferiu manter esta
figura monocromática, junto com a Figura 2.2, para não competir com as
duas figuras conceituais (1.2 e 2.1), que também voltaram ao monocromático.
Só a tipografia (Nimbus Sans) permanece da mudança anterior.

Output: viz/introducao_volume_trocas_temporada.png
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "team_trade_activity.csv"
OUT_PATH = PROJECT_DIR / "viz" / "introducao_volume_trocas_temporada.png"

INK = "#15181c"
MUTED = "#5a6472"
BAR_FILL = "#eceef1"
GRID = "#d7dbe0"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"


def season_label(year_end: int) -> str:
    return f"{str(year_end - 1)[-2:]}-{str(year_end)[-2:]}"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    agg = (
        df.groupby("season_end_year")
        .agg(total_players_moved=("n_players_moved", "sum"))
        .reset_index()
    )
    agg["label"] = agg["season_end_year"].apply(season_label)

    first_val = agg["total_players_moved"].iloc[0]
    last_val = agg["total_players_moved"].iloc[-1]
    growth_pct = round((last_val - first_val) / first_val * 100)

    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(agg)))
    ax.bar(x, agg["total_players_moved"], color=BAR_FILL, edgecolor=INK,
           linewidth=1.1, width=0.62, zorder=3)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.set_major_locator(MultipleLocator(20))

    for xi, val in zip(x, agg["total_players_moved"]):
        ax.text(xi, val + 2.6, f"{val}", ha="center", va="bottom",
                 fontsize=8.6, color=INK, family=SANS)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["label"], fontsize=8.6, color=MUTED, family=SANS)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=3, labelsize=8.4, colors=MUTED)
    for lbl in ax.get_yticklabels():
        lbl.set_family(SANS)

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(INK)
        ax.spines[spine_name].set_linewidth(1.0)

    y_top = agg["total_players_moved"].max() * 1.32
    ax.set_ylim(0, y_top)

    bracket_y = agg["total_players_moved"].max() * 1.20
    x0, x1 = x[0], x[-1]
    ax.plot([x0, x0, x1, x1],
            [bracket_y - 0.035 * y_top, bracket_y, bracket_y, bracket_y - 0.035 * y_top],
            color=INK, lw=1.0, zorder=4, solid_capstyle="round")
    ax.text((x0 + x1) / 2, bracket_y + 0.03 * y_top, f"+{growth_pct}%",
             ha="center", va="bottom", fontsize=11, color=INK,
             fontweight="bold", family=SANS)

    ax.set_xlabel("Temporada", fontsize=9.4, color=MUTED, labelpad=8, family=SANS)
    ax.set_ylabel(
        "Jogadores movimentados\n(janela de 30 dias antes do trade deadline)",
        fontsize=8.8, color=MUTED, labelpad=8, family=SANS,
    )

    plt.tight_layout(pad=0.6)
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    print(agg.to_string(index=False))
    print(f"Growth {first_val} -> {last_val}: +{growth_pct}%")


if __name__ == "__main__":
    main()
