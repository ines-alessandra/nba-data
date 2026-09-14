#!/usr/bin/env python3
"""
Team-spotlight charts, one figure per season: for a handful of selected teams,
show every player they acquired that season, the pre-trade deltaP that drove the
Preenche Lacuna / Redundante classification, and (where available) the post-trade
deltaP - i.e. whether the diagnosed gap actually closed.

Team selection per season (not cherry-picked for outcome - picked on structural
criteria before looking at results): the team with the single most negative
pre-trade deltaP (biggest diagnosed gap) that season, the team(s) with the most
acquisitions that season (richest multi-position story), for 4 teams per season.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR  # noqa: E402

BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DATA_DIR = PROJECT_DIR / "data"
VIZ_DIR = PROJECT_DIR / "viz"

SEASON_LABELS = {2019: "2018-19", 2020: "2019-20", 2021: "2020-21"}

# Selected on structural criteria (biggest diagnosed gap that season + teams with
# the most acquisitions that season), fixed BEFORE looking at any outcome numbers.
TEAM_SELECTION = {
    2019: ["UTA", "PHI", "CLE", "DAL"],
    2020: ["MEM", "MIN", "ATL", "DAL"],
    2021: ["BKN", "HOU", "ORL", "SAC"],
}

CLASS_COLORS = {"Preenche Lacuna": AQUA, "Redundante": RED}

GOLD = "#a8860a"
GAP_COLORS = {
    "Sim": AQUA, "Melhorou": GOLD, "Nao": RED,
    "N/A (nao era lacuna)": INK_MUTED, "Sem dado pos-troca": INK_MUTED,
}
GAP_SHORT_LABEL = {
    "Sim": "fechou", "Melhorou": "melhorou, não fechou", "Nao": "não melhorou",
    "N/A (nao era lacuna)": "não era lacuna",
}


def style_ax(ax):
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)


def generate_season_spotlight(df: pd.DataFrame, season: int, teams: list):
    sub_season = df[df["season_end_year"] == season]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    axes = axes.flatten()
    xmax = sub_season["deltaP"].abs().max()
    xmax_post = sub_season["deltaP_post_trade"].abs().max()
    xlim = max(xmax, xmax_post if not np.isnan(xmax_post) else 0) * 1.25

    for ax, team in zip(axes, teams):
        t = sub_season[sub_season["team_abbreviation"] == team].copy()
        t = t.sort_values("deltaP", ascending=True)
        t["label"] = t["player_name"] + "  (" + t["posicao"].str[:-1] + ")"

        y = np.arange(len(t))
        colors = [CLASS_COLORS[c] for c in t["preenche_lacuna"]]
        ax.barh(y, t["deltaP"], color=colors, edgecolor=INK_PRIMARY, linewidth=0.8,
                height=0.55, zorder=3)

        for yi, (pre, post, gap_status) in enumerate(zip(t["deltaP"], t["deltaP_post_trade"], t["gap_fechada"])):
            if pd.isna(post):
                continue
            dot_color = GAP_COLORS.get(gap_status, INK_MUTED)
            ax.plot([pre, post], [yi, yi], color=INK_MUTED, linewidth=1.0, linestyle=':', zorder=2)
            ax.scatter([post], [yi], marker='D', s=65, color=dot_color, edgecolor=INK_PRIMARY,
                       linewidth=0.9, zorder=5)
            label = GAP_SHORT_LABEL.get(gap_status)
            if label:
                offset = xlim * 0.02
                ha = 'left' if post >= 0 else 'right'
                ax.text(post + (offset if post >= 0 else -offset), yi, label, fontsize=7.8,
                        color=dot_color, fontweight='bold', va='center', ha=ha, zorder=6)

        ax.axvline(0, color=INK_PRIMARY, linewidth=1.1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(t["label"], fontsize=9.5)
        ax.set_xlim(-xlim * 1.35, xlim * 1.35)
        n_lacuna = int((t["preenche_lacuna"] == "Preenche Lacuna").sum())
        n_red = int((t["preenche_lacuna"] == "Redundante").sum())
        ax.set_title(f"{team}  —  {n_lacuna} preenche lacuna, {n_red} redundante",
                     fontsize=12.5, fontweight='bold', color=INK_PRIMARY, pad=8)
        ax.set_xlabel("δP  (barra = pré-troca; ◆ = pós-troca, cor = veredito)", fontsize=9.5, color=INK_SECONDARY)
        style_ax(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=AQUA, edgecolor=INK_PRIMARY, label="Barra: Preenche Lacuna (pré)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor=INK_PRIMARY, label="Barra: Redundante (pré)"),
        plt.Line2D([0], [0], marker='D', color=AQUA, linestyle='', markersize=8, markeredgecolor=INK_PRIMARY, label="◆ pós: lacuna fechou"),
        plt.Line2D([0], [0], marker='D', color=GOLD, linestyle='', markersize=8, markeredgecolor=INK_PRIMARY, label="◆ pós: melhorou, não fechou"),
        plt.Line2D([0], [0], marker='D', color=RED, linestyle='', markersize=8, markeredgecolor=INK_PRIMARY, label="◆ pós: não melhorou"),
        plt.Line2D([0], [0], marker='D', color=INK_MUTED, linestyle='', markersize=8, markeredgecolor=INK_PRIMARY, label="◆ pós: não era lacuna"),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=9.5, frameon=False)
    fig.suptitle(f"Times em Destaque — {SEASON_LABELS[season]}: Trocas, Classificação e Lacunas",
                 fontsize=16, fontweight='bold', y=1.13, color=INK_PRIMARY)
    fig.text(0.5, -0.02,
              "Cada barra é um jogador adquirido. A COR do losango (não a posição) é o que importa: verde = a lacuna "
              "fechou de fato; dourado = melhorou mas não fechou; vermelho = não melhorou; cinza = não era lacuna, "
              "ou não há dado pós-troca disponível.",
              ha='center', fontsize=9.3, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / f"times_destaque_{season}.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "wpa_trade_classification.csv")
    for season, teams in TEAM_SELECTION.items():
        generate_season_spotlight(df, season, teams)
