#!/usr/bin/env python3
"""
Figures that directly mirror the ones in Sa (2025), "Contribuicao Relativa:
Detectando Buracos de Elenco na NBA" (Kaio-Lucas-de-Sa-2019006850-Contribuicao-
Relativa.pdf), section V, applied to this project's real data (2019-2021 ESPN WPA):

  Fig A (paper Fig. 4) - histograma de WPA_bar por jogador (concentracao em zero)
  Fig B (paper Fig. 5) - relacao entre WPA_bar extrema e numero de jogos disputados
  Fig C (paper Fig. 7) - histograma de RCP_bar por posicao-time-temporada
  Fig D (paper Fig. 9) - scatter deltaP_Centers x deltaP_Guards, extremos destacados
  Fig E (paper Fig. 10) - evolucao (pre-troca -> pos-troca) de um caso real, no mesmo
                           espaco deltaP_Centers x deltaP_Guards

All purely descriptive (no significance tests), consistent with the decision to not
run inferential statistics on only 3 real seasons.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import (  # noqa: E402
    PROJECT_DIR, START_YEAR, END_YEAR, get_engine, load_espn_player_box, compute_wpa_acc,
)
from build_wpa_trade_analysis import league_window_snapshot, season_start_date  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

VIZ_DIR = PROJECT_DIR / "viz"
DATA_DIR = PROJECT_DIR / "data"

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = SURFACE
plt.rcParams['axes.facecolor'] = SURFACE
plt.rcParams['axes.edgecolor'] = GRID
plt.rcParams['grid.color'] = GRID
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['text.color'] = INK_PRIMARY
plt.rcParams['axes.labelcolor'] = INK_PRIMARY
plt.rcParams['xtick.color'] = INK_SECONDARY
plt.rcParams['ytick.color'] = INK_SECONDARY


def style_ax(ax):
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)


# -----------------------------------------------------------------------------
# Fig A + B: distribuicao de WPA_bar por jogador, e sua relacao com jogos disputados
# -----------------------------------------------------------------------------
def generate_wpa_bar_figures():
    engine = get_engine()
    pb = load_espn_player_box(engine)
    wpa_acc = compute_wpa_acc(pb, cutoff_date=None, start_date=None)
    wpa_acc = wpa_acc[wpa_acc["games"] > 0].copy()
    wpa_acc["WPA_bar"] = wpa_acc["WPA_acc"] / wpa_acc["games"]

    # Fig A: histogram
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.hist(wpa_acc["WPA_bar"], bins=60, color=BLUE, edgecolor=INK_PRIMARY, linewidth=0.4, zorder=3)
    ax.axvline(0, color=INK_MUTED, linestyle='--', linewidth=1.2, zorder=2)
    ax.set_title("Distribuição do WPA_bar por Jogador (2019-2021)", fontsize=15, fontweight='bold',
                  color=INK_PRIMARY, pad=12)
    ax.set_xlabel("WPA_bar (WPA_acc / jogos disputados)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Número de jogadores-temporada", fontsize=12, fontweight='bold')
    style_ax(ax)
    fig.text(0.5, -0.02, f"n={len(wpa_acc)} combinações jogador-time-temporada. "
              "Concentração acentuada perto de zero, como no paper original (Fig. 4).",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "wpa_bar_distribuicao.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")

    # Fig B: extreme WPA_bar vs games played
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    p05, p95 = wpa_acc["WPA_bar"].quantile([0.05, 0.95])
    is_extreme = (wpa_acc["WPA_bar"] <= p05) | (wpa_acc["WPA_bar"] >= p95)
    ax.scatter(wpa_acc.loc[~is_extreme, "games"], wpa_acc.loc[~is_extreme, "WPA_bar"],
               s=14, color=INK_MUTED, alpha=0.35, zorder=3, label="Demais jogadores")
    ax.scatter(wpa_acc.loc[is_extreme, "games"], wpa_acc.loc[is_extreme, "WPA_bar"],
               s=22, color=RED, alpha=0.75, zorder=4, label="WPA_bar extremo (5% caudas)")
    ax.axhline(0, color=INK_MUTED, linestyle='--', linewidth=1.0, zorder=2)
    ax.set_title("WPA_bar Extremo x Número de Jogos Disputados (2019-2021)", fontsize=14.5,
                  fontweight='bold', color=INK_PRIMARY, pad=12)
    ax.set_xlabel("Número de jogos disputados na temporada (com o time)", fontsize=12, fontweight='bold')
    ax.set_ylabel("WPA_bar", fontsize=12, fontweight='bold')
    style_ax(ax)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    fig.text(0.5, -0.02,
              "Valores extremos de WPA_bar se concentram em jogadores com poucos jogos disputados — "
              "o mesmo viés de volume identificado no paper original (Fig. 5).",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "wpa_bar_vs_jogos.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# Fig C: distribuicao de RCP_bar (posicao-time-temporada)
# -----------------------------------------------------------------------------
def generate_rcp_bar_distribution(pos: pd.DataFrame):
    df = pos[pos["stable"]].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.hist(df["RCP_pos_bar"], bins=50, color=AQUA, edgecolor=INK_PRIMARY, linewidth=0.4, zorder=3)
    ax.axvline(0, color=INK_MUTED, linestyle='--', linewidth=1.2, zorder=2)
    ax.set_title("Distribuição do RCP_bar (Contribuição Relativa Média por Partida)", fontsize=14.5,
                  fontweight='bold', color=INK_PRIMARY, pad=12)
    ax.set_xlabel("RCP_bar_P (por posição-time-temporada)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Número de observações (posição x time x temporada)", fontsize=12, fontweight='bold')
    style_ax(ax)
    fig.text(0.5, -0.02, f"n={len(df)} observações (30 times x 3 posições x 3 temporadas, estáveis). "
              "Sem outliers extremos, diferente da distribuição do WPA_bar (Fig. 7 do paper original).",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "rcp_bar_distribuicao.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# Fig D: scatter deltaP_Centers x deltaP_Guards, por temporada, extremos destacados
# -----------------------------------------------------------------------------
def generate_quadrant_scatter(pos: pd.DataFrame):
    df = pos[pos["stable"]].copy()
    piv_all = df.pivot_table(index=["team_id", "team_abbreviation", "season_end_year"],
                              columns="posicao", values="deltaP").reset_index()
    piv_all = piv_all.dropna(subset=["Guards", "Centers"])

    seasons = sorted(piv_all["season_end_year"].unique())
    season_labels = {2019: "2018-19", 2020: "2019-20", 2021: "2020-21"}
    xlim = (piv_all["Guards"].min() * 1.15, piv_all["Guards"].max() * 1.15)
    ylim = (piv_all["Centers"].min() * 1.15, piv_all["Centers"].max() * 1.15)

    fig, axes = plt.subplots(1, len(seasons), figsize=(6.2 * len(seasons), 6.8), dpi=300, sharex=True, sharey=True)
    for ax, season in zip(axes, seasons):
        piv = piv_all[piv_all["season_end_year"] == season]
        ax.scatter(piv["Guards"], piv["Centers"], s=55, color=INK_MUTED, alpha=0.5, zorder=3,
                   edgecolor='white', linewidth=0.5)
        ax.axhline(0, color=INK_PRIMARY, linewidth=1.0, zorder=2, linestyle='--')
        ax.axvline(0, color=INK_PRIMARY, linewidth=1.0, zorder=2, linestyle='--')

        extremes = {
            "Maior δP_Centers": piv.loc[piv["Centers"].idxmax()],
            "Menor δP_Centers": piv.loc[piv["Centers"].idxmin()],
            "Maior δP_Guards": piv.loc[piv["Guards"].idxmax()],
            "Menor δP_Guards": piv.loc[piv["Guards"].idxmin()],
        }
        seen = set()
        for label, row in extremes.items():
            key = row["team_abbreviation"]
            color = RED if "Menor" in label else BLUE
            ax.scatter([row["Guards"]], [row["Centers"]], s=150, color=color, zorder=5,
                       edgecolor=INK_PRIMARY, linewidth=1.3)
            if key not in seen:
                ax.annotate(row["team_abbreviation"], (row["Guards"], row["Centers"]),
                            textcoords="offset points", xytext=(8, 8), fontsize=10,
                            fontweight='bold', color=INK_PRIMARY)
                seen.add(key)

        ax.set_title(f"{season_labels[season]}  (n={len(piv)} times)", fontsize=13, fontweight='bold',
                     color=INK_PRIMARY, pad=10)
        ax.set_xlabel("δP_Guards (Armadores)", fontsize=11.5, fontweight='bold')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_ax(ax)

    axes[0].set_ylabel("δP_Centers (Pivôs)", fontsize=12.5, fontweight='bold')
    fig.suptitle("Lacunas Posicionais por Temporada: δP_Centers (Pivôs) x δP_Guards (Armadores)", fontsize=15.5,
                 fontweight='bold', y=1.04, color=INK_PRIMARY)
    fig.text(0.5, -0.03,
              "Cada ponto = 1 time nessa temporada. Vermelho = extremo negativo (maior lacuna); "
              "azul = extremo positivo (maior virtude). Réplica por temporada do quadrante do paper original (Fig. 9).",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "quadrante_pivo_armador.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")
    return piv_all


# -----------------------------------------------------------------------------
# Fig E: evolucao pre-troca -> pos-troca de um caso real, no mesmo espaco
# -----------------------------------------------------------------------------
def compute_trade_snapshot_deltaP(pb, season, trade_date, team_id, window_start, window_end):
    snap = league_window_snapshot(pb, season, window_start, window_end)
    stable = snap[snap["stable"]]
    muP = stable.groupby("posicao")["RCP_pos_bar"].mean()
    team_row = snap[snap["team_id"] == team_id].set_index("posicao")["RCP_pos_bar"]
    return {p: float(team_row.get(p, np.nan) - muP.get(p, np.nan)) for p in ["Guards", "Centers"]}


def generate_evolution_chart(piv_background: pd.DataFrame):
    engine = get_engine()
    pb = load_espn_player_box(engine)

    cases = [
        {"label": "BKN (James Harden, 14/01/2021)", "team_id": 1610612751, "season": 2021,
         "trade_date": pd.Timestamp("2021-01-14"), "color": BLUE},
    ]

    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=300)
    case_season = cases[0]["season"]
    bg = piv_background[piv_background["season_end_year"] == case_season]
    ax.scatter(bg["Guards"], bg["Centers"], s=35, color=INK_MUTED,
               alpha=0.3, zorder=2, label=f"Demais equipes, temporada {case_season}")
    ax.axhline(0, color=INK_PRIMARY, linewidth=1.0, zorder=2, linestyle='--')
    ax.axvline(0, color=INK_PRIMARY, linewidth=1.0, zorder=2, linestyle='--')

    for case in cases:
        season_start = season_start_date(engine, case["season"])
        from wpa_common import season_date_bounds  # noqa: E402
        _, season_end = season_date_bounds(engine, case["season"])
        season_end_excl = season_end + pd.Timedelta(days=1)

        pre = compute_trade_snapshot_deltaP(pb, case["season"], case["trade_date"], case["team_id"],
                                             season_start, case["trade_date"])
        post = compute_trade_snapshot_deltaP(pb, case["season"], case["trade_date"], case["team_id"],
                                              case["trade_date"], season_end_excl)

        xs = [pre["Guards"], post["Guards"]]
        ys = [pre["Centers"], post["Centers"]]
        ax.plot(xs, ys, color=case["color"], linewidth=2.2, zorder=4,
                marker='o', markersize=11, markeredgecolor=INK_PRIMARY, markeredgewidth=1.2)
        ax.annotate("pré-troca", (xs[0], ys[0]), textcoords="offset points", xytext=(-10, -14),
                    fontsize=9.5, fontweight='bold', color=case["color"], ha='right')
        ax.annotate("pós-troca", (xs[1], ys[1]), textcoords="offset points", xytext=(10, 8),
                    fontsize=9.5, fontweight='bold', color=case["color"])
        ax.annotate(case["label"], (xs[0], ys[0]), textcoords="offset points", xytext=(-10, 14),
                    fontsize=10.5, fontweight='bold', color=INK_PRIMARY, ha='right')

    ax.set_title("Evolução do Perfil Posicional: Pré-Troca → Pós-Troca (caso real)", fontsize=14.5,
                  fontweight='bold', color=INK_PRIMARY, pad=14)
    ax.set_xlabel("δP_Guards (Armadores)", fontsize=12.5, fontweight='bold')
    ax.set_ylabel("δP_Centers (Pivôs)", fontsize=12.5, fontweight='bold')
    style_ax(ax)
    ax.legend(loc='lower right', fontsize=9.5, frameon=False)
    fig.text(0.5, -0.02,
              "Réplica, no contexto de trocas intra-temporada, do gráfico de evolução do paper original (Fig. 10) — "
              "lá a comparação era entre temporadas consecutivas; aqui, entre o pré e o pós de uma troca real.",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "evolucao_pre_pos_troca.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("Generating paper-aligned WPA/RCP figures (Sa, 2025) from real 2019-2021 data...")
    generate_wpa_bar_figures()
    pos_df = pd.read_csv(DATA_DIR / "wpa_positional_analysis.csv")
    generate_rcp_bar_distribution(pos_df)
    piv = generate_quadrant_scatter(pos_df)
    generate_evolution_chart(piv)
    print("Done.")
