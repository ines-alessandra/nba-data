#!/usr/bin/env python3
"""
NBA Trade Impact Analysis - WPA Visualizations Generator (per-season, descriptive)

Generates publication-quality PNG charts in viz/ (300 DPI), reading exclusively
from the real, computed CSVs produced by build_wpa_season_profile.py,
build_wpa_trade_analysis.py and describe_wpa_results.py. Every number plotted or
annotated (medians, percentages, counts) is read from those files at run time -
nothing here is hardcoded or simulated.

IMPORTANT: this is a DESCRIPTIVE application of the methodology, not a
hypothesis-testing exercise - there are too few complete seasons of real ESPN WPA
data (2019-2021 only) to run inferential statistics with any power. No p-values,
significance tests, or "H1/H2 confirmed/rejected" language appear anywhere here -
only what the methodology (WPA -> RCP_pos -> muP -> deltaP -> classification ->
delta_WPA_time) actually computes, shown per season plus a clearly-labeled pooled
panel for context.

Palette: validated categorical/diverging colors from the dataviz skill
(references/palette.md) - fixed hue order, not cycled; diverging blue/red for
polarity (virtude/lacuna, positive/negative impact); direct value labels used
throughout since two of the categorical hues (aqua, yellow) sit under the 3:1
contrast floor against a white surface (the skill's documented "relief rule").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Validated palette (dataviz skill, references/palette.md) - light mode
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3
YELLOW = "#eda100"    # categorical slot 4
RED = "#e34948"       # categorical slot 8 / diverging pole
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

CLASS_COLORS = {"Preenche Lacuna": AQUA, "Redundante": RED}
OBJ_COLORS = {"Contender": BLUE, "Intermediario": ORANGE, "Tanking": AQUA}
OBJ_LABELS = {"Contender": "Contenders", "Intermediario": "Intermediários", "Tanking": "Tanking"}
SEASON_LABELS = {2019: "2018-19", 2020: "2019-20", 2021: "2020-21"}
SEASONS = [2019, 2020, 2021]

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = SURFACE
plt.rcParams['axes.facecolor'] = SURFACE
plt.rcParams['axes.edgecolor'] = GRID
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = GRID
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['text.color'] = INK_PRIMARY
plt.rcParams['axes.labelcolor'] = INK_PRIMARY
plt.rcParams['xtick.color'] = INK_SECONDARY
plt.rcParams['ytick.color'] = INK_SECONDARY

VIZ_DIR = PROJECT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_DIR / "data"


def load_data():
    pos = pd.read_csv(DATA_DIR / "wpa_positional_analysis.csv")
    impact = pd.read_csv(DATA_DIR / "wpa_trade_impact_analysis.csv")
    return pos, impact


def style_ax(ax):
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)


# -----------------------------------------------------------------------------
# A. Distribuicao de lacunas posicionais (deltaP) por temporada, entre as 30 equipes
#
# NOTE: the MEAN of deltaP across the 30 teams is tautologically ~0 for every
# position/season (deltaP is itself a deviation from muP, the across-team mean),
# so a bar-of-means chart is uninformative by construction. What actually varies,
# and is worth showing, is the DISTRIBUTION across teams (spread, outliers).
# -----------------------------------------------------------------------------
def generate_positional_gaps_by_season(pos: pd.DataFrame):
    df = pos[pos["stable"]].copy()
    positions = ["Guards", "Forwards", "Centers"]
    pos_colors = {"Guards": BLUE, "Forwards": ORANGE, "Centers": AQUA}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=300, sharey=True)
    y_all = df["deltaP"]
    y_lim = (y_all.min() - 0.6, y_all.max() + 0.6)

    for ax, season in zip(axes, SEASONS):
        sub = df[df["season_end_year"] == season]
        sns.boxplot(data=sub, x='posicao', y='deltaP', order=positions,
                    hue='posicao', palette=pos_colors, legend=False,
                    width=0.55, fliersize=0, linewidth=1.2, ax=ax, zorder=3,
                    boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=1.8),
                    whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
        sns.stripplot(data=sub, x='posicao', y='deltaP', order=positions,
                       color=INK_PRIMARY, alpha=0.4, size=4, jitter=0.18, ax=ax, zorder=4)
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=2, linestyle='--')
        ax.set_title(f"{SEASON_LABELS[season]}  (n={sub['team_abbreviation'].nunique()} times)",
                     fontsize=12.5, fontweight='bold', color=INK_PRIMARY, pad=10)
        ax.set_xlabel("")
        ax.set_ylim(*y_lim)
        style_ax(ax)
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel("δP (RCP_bar_pos − μP) por time", fontsize=12, fontweight='bold')
    fig.suptitle("Distribuição das Lacunas Posicionais (δP) entre as 30 Equipes, por Temporada",
                  fontsize=15.5, fontweight='bold', y=1.04, color=INK_PRIMARY)
    fig.text(0.5, -0.03,
              "Cada ponto = 1 time. δP > 0 = virtude (contribuição posicional acima da média da liga)  |  δP < 0 = lacuna estrutural. "
              "A média de δP é 0 por construção (é um desvio da média) — o que varia é a dispersão entre times.",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "positional_gaps_by_season.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# B. Impacto das trocas: Preenche Lacuna vs Redundante, por temporada (descritivo)
# -----------------------------------------------------------------------------
def generate_h1_by_season(impact: pd.DataFrame):
    order = ['Preenche Lacuna', 'Redundante']
    scopes = SEASONS
    panel_titles = [SEASON_LABELS[s] for s in SEASONS]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), dpi=300, sharey=True)
    y_all = impact["delta_WPA_time"].dropna()
    y_lim = (y_all.min() - 0.05, y_all.max() + 0.06)

    for ax, scope, title in zip(axes, scopes, panel_titles):
        sub = impact[impact["season_end_year"] == scope]

        sns.boxplot(data=sub, x='trade_classification', y='delta_WPA_time', order=order,
                    hue='trade_classification', palette=CLASS_COLORS, legend=False,
                    width=0.5, fliersize=3, linewidth=1.2, ax=ax, zorder=3,
                    boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=1.8),
                    whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
        sns.stripplot(data=sub, x='trade_classification', y='delta_WPA_time', order=order,
                       color=INK_PRIMARY, alpha=0.35, size=3.5, jitter=0.15, ax=ax, zorder=4)

        medians = sub.groupby('trade_classification')['delta_WPA_time'].median().reindex(order)
        ns = sub.groupby('trade_classification').size().reindex(order).fillna(0).astype(int)
        for i, cls in enumerate(order):
            m = medians[cls]
            if pd.isna(m):
                continue
            ax.text(i, y_lim[1] - 0.012, f"med={m:+.3f}".replace(".", ","), ha='center', va='top',
                    fontsize=8.7, fontweight='bold', color=INK_PRIMARY)

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{lbl}\n(n={ns[lbl]})" for lbl in order], fontsize=10)
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=2, linestyle='--')
        ax.set_title(title, fontsize=12.5, fontweight='bold', color=INK_PRIMARY, pad=8)
        ax.set_xlabel("")
        style_ax(ax)
        ax.set_ylim(*y_lim)

    axes[0].set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle("Impacto das Trocas: Preenche Lacuna vs. Redundante, por Temporada", fontsize=16,
                  fontweight='bold', y=1.05, color=INK_PRIMARY)
    fig.text(0.5, -0.05,
              "Distribuição descritiva de ΔWPA_time por classificação (δP pré-troca). Sem teste de significância — "
              "amostra de 3 temporadas reais é pequena demais para inferência estatística.",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "h1_lacuna_vs_redundante_por_temporada.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# C. Impacto das trocas por objetivo sazonal, por temporada (descritivo)
# -----------------------------------------------------------------------------
def generate_h2_by_season(impact: pd.DataFrame):
    order = ['Contender', 'Intermediario', 'Tanking']
    scopes = SEASONS
    panel_titles = [SEASON_LABELS[s] for s in SEASONS]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), dpi=300, sharey=True)
    y_all = impact["delta_WPA_time"].dropna()
    y_lim = (y_all.min() - 0.05, y_all.max() + 0.07)

    for ax, scope, title in zip(axes, scopes, panel_titles):
        sub = impact[impact["season_end_year"] == scope]
        sub = sub[sub["objetivo_sazonal"].isin(order)]

        sns.boxplot(data=sub, x='objetivo_sazonal', y='delta_WPA_time', order=order,
                    hue='objetivo_sazonal', palette=OBJ_COLORS, legend=False,
                    width=0.55, fliersize=3, linewidth=1.2, ax=ax, zorder=3,
                    boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=1.8),
                    whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
        sns.stripplot(data=sub, x='objetivo_sazonal', y='delta_WPA_time', order=order,
                       color=INK_PRIMARY, alpha=0.35, size=3.5, jitter=0.15, ax=ax, zorder=4)

        ns = sub.groupby('objetivo_sazonal').size().reindex(order).fillna(0).astype(int)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{OBJ_LABELS[o]}\n(n={ns[o]})" for o in order], fontsize=9.5)
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=2, linestyle='--')
        ax.set_title(title, fontsize=12.5, fontweight='bold', color=INK_PRIMARY, pad=8)
        ax.set_xlabel("")
        style_ax(ax)
        ax.set_ylim(*y_lim)

    axes[0].set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle("Impacto das Trocas por Objetivo Sazonal, por Temporada", fontsize=16.5, fontweight='bold',
                  y=1.05, color=INK_PRIMARY)
    fig.text(0.5, -0.05,
              "Distribuição descritiva de ΔWPA_time por objetivo sazonal (rank de conferência pré-troca). "
              "Sem teste de significância — amostra de 3 temporadas reais é pequena demais para inferência estatística.",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "h2_objetivo_sazonal_por_temporada.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# D. Taxa de trocas com impacto positivo, por objetivo sazonal, por temporada
# -----------------------------------------------------------------------------
def generate_success_rate_by_season(impact: pd.DataFrame):
    order = ['Contender', 'Intermediario', 'Tanking']
    df = impact[impact["objetivo_sazonal"].isin(order)].copy()

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    x = np.arange(len(SEASONS))
    width = 0.24

    for i, obj in enumerate(order):
        rates, ns = [], []
        for season in SEASONS:
            sub = df[(df["season_end_year"] == season) & (df["objetivo_sazonal"] == obj)]
            rates.append((sub["delta_WPA_time"] > 0).mean() * 100 if len(sub) else np.nan)
            ns.append(len(sub))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, rates, width, label=OBJ_LABELS[obj], color=OBJ_COLORS[obj],
                       edgecolor=INK_PRIMARY, linewidth=0.7, zorder=3)
        for bar, r, n in zip(bars, rates, ns):
            if np.isnan(r):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, r + 2.2, f"{r:.0f}%\n(n={n})",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color=INK_PRIMARY)

    ax.axhline(50, color=INK_MUTED, linestyle='--', linewidth=1.3, zorder=2, label='Referência (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_LABELS[s] for s in SEASONS], fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% de trocas com ΔWPA_time > 0", fontsize=12.5, fontweight='bold')
    ax.set_xlabel("Temporada", fontsize=12.5, fontweight='bold')
    ax.set_title("Proporção de Trocas com Impacto Positivo, por Objetivo Sazonal e Temporada",
                  fontsize=14.5, fontweight='bold', pad=14, color=INK_PRIMARY)
    style_ax(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10.5, frameon=False)
    plt.tight_layout()
    out = VIZ_DIR / "h2_taxa_sucesso_por_temporada.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# E. Resumo descritivo por temporada (medianas, sem teste de significancia)
# -----------------------------------------------------------------------------
def generate_effect_summary(impact: pd.DataFrame):
    scopes = SEASONS
    row_labels = [SEASON_LABELS[s] for s in SEASONS]

    def median_diff(df, col, g1, g2):
        s = df.groupby(col)["delta_WPA_time"].median()
        if g1 not in s.index or g2 not in s.index:
            return None
        n1 = int((df[col] == g1).sum())
        n2 = int((df[col] == g2).sum())
        return {"diff": s[g1] - s[g2], "n1": n1, "n2": n2}

    h1_rows, h2_rows = [], []
    for scope in scopes:
        df = impact[impact["season_end_year"] == scope]
        h1_rows.append(median_diff(df, "trade_classification", "Preenche Lacuna", "Redundante"))
        h2_rows.append(median_diff(df, "objetivo_sazonal", "Contender", "Tanking"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300, sharey=True)
    y_pos = np.arange(len(scopes))[::-1]

    for ax, rows, title, sub in zip(
        axes, [h1_rows, h2_rows], ["Preenche Lacuna − Redundante", "Contender − Tanking"],
        ["mediana ΔWPA_time(Lacuna) − mediana ΔWPA_time(Redundante)",
         "mediana ΔWPA_time(Contender) − mediana ΔWPA_time(Tanking)"],
    ):
        ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=2)
        diffs = [r["diff"] for r in rows if r is not None]
        x_range = (max(diffs) - min(diffs)) if diffs else 0.1
        offset = max(x_range * 0.05, 0.006)
        for y, r in zip(y_pos, rows):
            if r is None:
                ax.text(0, y, "amostra insuficiente", ha='center', va='center', fontsize=9.5,
                        color=INK_MUTED, fontstyle='italic')
                continue
            color = BLUE if r["diff"] >= 0 else RED
            ax.scatter([r["diff"]], [y], s=130, color=color, zorder=4, edgecolor=INK_PRIMARY, linewidth=1.0)
            label = f"{r['diff']:+.3f}".replace(".", ",") + f"   (n={r['n1']} vs {r['n2']})"
            ax.text(r["diff"] + offset, y, label, ha='left', va='center', fontsize=9.5, color=INK_PRIMARY)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(row_labels, fontsize=11.5)
        ax.set_title(title, fontsize=13, fontweight='bold', color=INK_PRIMARY, pad=10)
        ax.set_xlabel(sub, fontsize=9.5, color=INK_SECONDARY)
        style_ax(ax)
        ax.grid(axis='x', zorder=0)
        ax.set_ylim(-0.7, len(scopes) - 0.3)

    fig.suptitle("Resumo Descritivo por Temporada — diferença de mediana de ΔWPA_time", fontsize=15.5,
                  fontweight='bold', y=1.06, color=INK_PRIMARY)
    fig.text(0.5, -0.04,
              "Apenas descritivo: diferença observada entre medianas, sem teste de significância estatística. "
              "azul = diferença positiva   •   vermelho = diferença negativa",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "resumo_descritivo_por_temporada.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


# -----------------------------------------------------------------------------
# F. Trocas por posicao x classificacao (lacuna/redundante), por temporada
# -----------------------------------------------------------------------------
def generate_position_breakdown_by_season():
    df = pd.read_csv(DATA_DIR / "wpa_positional_analysis_summary.csv")
    positions = ["Guards", "Forwards", "Centers"]
    cls_order = ["Preenche Lacuna", "Redundante"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), dpi=300, sharey=True)
    y_max = df.groupby(["escopo", "posicao"])["n_jogadores"].sum().max()

    for ax, season in zip(axes, SEASONS):
        sub = df[df["escopo"] == season]
        x = np.arange(len(positions))
        width = 0.35
        for i, cls in enumerate(cls_order):
            vals = [int(sub[(sub["posicao"] == p) & (sub["classificacao"] == cls)]["n_jogadores"].sum())
                    for p in positions]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, color=CLASS_COLORS[cls], edgecolor=INK_PRIMARY,
                           linewidth=0.7, zorder=3, label=cls)
            for bar, v in zip(bars, vals):
                if v == 0:
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, str(v), ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color=INK_PRIMARY)

        ax.set_xticks(x)
        ax.set_xticklabels(positions, fontsize=11)
        ax.set_title(SEASON_LABELS[season], fontsize=13, fontweight='bold', color=INK_PRIMARY, pad=8)
        ax.set_ylim(0, y_max * 1.2)
        style_ax(ax)

    axes[0].set_ylabel("Número de jogadores adquiridos", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9.5, frameon=False)
    fig.suptitle("Trocas por Posição: Preenche Lacuna vs. Redundante, por Temporada", fontsize=15.5,
                  fontweight='bold', y=1.04, color=INK_PRIMARY)
    fig.text(0.5, -0.03,
              "Contagem de jogadores adquiridos em trocas intra-temporada, por posição e classificação "
              "(δP pré-troca na posição do jogador adquirido).",
              ha='center', fontsize=9.5, color=INK_SECONDARY, fontstyle='italic')
    plt.tight_layout()
    out = VIZ_DIR / "trocas_por_posicao_por_temporada.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("Generating WPA per-season DESCRIPTIVE visualizations from real computed data (2019-2021)...")
    pos_df, impact_df = load_data()
    generate_positional_gaps_by_season(pos_df)
    generate_h1_by_season(impact_df)
    generate_h2_by_season(impact_df)
    generate_success_rate_by_season(impact_df)
    generate_position_breakdown_by_season()
    generate_effect_summary(impact_df)
    print("All visualizations generated in 'viz/' - every number above was computed from data/*.csv, none simulated.")
