#!/usr/bin/env python3
"""
Renders data/wpa_descriptive_summary.csv as a publication-quality table image
(viz/tabela_resultados_descritivos.png) - every value read straight from the CSV,
nothing hardcoded. Purely descriptive (n, medians, % positive) - no significance
tests or p-values, per the decision to not run inferential statistics on only
3 real seasons of data. Also renders a compact team-archetype transition summary
(viz/tabela_transicao_perfil.png) from wpa_trade_impact_analysis.csv, purely
descriptive (pre-trade vs post-trade cluster - see wpa_pretrade_clusters.py for
why this is never used as a causal classifier).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR  # noqa: E402

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
HEADER_BG = "#0b0b0b"
HEADER_TEXT = "#ffffff"
ROW_ALT = "#f2f1ee"
SIG_BG = "#eafaea"
SIG_TEXT = "#0ca30c"

DATA_DIR = PROJECT_DIR / "data"
VIZ_DIR = PROJECT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SEASON_LABELS = {"2019": "2018-19", "2020": "2019-20", "2021": "2020-21",
                 "Pooled 2019-2021": "2019-2021 (agregado)"}


def fmt_num(x, decimals=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}".replace(".", ",")


def generate_results_table():
    df = pd.read_csv(DATA_DIR / "wpa_descriptive_summary.csv")
    df["escopo_label"] = df["escopo"].astype(str).map(SEASON_LABELS)
    df["mediana_fmt"] = df["mediana_delta_WPA_time"].apply(fmt_num)
    df["media_fmt"] = df["media_delta_WPA_time"].apply(fmt_num)
    df["pct_fmt"] = df["pct_positivo"].apply(lambda x: "—" if pd.isna(x) else f"{x:.1f}%".replace(".", ","))
    df["rho_fmt"] = df["spearman_rho"].apply(lambda x: "—" if pd.isna(x) else fmt_num(x, 4))
    df["n_fmt"] = df["n"].apply(lambda x: f"n={int(x)}")

    columns = ["escopo_label", "eixo", "grupo", "n_fmt", "mediana_fmt", "media_fmt", "pct_fmt", "rho_fmt"]
    headers = ["Temporada", "Eixo", "Grupo", "n", "Mediana ΔWPA", "Média ΔWPA", "% positivo", "rho (|δP| x ΔWPA)"]
    col_widths = [0.11, 0.20, 0.16, 0.06, 0.11, 0.11, 0.10, 0.13]

    n_rows = len(df)
    row_h = 0.85
    fig_h = 1.6 + n_rows * row_h * 0.34
    fig, ax = plt.subplots(figsize=(16, fig_h), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 2)

    ax.text(0.0, n_rows + 1.55, "Tabela 1 — Resumo descritivo do impacto das trocas, por temporada",
            fontsize=15, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
    ax.text(0.0, n_rows + 1.05,
            "Fonte: data/wpa_descriptive_summary.csv. Puramente descritivo — sem teste de significância estatística "
            "(amostra de 3 temporadas reais). rho = Spearman, força de associação observada, não um teste de hipótese.",
            fontsize=9, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')

    header_y = n_rows + 0.35
    x = 0.0
    for w, h in zip(col_widths, headers):
        ax.add_patch(plt.Rectangle((x, header_y - 0.4), w, 0.8, facecolor=HEADER_BG, edgecolor='none', zorder=2))
        ax.text(x + 0.008, header_y, h, fontsize=9.5, fontweight='bold', color=HEADER_TEXT,
                ha='left', va='center', zorder=3)
        x += w

    prev_scope = None
    for i, (_, row) in enumerate(df.iterrows()):
        y = n_rows - i - 0.35
        band = ROW_ALT if i % 2 == 0 else SURFACE
        ax.add_patch(plt.Rectangle((0, y - 0.4), sum(col_widths), 0.8, facecolor=band, edgecolor='none', zorder=1))

        x = 0.0
        vals = [row[c] for c in columns]
        if vals[0] == prev_scope:
            vals[0] = ""
        else:
            prev_scope = vals[0]
        for w, v in zip(col_widths, vals):
            ax.text(x + 0.008, y, str(v), fontsize=9, color=INK_PRIMARY, ha='left', va='center', zorder=3)
            x += w

    ax.add_patch(plt.Rectangle((0, -0.45), sum(col_widths), header_y + 0.8, fill=False,
                                edgecolor=GRID, linewidth=1.2, zorder=4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.tight_layout()
    out = VIZ_DIR / "tabela_resultados_descritivos.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {out}")


def generate_transition_table():
    df = pd.read_csv(DATA_DIR / "wpa_trade_impact_analysis.csv")
    df = df.dropna(subset=["cluster_name", "cluster_name_post_trade"])
    df["mudou"] = df["cluster_name"] != df["cluster_name_post_trade"]

    summary = df.groupby("trade_classification").agg(
        n=("mudou", "size"),
        pct_mudou=("mudou", "mean"),
    ).reindex(["Preenche Lacuna", "Redundante"])
    overall_pct = df["mudou"].mean()

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)

    ax.text(0.0, 3.6, "Tabela 2 — Transição de arquétipo (perfil pré→pós-troca), descritiva",
            fontsize=14, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
    ax.text(0.0, 3.15,
            f"{int(df['mudou'].sum())} de {len(df)} eventos ({overall_pct*100:.1f}%) mudaram de arquétipo entre o "
            "cluster pré-troca e o pós-troca.",
            fontsize=10, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')

    headers = ["Classificação da troca", "n", "% que mudou de arquétipo pré→pós"]
    col_widths = [0.42, 0.15, 0.43]
    header_y = 2.55
    x = 0.0
    for w, h in zip(col_widths, headers):
        ax.add_patch(plt.Rectangle((x, header_y - 0.32), w, 0.64, facecolor=HEADER_BG, edgecolor='none', zorder=2))
        ax.text(x + 0.01, header_y, h, fontsize=10, fontweight='bold', color=HEADER_TEXT, ha='left', va='center', zorder=3)
        x += w

    for i, (label, row) in enumerate(summary.iterrows()):
        y = 1.85 - i * 0.75
        band = ROW_ALT if i % 2 == 0 else SURFACE
        ax.add_patch(plt.Rectangle((0, y - 0.32), sum(col_widths), 0.64, facecolor=band, edgecolor='none', zorder=1))
        x = 0.0
        vals = [label, f"n={int(row['n'])}", f"{row['pct_mudou']*100:.1f}%"]
        for w, v in zip(col_widths, vals):
            ax.text(x + 0.01, y, v, fontsize=10, color=INK_PRIMARY, ha='left', va='center', zorder=3)
            x += w

    ax.add_patch(plt.Rectangle((0, 0.3), sum(col_widths), 2.55, fill=False, edgecolor=GRID, linewidth=1.2, zorder=4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.tight_layout()
    out = VIZ_DIR / "tabela_transicao_perfil.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {out}")


def generate_trade_ledger():
    """One detailed table per season, listing every acquired player: team, data,
    posicao, deltaP PRE-trade (used to classify), deltaP POS-trade (descriptive -
    did the diagnosed gap at that position actually close?), classificacao, and the
    team-level outcome (delta_WPA_time, objetivo sazonal) for the trade event."""
    df_class = pd.read_csv(DATA_DIR / "wpa_trade_classification.csv")
    df_impact = pd.read_csv(DATA_DIR / "wpa_trade_impact_analysis.csv")

    df = df_class.merge(
        df_impact[["team_id", "season_end_year", "trade_date", "delta_WPA_time", "objetivo_sazonal"]],
        on=["team_id", "season_end_year", "trade_date"], how="left",
    )

    GAP_FECHADA_COLORS = {"Sim": SIG_TEXT, "Melhorou": "#8a7a00", "Nao": "#c0392b",
                           "N/A (nao era lacuna)": INK_MUTED, "Sem dado pos-troca": INK_MUTED}

    for season in sorted(df["season_end_year"].unique()):
        sub = df[df["season_end_year"] == season].copy()
        sub = sub.sort_values(["trade_date", "team_abbreviation", "posicao"])
        sub["sig"] = sub["preenche_lacuna"] == "Preenche Lacuna"

        columns = ["trade_date", "team_abbreviation", "player_name", "posicao", "deltaP_fmt",
                   "preenche_lacuna", "deltaP_post_fmt", "gap_fechada", "delta_WPA_time_fmt", "objetivo_sazonal"]
        headers = ["Data", "Time", "Jogador", "Posição", "δP pré", "Classificação",
                   "δP pós", "Lacuna fechou?", "ΔWPA_time (time)", "Objetivo Sazonal"]
        col_widths = [0.075, 0.05, 0.145, 0.08, 0.075, 0.135, 0.075, 0.125, 0.11, 0.105]

        sub["deltaP_fmt"] = sub["deltaP"].apply(lambda x: fmt_num(x, 4))
        sub["deltaP_post_fmt"] = sub["deltaP_post_trade"].apply(lambda x: fmt_num(x, 4))
        sub["delta_WPA_time_fmt"] = sub["delta_WPA_time"].apply(fmt_num)

        n_rows = len(sub)
        fig_h = 1.5 + n_rows * 0.30
        fig, ax = plt.subplots(figsize=(17, fig_h), dpi=300)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_rows + 2)

        season_label = SEASON_LABELS.get(str(season), str(season))
        ax.text(0.0, n_rows + 1.55,
                f"Tabela — Trocas Intra-Temporada Detalhadas, {season_label} (n={n_rows} jogadores adquiridos)",
                fontsize=14.5, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
        ax.text(0.0, n_rows + 1.05,
                "Fonte: data/wpa_trade_classification.csv + wpa_trade_impact_analysis.csv. "
                "δP < 0 = lacuna (Preenche Lacuna); δP ≥ 0 = Redundante.",
                fontsize=9, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')

        header_y = n_rows + 0.35
        x = 0.0
        for w, h in zip(col_widths, headers):
            ax.add_patch(plt.Rectangle((x, header_y - 0.4), w, 0.8, facecolor=HEADER_BG, edgecolor='none', zorder=2))
            ax.text(x + 0.008, header_y, h, fontsize=9, fontweight='bold', color=HEADER_TEXT,
                    ha='left', va='center', zorder=3)
            x += w

        for i, (_, row) in enumerate(sub.iterrows()):
            y = n_rows - i - 0.35
            band = ROW_ALT if i % 2 == 0 else SURFACE
            ax.add_patch(plt.Rectangle((0, y - 0.4), sum(col_widths), 0.8, facecolor=band, edgecolor='none', zorder=1))
            x = 0.0
            for w, c in zip(col_widths, columns):
                v = row[c]
                if c == "preenche_lacuna":
                    color = SIG_TEXT if row["sig"] else INK_MUTED
                    weight = 'bold' if row["sig"] else 'normal'
                elif c == "gap_fechada":
                    color = GAP_FECHADA_COLORS.get(v, INK_PRIMARY)
                    weight = 'bold' if v in ("Sim", "Nao") else 'normal'
                else:
                    color, weight = INK_PRIMARY, 'normal'
                ax.text(x + 0.008, y, str(v), fontsize=8.3, color=color, fontweight=weight, ha='left', va='center', zorder=3)
                x += w

        ax.add_patch(plt.Rectangle((0, -0.45), sum(col_widths), header_y + 0.8, fill=False,
                                    edgecolor=GRID, linewidth=1.2, zorder=4))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        plt.tight_layout()
        out = VIZ_DIR / f"tabela_trocas_detalhada_{season}.png"
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
        plt.close()
        print(f"[OK] {out}")


if __name__ == "__main__":
    generate_results_table()
    generate_transition_table()
    generate_trade_ledger()
