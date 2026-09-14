#!/usr/bin/env python3
"""
Slide 2 of the clustering presentation: a simple "what data are we using"
table -- universe size, number of variables, and the variable groups that
make up team_season_features.csv. All counts computed live from the CSV
header (never hardcoded), so this stays correct if the feature set changes.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_DIR / "data" / "team_season_features.csv"
OUT_PNG = PROJECT_DIR / "viz" / "tabela_dados_clusterizacao.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
HEADER_BG = "#0b0b0b"
HEADER_TEXT = "#ffffff"
ROW_ALT = "#f2f1ee"
ACCENT = "#2a78d6"

META_COLS = ['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year']


def build():
    df = pd.read_csv(FEATURES_CSV)
    feats = [c for c in df.columns if c not in META_COLS]
    avg_cols = [c for c in feats if c.startswith('avg_')]
    std_cols = [c for c in feats if c.startswith('std_')]
    other_cols = [c for c in feats if c not in avg_cols and c not in std_cols]

    n_teams_seasons = len(df)
    year_min, year_max = int(df['season_end_year'].min()), int(df['season_end_year'].max())
    n_seasons = df['season_end_year'].nunique()
    n_teams = df['team_id'].nunique()

    rows = [
        ("Universo", f"{n_teams_seasons}", f"times-temporada = {n_teams} times × {n_seasons} temporadas ({year_min-1}-{str(year_min)[2:]} a {year_max-1}-{str(year_max)[2:]})"),
        ("Total de variáveis", f"{len(feats)}", "por time-temporada, calculadas a partir dos jogos da temporada regular"),
        ("Médias (avg_)", f"{len(avg_cols)}", "pontos, rebotes, assistências, %FG, %3P, %FT, ritmo (posses/jogo), ratings ofensivo/defensivo/net..."),
        ("Consistência (std_)", f"{len(std_cols)}", "desvio-padrão das mesmas estatísticas — o quão irregular o time é jogo a jogo"),
        ("Contexto e forma", f"{len(other_cols)}", "jogos disputados, tendência intra-temporada (net rating: 10 primeiros x 10 últimos jogos), desempenho em jogos \"clutch\" (decididos por ≤5 pts), net rating ajustado pela força do adversário"),
    ]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(15.5, 2.0 + n_rows * 1.15), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 2.3)

    ax.text(0.0, n_rows + 1.85, "Tabela — Dados e Variáveis Usadas na Clusterização",
            fontsize=17, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
    ax.text(0.0, n_rows + 1.35,
            f"Fonte: data/team_season_features.csv, gerado por src/build_team_features.py.",
            fontsize=10, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')

    headers = ["Categoria", "Quantidade", "O que inclui (exemplos)"]
    col_widths = [0.22, 0.14, 0.64]
    header_y = n_rows + 0.55
    x = 0.0
    for w, h in zip(col_widths, headers):
        ax.add_patch(plt.Rectangle((x, header_y - 0.45), w, 0.9, facecolor=HEADER_BG, edgecolor='none', zorder=2))
        ax.text(x + 0.012, header_y, h, fontsize=12, fontweight='bold', color=HEADER_TEXT, ha='left', va='center', zorder=3)
        x += w

    for i, (cat, qty, desc) in enumerate(rows):
        y = n_rows - i - 0.15
        band = ROW_ALT if i % 2 == 0 else SURFACE
        row_h = 1.05
        ax.add_patch(plt.Rectangle((0, y - row_h / 2), sum(col_widths), row_h, facecolor=band, edgecolor='none', zorder=1))
        x = 0.0
        ax.text(x + 0.012, y, cat, fontsize=12.5, fontweight='bold', color=INK_PRIMARY, ha='left', va='center', zorder=3)
        x += col_widths[0]
        ax.text(x + 0.012, y, qty, fontsize=15, fontweight='bold', color=ACCENT, ha='left', va='center', zorder=3)
        x += col_widths[1]
        import textwrap
        wrapped = "\n".join(textwrap.wrap(desc, width=78))
        ax.text(x + 0.012, y, wrapped, fontsize=10, color=INK_SECONDARY, ha='left', va='center', zorder=3, linespacing=1.4)

    total_h = header_y + 0.9
    ax.add_patch(plt.Rectangle((0, -0.6), sum(col_widths), total_h + 0.6, fill=False, edgecolor="#cfcfc7", linewidth=1.2, zorder=4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")


if __name__ == "__main__":
    build()
