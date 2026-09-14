#!/usr/bin/env python3
"""
Figura do Apêndice (Modelagem dimensional) do TCC.

Recorte em esquema estrela do schema `silver`: duas tabelas de dimensão
(dim_teams, dim_games) ligadas por chave estrangeira a uma tabela de fato
(fact_team_gamelogs), a mesma tríade usada como exemplo no texto do
apêndice e definida em src/sql/create_silver.sql. Não é o schema
completo (haveria mais dimensões e fatos), só o suficiente para ilustrar
o padrão dimensão/fato descrito na Subseção 2.7.2.

Mesma paleta e convenções visuais de generate_background_pipeline_diagram.py
(INK, INK_MUTED, LINE, Nimbus Sans), para consistência com as demais
figuras do trabalho.

Output: viz/apendice_modelo_dimensional.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "viz" / "apendice_modelo_dimensional.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
DIM_FILL = "#f2f4f7"
FACT_FILL = "#ffffff"
SURFACE = "#ffffff"
ACCENT = "#8a1f2b"
SANS = "Nimbus Sans"

DIM_TEAMS = {
    "title": "silver.dim_teams",
    "rows": ["team_id  PK", "team_abbreviation", "team_name", "team_city"],
}
DIM_GAMES = {
    "title": "silver.dim_games",
    "rows": ["game_id  PK", "game_date", "season_end_year", "game_type"],
}
FACT_GAMELOGS = {
    "title": "silver.fact_team_gamelogs",
    "rows": [
        "game_id      FK → dim_games",
        "team_id      FK → dim_teams",
        "opponent_team_id",
        "    FK → dim_teams",
        "pts, pts_allowed, off_rating,",
        "def_rating, net_rating, ...",
        "PK (game_id, team_id)",
    ],
}


def draw_table_box(ax, x, y0, w, h, title, rows, fill):
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=1.3, zorder=3))
    header_h = 0.52
    ax.add_patch(Rectangle((x, y0 + h - header_h), w, header_h,
                            facecolor=INK, edgecolor=LINE, linewidth=1.3, zorder=4))
    header_size = 12.5 if len(title) > 20 else 16
    ax.text(x + w / 2, y0 + h - header_h / 2, title, ha="center", va="center",
            fontsize=header_size, color="#ffffff", fontweight="bold", zorder=5,
            family="monospace")
    row_top = y0 + h - header_h - 0.14
    line_h = (h - header_h - 0.24) / max(len(rows), 1)
    for i, row in enumerate(rows):
        ry = row_top - i * line_h - line_h / 2
        color = ACCENT if row.strip().endswith(("PK", ")")) or "PK" in row else INK_MUTED
        size = 11.5 if len(row) > 24 else 13.0
        ax.text(x + 0.18, ry, row, ha="left", va="center", fontsize=size,
                color=color, zorder=5, family="monospace")


def arrow(ax, p0, p1, lw=1.3, mutation_scale=11):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.9)
    ax.axis("off")

    dim_w, dim_h = 3.75, 2.55
    fact_w, fact_h = 5.6, 3.9

    teams_x, teams_y0 = 0.5, 3.9
    games_x, games_y0 = 0.5, 0.55
    fact_x = 7.6
    fact_y0 = (6.9 - fact_h) / 2 - 0.15

    draw_table_box(ax, teams_x, teams_y0, dim_w, dim_h, DIM_TEAMS["title"], DIM_TEAMS["rows"], DIM_FILL)
    draw_table_box(ax, games_x, games_y0, dim_w, dim_h, DIM_GAMES["title"], DIM_GAMES["rows"], DIM_FILL)
    draw_table_box(ax, fact_x, fact_y0, fact_w, fact_h, FACT_GAMELOGS["title"], FACT_GAMELOGS["rows"], FACT_FILL)

    arrow(ax, (teams_x + dim_w + 0.05, teams_y0 + dim_h * 0.35), (fact_x - 0.05, fact_y0 + fact_h * 0.72))
    arrow(ax, (games_x + dim_w + 0.05, games_y0 + dim_h * 0.65), (fact_x - 0.05, fact_y0 + fact_h * 0.28))

    ax.text(teams_x + dim_w / 2, teams_y0 + dim_h + 0.28, "Dimensão", ha="center", va="center",
            fontsize=14, color=INK_MUTED, style="italic")
    ax.text(games_x + dim_w / 2, games_y0 + dim_h + 0.28, "Dimensão", ha="center", va="center",
            fontsize=14, color=INK_MUTED, style="italic")
    ax.text(fact_x + fact_w / 2, fact_y0 + fact_h + 0.28, "Fato", ha="center", va="center",
            fontsize=14, color=INK_MUTED, style="italic")

    fig.savefig(OUT_PATH, dpi=400, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {OUT_PATH}")


if __name__ == "__main__":
    main()
