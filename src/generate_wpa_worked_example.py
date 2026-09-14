#!/usr/bin/env python3
"""
Step-by-step worked example of the full WPA -> RCP -> deltaP -> classification ->
delta_WPA_time -> objetivo sazonal pipeline, using ONE real trade: James Harden to
the Brooklyn Nets, 2021-01-14 (season_end_year=2021). Chosen because it's a
recognizable trade with a clean single-player acquisition and a large, real effect.

Every number in this figure is re-derived live from data/espn/player_box.parquet
and the DB (not copied from the already-computed CSVs), so it doubles as an
independent sanity check of the pipeline.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import (  # noqa: E402
    PROJECT_DIR, get_engine, load_espn_player_box, assign_positions, compute_wpa_acc,
    team_games_asof, classify_objective,
)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
GOOD = "#0ca30c"

VIZ_DIR = PROJECT_DIR / "viz"
TEAM_ID = 1610612751  # BKN
SEASON = 2021
TRADE_DATE = pd.Timestamp("2021-01-14")
SEASON_START = pd.Timestamp("2020-12-22")

FIG_W_IN = 11.5
LEFT_MARGIN_IN, RIGHT_MARGIN_IN = 0.6, 0.6
BOX_W_IN = FIG_W_IN - LEFT_MARGIN_IN - RIGHT_MARGIN_IN
TITLE_H_IN = 0.34
LINE_H_IN = 0.27
BOX_PAD_TOP_IN = 0.12
BOX_PAD_BOTTOM_IN = 0.12
BOX_GAP_IN = 0.20
HEADER_H_IN = 1.05
FOOTER_H_IN = 0.5


def gather_numbers():
    engine = get_engine()
    pb = load_espn_player_box(engine)
    season_pb = pb[pb["season_end_year"] == SEASON]
    bkn_pb = season_pb[season_pb["team_id"] == TEAM_ID]

    positions = assign_positions(bkn_pb, cutoff_date=TRADE_DATE)
    wpa_acc = compute_wpa_acc(bkn_pb, cutoff_date=TRADE_DATE, start_date=SEASON_START)
    merged = wpa_acc.merge(positions, on=["player_id", "team_id", "season_end_year"])
    names = pb[["player_id", "name"]].drop_duplicates("player_id").set_index("player_id")["name"]
    merged["name"] = merged["player_id"].map(names)

    wpa_time_team = float(merged["WPA_acc"].sum())
    guards = merged[merged["posicao"] == "Guards"].sort_values("WPA_acc", ascending=False)
    wpa_pos_guards = float(guards["WPA_acc"].sum())
    rcp_pos = wpa_pos_guards / wpa_time_team

    pre = bkn_pb[bkn_pb["game_date"] < TRADE_DATE]
    post = bkn_pb[bkn_pb["game_date"] >= TRADE_DATE]
    games_pre, games_post = pre["game_id"].nunique(), post["game_id"].nunique()
    rcp_pos_bar = rcp_pos / games_pre  # RCP_bar_P = RCP_P / N (paper section III-C)

    from build_wpa_trade_analysis import asof_league_snapshot  # noqa: E402
    snap = asof_league_snapshot(pb, SEASON, TRADE_DATE, SEASON_START)
    muP = snap[snap["stable"]].groupby("posicao")["RCP_pos_bar"].mean()["Guards"]
    deltaP = rcp_pos_bar - muP

    rate_pre = pre["tWPA"].sum() / games_pre
    rate_post = post["tWPA"].sum() / games_post
    delta_wpa_time = rate_post - rate_pre

    standings = team_games_asof(engine, SEASON, TRADE_DATE)
    objective = classify_objective(TEAM_ID, SEASON, TRADE_DATE, standings)

    return {
        "guards_roster": guards[["name", "games", "WPA_acc"]].to_dict("records"),
        "wpa_pos_guards": wpa_pos_guards,
        "wpa_time_team": wpa_time_team,
        "rcp_pos": rcp_pos,
        "rcp_pos_bar": rcp_pos_bar,
        "muP": float(muP),
        "deltaP": float(deltaP),
        "games_pre": int(games_pre), "games_post": int(games_post),
        "rate_pre": float(rate_pre), "rate_post": float(rate_post),
        "delta_wpa_time": float(delta_wpa_time),
        "conf_rank": objective["conf_rank"], "win_pct": objective["win_pct_asof"],
        "objetivo": objective["objetivo_sazonal"],
    }


def fmt(x, d=3):
    return f"{x:+.{d}f}".replace(".", ",")


def build_steps(n):
    guards_top3 = ", ".join(f"{g['name']} ({fmt(g['WPA_acc'])})" for g in n["guards_roster"][:3])
    steps = [
        ("1. WPA por jogador (ESPN oWPA/dWPA/tWPA), elenco do BKN antes de 14/01/2021", [
            f"Ex.: {guards_top3}, ...",
            f"({len(n['guards_roster'])} guards no elenco pré-troca, jogos de 22/12/2020 a 13/01/2021)",
        ], BLUE),
        ("2. Agregar WPA_acc por grupo posicional e para o time inteiro", [
            f"WPA_pos(Guards) = soma do WPA_acc de todos os guards acima = {fmt(n['wpa_pos_guards'])}",
            f"WPA_acc(time) = soma de TODAS as posições (Guards + Forwards + Centers) = {fmt(n['wpa_time_team'])}",
        ], BLUE),
        ("3. RCP_P = WPA_P / WPA_acc(time)  [paper, secao III-C]", [
            f"RCP_P(Guards, BKN) = {fmt(n['wpa_pos_guards'])} / {fmt(n['wpa_time_team'])} = {fmt(n['rcp_pos'])}",
            "Negativo aqui não é porque os guards jogaram mal (WPA_P foi positivo) — é porque o",
            "WPA_acc do TIME inteiro estava negativo nesse trecho; dividir por um total negativo",
            "inverte o sinal da razão. É uma propriedade do próprio formato da métrica.",
        ], AQUA),
        ("4. RCP_bar_P = RCP_P / N  (N = jogos do time na janela pré-troca)  [III-C]", [
            f"RCP_bar_P(Guards, BKN) = {fmt(n['rcp_pos'])} / {n['games_pre']} jogos = {fmt(n['rcp_pos_bar'], 5)}",
        ], AQUA),
        ("5. muP = média do RCP_bar_P(Guards) das equipes da liga, na mesma data (14/01/2021)  [III-D]", [
            f"muP(Guards, 14/01/2021) = {fmt(n['muP'], 5)}",
            "(média sobre as equipes com WPA_acc estável nessa mesma data-corte, pré-troca para todas)",
        ], AQUA),
        ("6. deltaP = RCP_bar_P − muP  [III-D]", [
            f"deltaP(Guards, BKN) = {fmt(n['rcp_pos_bar'], 5)} − ({fmt(n['muP'], 5)}) = {fmt(n['deltaP'], 5)}",
            "deltaP < 0  →  LACUNA estrutural na posição Guards, diagnosticada ANTES da troca",
        ], RED),
        ("7. Classificar a troca", [
            "James Harden joga na posição Guards (rótulo do log da troca)",
            f"Guards tinha deltaP < 0 nesse instante  →  troca classificada \"Preenche Lacuna\"",
        ], AQUA),
        ("8. Impacto: delta_WPA_time (taxa de WPA do time, pós − pré)", [
            f"Taxa pré  = WPA_acc(time, {n['games_pre']} jogos antes) / {n['games_pre']} jogos = {fmt(n['rate_pre'])}",
            f"Taxa pós  = WPA_acc(time, {n['games_post']} jogos depois) / {n['games_post']} jogos = {fmt(n['rate_post'])}",
            f"delta_WPA_time = {fmt(n['rate_post'])} − ({fmt(n['rate_pre'])}) = {fmt(n['delta_wpa_time'])}",
        ], BLUE),
        ("9. Objetivo sazonal — rank de conferência pré-troca (extensão própria, fora do paper original)", [
            f"BKN: {n['conf_rank']}º no Leste, win_pct pré-troca = {n['win_pct']:.3f}  →  \"{n['objetivo']}\"",
            "(rank 1-6 = Contender | 7-10 = Intermediário | 11-15 = Tanking)",
        ], BLUE),
        ("Resultado desta observação", [
            f"Troca classificada \"Preenche Lacuna\", feita por um \"{n['objetivo']}\", com delta_WPA_time = {fmt(n['delta_wpa_time'])}",
            "Este é o resultado da metodologia aplicada a UMA troca específica — descritivo, não um teste de hipótese.",
            "O comportamento agregado dessas variáveis nas 3 temporadas está nas tabelas e gráficos descritivos.",
        ], GOOD),
    ]
    return steps


def generate():
    n = gather_numbers()
    steps = build_steps(n)

    # 1st pass: compute each box's height from its content, and the total figure height
    box_heights = []
    for title, lines, color in steps:
        h = BOX_PAD_TOP_IN + TITLE_H_IN + len(lines) * LINE_H_IN + BOX_PAD_BOTTOM_IN
        box_heights.append(h)

    total_h_in = HEADER_H_IN + sum(box_heights) + BOX_GAP_IN * (len(steps) - 1) + FOOTER_H_IN

    fig = plt.figure(figsize=(FIG_W_IN, total_h_in), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, FIG_W_IN)
    ax.set_ylim(0, total_h_in)

    # Header (figure coords, from the top)
    y_top = total_h_in
    ax.text(FIG_W_IN / 2, y_top - 0.42, "Exemplo passo a passo da metodologia",
            fontsize=19, fontweight='bold', ha='center', va='top', color=INK_PRIMARY)
    ax.text(FIG_W_IN / 2, y_top - 0.78, "Troca real: James Harden → Brooklyn Nets, 14/01/2021 (temporada 2020-21)",
            fontsize=12, ha='center', va='top', color=INK_SECONDARY, style='italic')

    # Boxes, top to bottom
    cursor_y = y_top - HEADER_H_IN
    for (title, lines, color), h in zip(steps, box_heights):
        box_top = cursor_y
        box_bottom = cursor_y - h
        box_left = LEFT_MARGIN_IN

        fig_box = patches.FancyBboxPatch(
            (box_left, box_bottom), BOX_W_IN, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.08, zorder=3,
        )
        ax.add_patch(fig_box)
        border = patches.FancyBboxPatch(
            (box_left, box_bottom), BOX_W_IN, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.8, edgecolor=color, facecolor="none", zorder=4,
        )
        ax.add_patch(border)

        text_x = box_left + 0.22
        ax.text(text_x, box_top - BOX_PAD_TOP_IN, title, fontsize=12.5, fontweight='bold',
                color=INK_PRIMARY, va='top', ha='left', zorder=5)
        for i, line in enumerate(lines):
            ax.text(text_x, box_top - BOX_PAD_TOP_IN - TITLE_H_IN - i * LINE_H_IN, line,
                    fontsize=10.3, color=INK_SECONDARY, va='top', ha='left', zorder=5, family='monospace')

        cursor_y = box_bottom - BOX_GAP_IN

    ax.text(FIG_W_IN / 2, FOOTER_H_IN * 0.55,
            "Todos os números foram recalculados diretamente de data/espn/player_box.parquet e do banco "
            "para esta figura — não copiados dos CSVs já processados.",
            ha='center', va='center', fontsize=8.8, color=INK_SECONDARY, style='italic')

    out = VIZ_DIR / "exemplo_passo_a_passo.png"
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close()
    print(f"[OK] {out}")
    for k, v in n.items():
        if k != "guards_roster":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    generate()
