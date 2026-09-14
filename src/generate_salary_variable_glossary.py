#!/usr/bin/env python3
"""
Glossary table for the 5 variables in viz/salary_inequality_correlation_matrix.png
and viz/salary_inequality_core_scatter.png (analyze_salary_inequality_trades.py).

Purely presentational -- no statistics computed here, just documents what each
variable IS, how it's built, and how to read its sign, so the correlation
matrix doesn't require re-reading three scripts' docstrings to interpret.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PNG = PROJECT_DIR / "viz" / "salary_inequality_variable_glossary.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
HEADER_BG = "#0b0b0b"
HEADER_TEXT = "#ffffff"
ROW_ALT = "#f2f1ee"

ROWS = [
    ("Gini (nível pré)", "gini_pre",
     "Gini Ponderado por Minutos do elenco, medido só na janela ANTES do trade deadline real da temporada.",
     "Nível: quão desigual já era a folha em quadra antes de qualquer troca. Não é uma variação."),
    ("Δ Gini", "delta_gini",
     "gini_pos − gini_pre: variação do Gini Ponderado por Minutos entre a janela pós e a janela pré-deadline.",
     "> 0 = desigualdade salarial em quadra aumentou depois do deadline. < 0 = diminuiu."),
    ("Magnitude da troca", "trade_magnitude",
     "Número de jogadores movimentados (entradas + saídas) pelo time na janela de 30 dias antes do deadline.",
     "Maior valor = reformulação de elenco mais intensa, independente da direção do Δ Gini."),
    ("Δ Performance (ajustada)", "delta_adjusted_perf",
     "Variação do Net Rating ponderado pela força do adversário (SRS), pós menos pré-deadline (adjusted_performance.py).",
     "> 0 = time melhorou o desempenho, controlando a dificuldade do calendário em cada janela. Métrica principal de resultado."),
    ("Δ Net Rating (bruto)", "delta_raw_net_rating",
     "Mesma comparação pós − pré, mas com o Net Rating simples (sem ajuste por força de adversário).",
     "Variável de robustez: se contar história parecida com a Δ Performance ajustada, o ajuste de calendário não estava dirigindo o resultado."),
]

COL2_WRAP = 46
COL3_WRAP = 38
LINE_H = 0.30
ROW_PAD = 0.28


def wrapped(text, width):
    return textwrap.wrap(text, width=width)


def build():
    # Pre-compute wrapped lines and each row's height
    row_data = []
    for label, code, definition, sign in ROWS:
        def_lines = wrapped(definition, COL2_WRAP)
        sign_lines = wrapped(sign, COL3_WRAP)
        n_lines = max(len(def_lines), len(sign_lines), 2)
        row_h = ROW_PAD * 2 + n_lines * LINE_H
        row_data.append((label, code, def_lines, sign_lines, row_h))

    total_rows_h = sum(r[4] for r in row_data)
    header_h = 0.7
    title_h = 1.5
    fig_h_units = title_h + header_h + total_rows_h + 0.3
    fig, ax = plt.subplots(figsize=(16.5, fig_h_units * 0.62), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h_units)

    top = fig_h_units
    ax.text(0.0, top - 0.28, "Glossário — Variáveis da Matriz de Correlação de Desigualdade Salarial x Trocas",
            fontsize=15, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
    ax.text(0.0, top - 0.75,
            "Cada linha corresponde a uma variável em viz/salary_inequality_correlation_matrix.png e\n"
            "viz/salary_inequality_core_scatter.png. Fonte: src/analyze_salary_inequality_trades.py.",
            fontsize=9.5, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic', linespacing=1.4)

    table_top = top - title_h
    headers = ["Variável", "O que é / como é calculada", "Como interpretar o sinal"]
    col_widths = [0.16, 0.44, 0.40]
    ax.add_patch(plt.Rectangle((0, table_top - header_h), sum(col_widths), header_h,
                                facecolor=HEADER_BG, edgecolor='none', zorder=2))
    x = 0.0
    for w, h in zip(col_widths, headers):
        ax.text(x + 0.012, table_top - header_h / 2, h, fontsize=10.5, fontweight='bold', color=HEADER_TEXT,
                ha='left', va='center', zorder=3)
        x += w

    cursor_y = table_top - header_h
    for i, (label, code, def_lines, sign_lines, row_h) in enumerate(row_data):
        band = ROW_ALT if i % 2 == 0 else SURFACE
        ax.add_patch(plt.Rectangle((0, cursor_y - row_h), sum(col_widths), row_h,
                                    facecolor=band, edgecolor='none', zorder=1))

        row_top = cursor_y
        x = 0.0
        ax.text(x + 0.012, row_top - ROW_PAD - 0.05, label, fontsize=10.8, fontweight='bold',
                color=INK_PRIMARY, ha='left', va='top', zorder=3)
        ax.text(x + 0.012, row_top - ROW_PAD - 0.42, code, fontsize=7.8, color=INK_SECONDARY,
                family='monospace', ha='left', va='top', zorder=3)
        x += col_widths[0]

        for li, line in enumerate(def_lines):
            ax.text(x + 0.012, row_top - ROW_PAD - li * LINE_H, line, fontsize=9.3, color=INK_PRIMARY,
                    ha='left', va='top', zorder=3)
        x += col_widths[1]

        for li, line in enumerate(sign_lines):
            ax.text(x + 0.012, row_top - ROW_PAD - li * LINE_H, line, fontsize=9.0, color=INK_SECONDARY,
                    ha='left', va='top', zorder=3, fontstyle='italic')

        cursor_y -= row_h

    ax.add_patch(plt.Rectangle((0, cursor_y), sum(col_widths), table_top - cursor_y, fill=False,
                                edgecolor="#cfcfc7", linewidth=1.2, zorder=4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")


if __name__ == "__main__":
    build()
