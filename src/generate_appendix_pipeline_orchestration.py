#!/usr/bin/env python3
"""
Figuras do Apêndice (Orquestração do pipeline) do TCC.

O fluxograma completo do pipeline analítico (Makefile) não cabia,
legível, numa única figura de página vertical, e a alternativa de
página em paisagem foi descartada. Por isso ele é dividido em DUAS
figuras verticais, cada uma no mesmo tamanho de página do resto do
documento:

  1) Arquétipos competitivos + Hipótese 1 (desigualdade salarial).
  2) Hipóteses 2-4 (WPA / Contribuição Relativa) — parte da mesma carga
     inicial e da clusterização de arquétipos (já mostrada na figura
     anterior), por isso essas duas entradas aparecem aqui como caixas
     pequenas de referência, não repetindo o conteúdo da primeira figura.

Tamanho de caixa e de fonte NÃO são mais estimados por contagem de
caracteres (isso vinha subestimando a largura real do texto em negrito/
monoespaçado e causando texto vazando da caixa). Em vez disso, cada
texto é MEDIDO de verdade (matplotlib.textpath.TextPath, com a mesma
fonte/peso/estilo usados no desenho) antes de decidir quebra de linha
ou redução de fonte — ver `measure_pt` / `fit_lines`.

Output: viz/apendice_orquestracao_pipeline_1.png
        viz/apendice_orquestracao_pipeline_2.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT1 = PROJECT_DIR / "viz" / "apendice_orquestracao_pipeline_1.png"
OUT2 = PROJECT_DIR / "viz" / "apendice_orquestracao_pipeline_2.png"

INK = "#15181c"
INK_MUTED = "#5a6472"
LINE = "#1a1a1a"
FILL = "#f2f4f7"
ROOT_FILL = "#15181c"
SURFACE = "#ffffff"
SANS = "Nimbus Sans"
MONO = "monospace"

BOX_H = 1.55
MIN_BOX_W, MAX_BOX_W = 2.1, 3.3
TARGET_SIZE = 15.0
TARGET_MIN_SIZE = 10.5
SCRIPT_SIZE = 12.0
EXTRA_SIZE = 10.0
LANE_LABEL_SIZE = 18.0
PAD_IN = 0.26  # margem horizontal mínima dentro da caixa, cada lado


MEASURE_SAFETY = 1.10  # folga sobre a medição (TextPath subestima um pouco o avanço real)


def measure_pt(s, fontsize, family, weight="normal", style="normal"):
    """Largura real do texto renderizado, em pontos (1/72 pol.), com folga de segurança."""
    if not s:
        return 0.0
    fp = FontProperties(family=family, weight=weight, style=style)
    tp = TextPath((0, 0), s, size=fontsize, prop=fp)
    return tp.get_extents().width * MEASURE_SAFETY


def _greedy_wrap(chunks, fontsize, family, weight, style, max_width_pt):
    lines, cur = [], ""
    for c in chunks:
        trial = cur + c
        if cur and measure_pt(trial, fontsize, family, weight, style) > max_width_pt:
            lines.append(cur)
            cur = c
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def fit_lines(s, fontsize, family, weight, style, max_width_pt):
    """Quebra `s` em linhas que caibam em max_width_pt, usando a largura REAL
    do texto renderizado (não contagem de caracteres). Prefere quebrar em
    fronteiras de '_'/'.'/'-' (mantidas no fim de cada trecho); só desce a
    quebra por caractere se nem isso couber."""
    if measure_pt(s, fontsize, family, weight, style) <= max_width_pt:
        return [s]
    if any(sep in s for sep in "_.- "):
        chunks, cur = [], ""
        for ch in s:
            cur += ch
            if ch in "_.- ":
                chunks.append(cur)
                cur = ""
        if cur:
            chunks.append(cur)
        lines = _greedy_wrap(chunks, fontsize, family, weight, style, max_width_pt)
        if all(measure_pt(l, fontsize, family, weight, style) <= max_width_pt for l in lines):
            return lines
    return _greedy_wrap(list(s), fontsize, family, weight, style, max_width_pt)


def target_fit(target, box_w_in):
    """Linhas + tamanho de fonte do rótulo em negrito, garantidos a caber em box_w_in.
    Prioriza reduzir a fonte (até TARGET_MIN_SIZE) antes de quebrar linha; só
    quebra se nem no tamanho mínimo o texto inteiro couber numa linha."""
    max_width_pt = (box_w_in - 2 * PAD_IN) * 72
    if "\n" in target:
        lines = target.split("\n")
        size = TARGET_SIZE
        while size > TARGET_MIN_SIZE and max(
                measure_pt(l, size, MONO, "bold", "normal") for l in lines) > max_width_pt:
            size -= 0.5
        return lines, size

    size = TARGET_SIZE
    while size > TARGET_MIN_SIZE:
        if measure_pt(target, size, MONO, "bold", "normal") <= max_width_pt:
            return [target], size
        size -= 0.5

    lines = fit_lines(target, TARGET_MIN_SIZE, MONO, "bold", "normal", max_width_pt)
    size = TARGET_MIN_SIZE
    while size > 8 and max(
            measure_pt(l, size, MONO, "bold", "normal") for l in lines) > max_width_pt:
        size -= 0.5
        lines = fit_lines(target, size, MONO, "bold", "normal", max_width_pt)
    return lines, size


def script_fit(script, box_w_in, size=SCRIPT_SIZE):
    max_width_pt = (box_w_in - 2 * PAD_IN) * 72
    return fit_lines(script, size, SANS, "normal", "italic", max_width_pt)


def natural_box_w(target, script):
    """Largura da caixa ajustada ao conteúdo: cabe o rótulo e o script numa
    linha só, sem estourar MAX_BOX_W (target/script mais longos que isso
    quebram linha ou reduzem fonte dentro de node(), como antes)."""
    if "\n" in target:
        t_w = max(measure_pt(l, TARGET_SIZE, MONO, "bold", "normal") for l in target.split("\n"))
    else:
        t_w = measure_pt(target, TARGET_SIZE, MONO, "bold", "normal")
    s_w = measure_pt(script, SCRIPT_SIZE, SANS, "normal", "italic")
    w_in = max(t_w, s_w) / 72 + 2 * PAD_IN
    return min(max(w_in, MIN_BOX_W), MAX_BOX_W)


def node(ax, cx, cy, target, script, fill=FILL, text_color=INK, w=None, h=BOX_H, extra=None):
    if w is None:
        w = natural_box_w(target, script)
    x, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=1.3, zorder=3))

    target_lines, target_size = target_fit(target, w)
    script_lines = script_fit(script, w)

    line_gap_t = (target_size / 72) * 1.55
    line_gap_s = (SCRIPT_SIZE / 72) * 1.55
    block_h = len(target_lines) * line_gap_t + len(script_lines) * line_gap_s + (0.20 if extra else 0)
    top_y = cy + block_h / 2

    ty = top_y
    for line in target_lines:
        ty -= line_gap_t * 0.72
        ax.text(cx, ty, line, ha="center", va="center", fontsize=target_size,
                color=text_color, fontweight="bold", family=MONO, zorder=5)
        ty -= line_gap_t * 0.28

    for line in script_lines:
        ty -= line_gap_s * 0.72
        ax.text(cx, ty, line, ha="center", va="center", fontsize=SCRIPT_SIZE,
                color=(INK_MUTED if fill != ROOT_FILL else "#c9cdd3"), style="italic", zorder=5)
        ty -= line_gap_s * 0.28

    if extra:
        ty -= 0.20
        max_width_pt = (w - 2 * PAD_IN) * 72
        esize = EXTRA_SIZE
        while measure_pt(extra, esize, SANS, "normal", "italic") > max_width_pt and esize > 6:
            esize -= 0.5
        ax.text(cx, ty, extra, ha="center", va="center", fontsize=esize,
                color="#c9cdd3", style="italic", zorder=5)

    return (x, y0, w, h)


def small_node(ax, cx, cy, label, w=1.9, h=0.9, fill=FILL, text_color=INK):
    """Caixa pequena de referência (entrada já mostrada em outra figura)."""
    x, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=fill, edgecolor=LINE,
                            linewidth=1.1, zorder=3, linestyle=(0, (4, 2))))
    size = 13.0
    max_width_pt = (w - 2 * PAD_IN) * 72
    while measure_pt(label, size, MONO, "bold", "normal") > max_width_pt and size > 8:
        size -= 0.5
    ax.text(cx, cy, label, ha="center", va="center", fontsize=size,
            color=text_color, fontweight="bold", family=MONO, zorder=5)
    return (x, y0, w, h)


def arrow(ax, p0, p1, lw=1.4, mutation_scale=13, **kw):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, linewidth=lw,
        color=LINE, connectionstyle=kw.pop("connectionstyle", "arc3,rad=0"),
        zorder=2, shrinkA=0, shrinkB=0, **kw))


def right(box):
    x, y0, w, h = box
    return (x + w, y0 + h / 2)


def left(box):
    x, y0, w, h = box
    return (x, y0 + h / 2)


def top(box):
    x, y0, w, h = box
    return (x + w / 2, y0 + h)


def bottom(box):
    x, y0, w, h = box
    return (x + w / 2, y0)


def figure_1() -> None:
    """Arquétipos competitivos + Hipótese 1 (desigualdade salarial)."""
    fig, ax = plt.subplots(figsize=(17.0, 9.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 9.8)
    ax.axis("off")

    LANE_A, LANE_B = 7.85, 3.15
    COL0, COL1, COL2, COL3, COL4 = 1.15, 4.0, 7.6, 11.2, 14.8
    STACK_B = 1.95

    root = node(ax, COL0, (LANE_A + LANE_B) / 2, "db-load", "db_pipeline.py",
                fill=ROOT_FILL, text_color="#ffffff", w=1.9, h=1.6,
                extra="Bronze→Silver→Ouro")

    ax.text(0.15, 9.35, "Arquétipos competitivos", fontsize=LANE_LABEL_SIZE,
            color=INK, fontweight="bold", ha="left", va="center")
    a1 = node(ax, COL1, LANE_A, "team-features", "build_team_features.py")
    a2 = node(ax, COL2, LANE_A, "clusters", "fit_clusters.py")
    a3 = node(ax, COL3, LANE_A, "k-selection", "generate_k_selection_table.py")
    a4 = node(ax, COL4, LANE_A, "k4-justification", "generate_k4_justification.py")
    arrow(ax, right(root), left(a1))
    arrow(ax, right(a1), left(a2))
    arrow(ax, right(a2), left(a3))
    arrow(ax, right(a3), left(a4))

    ax.text(0.15, 0.4, "H1 — desigualdade salarial", fontsize=LANE_LABEL_SIZE,
            color=INK, fontweight="bold", ha="left", va="center")
    b1 = node(ax, COL1, LANE_B, "trade-activity", "build_trade_activity.py")
    b2 = node(ax, COL2, LANE_B + STACK_B, "metric-selection", "select_inequality_metric.py")
    b3 = node(ax, COL2, LANE_B, "window-robustness", "validate_trade_window_robustness.py")
    b4 = node(ax, COL2, LANE_B - STACK_B, "salary-inequality-\nanalysis", "analyze_salary_inequality_trades.py")
    arrow(ax, right(root), left(b1))
    arrow(ax, right(b1), left(b2))
    arrow(ax, right(b1), left(b3))
    arrow(ax, right(b1), left(b4))

    fig.savefig(OUT1, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {OUT1}")


def figure_2() -> None:
    """Hipóteses 2-4 (WPA / Contribuição Relativa)."""
    fig, ax = plt.subplots(figsize=(12.5, 8.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    TITLE_Y = 7.4
    ROOT_Y = 6.35
    C1_Y = 4.75
    LEAF_Y = 2.85
    ROW3_Y = 1.0
    COL_L, COL_C, COL_R = 2.3, 6.15, 10.0

    ax.text(0.15, TITLE_Y, "Hipóteses 2-4 — WPA e Contribuição Relativa",
            fontsize=LANE_LABEL_SIZE, color=INK, fontweight="bold", ha="left", va="center")

    db = small_node(ax, 2.3, ROOT_Y, "db-load", fill=ROOT_FILL, text_color="#ffffff")
    cl = small_node(ax, 5.6, ROOT_Y, "clusters")

    c1 = node(ax, COL_C, C1_Y, "wpa-pretrade-\nclusters", "build_wpa_pretrade_clusters.py")
    arrow(ax, bottom(db), top(c1), connectionstyle="arc3,rad=-0.15")
    arrow(ax, bottom(cl), top(c1), connectionstyle="arc3,rad=0.15")

    c2 = node(ax, COL_L, LEAF_Y, "wpa-trade-analysis", "build_wpa_trade_analysis.py")
    c3 = node(ax, COL_R, LEAF_Y, "wpa-trade-analysis-\nextended", "build_wpa_trade_analysis_extended.py")
    arrow(ax, left(c1), top(c2), connectionstyle="arc3,rad=-0.2")
    arrow(ax, right(c1), top(c3), connectionstyle="arc3,rad=0.2")

    c4 = node(ax, COL_L, ROW3_Y, "wpa-descriptive", "describe_wpa_results.py")
    c5 = node(ax, COL_R, ROW3_Y, "wpa-hypothesis-\ntests", "test_wpa_hypotheses_extended.py")
    arrow(ax, bottom(c2), top(c4))
    arrow(ax, bottom(c3), top(c5))

    fig.savefig(OUT2, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {OUT2}")


if __name__ == "__main__":
    plt.rcParams["font.family"] = SANS
    figure_1()
    figure_2()
