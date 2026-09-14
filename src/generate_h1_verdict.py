#!/usr/bin/env python3
"""
Veredito da Hipótese 1 -- Desigualdade Salarial x Performance
Author: Data Visualization Specialist

Gera UMA figura de síntese cujo único objetivo é responder à Hipótese 1: qual
das duas teorias concorrentes -- Equidade ou Torneio -- encontra suporte nos
dados, ou se o resultado é estatisticamente indiferente (nenhuma das duas).

Nada aqui é hardcoded: o veredito (texto, cor do marcador, largura da zona
"sem efeito") é recalculado a partir dos dados a cada execução, então a figura
continua correta se a amostra crescer ou o resultado mudar de sinal.

Insumo: data/salary_inequality_trades_analysis.csv (a mesma base usada em
analyze_salary_inequality_trades.py e generate_methodology_explainer_figures.py
-- nenhuma estatística nova é computada além do que este script mesmo deriva
do teste de Spearman já reportado nessas duas fontes).

Método (tudo derivado da amostra, nada arbitrário):
  - rho, p: Spearman entre delta_gini e delta_adjusted_perf (igual ao teste
    central já publicado).
  - r_critico: |rho| mínimo para p<0,05 nesta amostra, via aproximação t
    (mesma que o scipy usa internamente para o p-valor do Spearman em N
    grande) -- define a largura da faixa cinza "sem efeito estatístico".
  - IC 95%: via transformação z de Fisher.
  - Magnitude do efeito: convenção de Cohen (1988) sobre |rho|.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALPHA = 0.05

# Par divergente (polos que leem como opostos + neutro cinza no meio) --
# reaproveitado aqui em vez do par pastel vermelho/verde do
# salary_quadrant_analysis.png porque esta figura precisa de um cinza que
# leia genuinamente como "nada" (indiferente), não apenas mais um quadrante.
COLOR_EQUIDADE = '#2a78d6'        # polo azul
COLOR_TORNEIO = '#e34948'         # polo vermelho
COLOR_NEUTRAL = '#6b6a66'         # marcador/texto para o caso "sem efeito"
COLOR_ZONE_FILL = '#e9e8e3'       # preenchimento da faixa de não-significância
COLOR_ZONE_EDGE = '#c3c2b7'
COLOR_TINT_EQUIDADE = '#dce9f9'   # tinta de fundo bem clara
COLOR_TINT_TORNEIO = '#fbdedd'
INK_PRIMARY = '#1a1a1a'
INK_SECONDARY = '#52514e'


def effect_size_label(abs_rho: float) -> str:
    """Convenção de Cohen (1988) para magnitude de correlação."""
    if abs_rho < 0.10:
        return "desprezível"
    if abs_rho < 0.30:
        return "fraco"
    if abs_rho < 0.50:
        return "moderado"
    return "forte"


def critical_rho(n: int, alpha: float = ALPHA) -> float:
    """|rho| necessário para significância nesta N (aproximação t)."""
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return t_crit / np.sqrt(df + t_crit ** 2)


def spearman_ci(rho: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """IC via transformação z de Fisher."""
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - (1 - conf) / 2)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def main():
    df = pd.read_csv(PROJECT_DIR / "data" / "salary_inequality_trades_analysis.csv")
    n = len(df)
    rho, p = stats.spearmanr(df['delta_gini'], df['delta_adjusted_perf'])
    ci_lo, ci_hi = spearman_ci(rho, n)
    r_crit = critical_rho(n)
    magnitude = effect_size_label(abs(rho))
    significant = p < ALPHA

    if not significant:
        verdict = "Nenhuma das duas teorias encontra suporte estatístico"
        theory_line = "A associação entre a variação da desigualdade e a variação de performance não é distinguível de zero nesta amostra."
        marker_color = COLOR_NEUTRAL
        banner_tint = COLOR_ZONE_FILL
    elif rho < 0:
        verdict = "A Teoria da Equidade encontra suporte estatístico"
        theory_line = "Nesta amostra, menor desigualdade salarial em quadra está associada a melhora de performance."
        marker_color = COLOR_EQUIDADE
        banner_tint = COLOR_TINT_EQUIDADE
    else:
        verdict = "A Teoria do Torneio encontra suporte estatístico"
        theory_line = "Nesta amostra, maior desigualdade salarial em quadra está associada a melhora de performance."
        marker_color = COLOR_TORNEIO
        banner_tint = COLOR_TINT_TORNEIO

    print(f"N={n}  rho={rho:.4f}  p={p:.4f}  IC95%=({ci_lo:.4f}, {ci_hi:.4f})  "
          f"r_critico(alpha=0,05)={r_crit:.4f}  magnitude={magnitude}  "
          f"significativo={significant}")
    print(f"Veredito: {verdict}")

    # ------------------------------- Figura -------------------------------
    # Monografia, não slide: sem banner de veredito, sem frase de manchete,
    # sem rodapé -- só a régua com seus rótulos de eixo. O texto que
    # explicava o achado em prosa (veredito, IC, N, amostra) fica no corpo
    # do capítulo e na legenda da figura, não dentro da imagem.
    fig = plt.figure(figsize=(11.5, 2.6), dpi=300)
    fig.patch.set_facecolor('white')

    # -------- eixo da régua (-1 a +1) --------
    ax = fig.add_axes([0.07, 0.18, 0.88, 0.68])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.68, 0.95)
    ax.axis('off')

    band_h = 0.30
    # tintas de fundo fora da zona de não-significância
    ax.add_patch(plt.Rectangle((-1, -band_h / 2), (1 - r_crit), band_h,
                                facecolor=COLOR_TINT_EQUIDADE, edgecolor='none', zorder=1))
    ax.add_patch(plt.Rectangle((r_crit, -band_h / 2), (1 - r_crit), band_h,
                                facecolor=COLOR_TINT_TORNEIO, edgecolor='none', zorder=1))
    # zona cinza "sem efeito estatístico" (largura = +-r_critico desta amostra)
    ax.add_patch(plt.Rectangle((-r_crit, -band_h / 2), 2 * r_crit, band_h,
                                facecolor=COLOR_ZONE_FILL, edgecolor=COLOR_ZONE_EDGE,
                                linewidth=1, zorder=2))

    # linha de base (hairline) e marca do zero
    ax.plot([-1, 1], [0, 0], color=COLOR_ZONE_EDGE, linewidth=1, zorder=3)
    ax.plot([0, 0], [-band_h / 2, band_h / 2], color=COLOR_ZONE_EDGE, linewidth=1, zorder=3)

    # ticks -1 / -0.5 / 0 / 0.5 / 1
    for tick in (-1, -0.5, 0, 0.5, 1):
        ax.plot([tick, tick], [-band_h / 2 - 0.05, -band_h / 2], color=COLOR_ZONE_EDGE, linewidth=1, zorder=3)
        ax.text(tick, -band_h / 2 - 0.14, f"{tick:+.1f}" if tick != 0 else "0",
                 ha='center', va='center', fontsize=9, color=INK_SECONDARY)

    # IC 95% (whisker) + estimativa pontual (marcador do veredito)
    ax.plot([ci_lo, ci_hi], [0, 0], color='#333333', linewidth=2.2, zorder=4, solid_capstyle='round')
    ax.plot([ci_lo, ci_lo], [-0.06, 0.06], color='#333333', linewidth=2.2, zorder=4)
    ax.plot([ci_hi, ci_hi], [-0.06, 0.06], color='#333333', linewidth=2.2, zorder=4)
    ax.scatter([rho], [0], s=280, color=marker_color, edgecolors='white', linewidth=2, zorder=5)
    ax.text(rho, 0.20, f"ρ = {rho:+.3f}", ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=INK_PRIMARY)

    # rótulos dos polos (eixo da régua) e da zona neutra -- únicos textos que
    # sobram na figura, porque são necessários para ler a régua, não para
    # narrar o achado.
    ax.text(-1, 0.60, "← EQUIDADE", ha='left', va='center',
            fontsize=10.5, fontweight='bold', color=COLOR_EQUIDADE)
    ax.text(1, 0.60, "TORNEIO →", ha='right', va='center',
            fontsize=10.5, fontweight='bold', color=COLOR_TORNEIO)
    ax.text(0, -0.48, "zona cinza = indiferente (p ≥ 0,05)", ha='center', va='center',
            fontsize=8.3, color=INK_SECONDARY)

    out_path = PROJECT_DIR / "viz" / "salary_h1_verdict.png"
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
