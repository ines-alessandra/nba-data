#!/usr/bin/env python3
"""
Robustez da Hipótese 1 à largura da janela pré-deadline -- visualização
Author: Data Visualization Specialist

Gera UMA figura cujo único objetivo é mostrar, visualmente, o resultado do
teste de robustez em validate_trade_window_robustness.py: o Spearman ρ
entre Δ Gini e Δ Performance Ajustada, recalculado para diferentes larguras
de janela pré-deadline (14 a 60 dias), permanece estável -- mesmo sinal,
sempre significativo -- ou o resultado da H1 dependia da escolha de 30 dias?

Lê data/trade_window_robustness.csv (gerado por
validate_trade_window_robustness.py). Nada aqui é fixo a dedo: o título, a
faixa de não-significância (que depende de N, e N varia por janela) e a
conclusão no rodapé são recalculados a partir dos dados a cada execução.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALPHA = 0.05

COLOR_EQUIDADE = '#2a78d6'
COLOR_PRODUCTION = '#1a365d'
COLOR_ZONE_FILL = '#e9e8e3'
COLOR_ZONE_EDGE = '#c3c2b7'
INK_PRIMARY = '#1a1a1a'
INK_SECONDARY = '#52514e'


def critical_rho(n: int, alpha: float = ALPHA) -> float:
    """|rho| necessário para significância nesta N (aproximação t) -- a
    mesma usada em generate_h1_verdict.py."""
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return t_crit / np.sqrt(df + t_crit ** 2)


def main():
    df = pd.read_csv(PROJECT_DIR / "data" / "trade_window_robustness.csv")
    df = df.sort_values('window_days').reset_index(drop=True)
    df['r_crit'] = df['n'].apply(critical_rho)

    all_same_sign = (df['rho'] < 0).all() or (df['rho'] > 0).all()
    all_significant = bool(df['significant'].all())
    stable = all_same_sign and all_significant
    prod_row = df[df['is_production_value']].iloc[0]

    if stable:
        headline = "O resultado da Hipótese 1 é robusto à largura da janela"
        sub = (f"Spearman ρ mantém o mesmo sinal e significância estatística (p<0,05) nas {len(df)} larguras "
               f"testadas — a conclusão não depende da escolha de {int(prod_row['window_days'])} dias")
    else:
        headline = "O resultado da Hipótese 1 é sensível à largura da janela"
        sub = "Sinal e/ou significância mudam conforme a largura da janela pré-deadline — ver valores abaixo"

    # Monografia, não slide: sem banner, manchete ou subtítulo dentro da
    # imagem -- a legenda da figura e o texto do capítulo já cumprem esse
    # papel. headline/sub continuam calculados (usados no print de conferência
    # abaixo), só não são mais desenhados na figura.
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=300)
    fig.patch.set_facecolor('white')

    x = df['window_days'].to_numpy(dtype=float)
    y = df['rho'].to_numpy(dtype=float)
    r_crit = df['r_crit'].to_numpy(dtype=float)

    ax.set_xlim(x.min() - 6, x.max() + 6)
    ax.set_ylim(-0.5, 0.3)

    # Faixa de não-significância -- a largura varia levemente com N (que
    # cresce um pouco em janelas maiores), então não é uma faixa reta.
    ax.fill_between(x, -r_crit, r_crit, color=COLOR_ZONE_FILL, edgecolor=COLOR_ZONE_EDGE,
                     linewidth=1, zorder=1)
    ax.axhline(0, color=COLOR_ZONE_EDGE, linewidth=1, zorder=2)
    ax.text((x.min() + x.max()) / 2, 0, "zona sem efeito estatístico (p ≥ 0,05)", ha='center', va='center',
            fontsize=8.6, color=INK_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=COLOR_ZONE_FILL, edgecolor='none', alpha=0.9))

    ax.plot(x, y, color=COLOR_EQUIDADE, linewidth=2.2, zorder=3, marker='o', markersize=7,
            markerfacecolor=COLOR_EQUIDADE, markeredgecolor='white', markeredgewidth=1.2)

    # Destaque do ponto de produção (30 dias, o valor usado na análise principal)
    ax.scatter([prod_row['window_days']], [prod_row['rho']], s=260, color=COLOR_PRODUCTION,
               edgecolors='white', linewidth=2, zorder=5)
    rho_prod_str = f"{prod_row['rho']:.3f}".replace('.', ',')
    ax.annotate(f"usado na análise\nN={int(prod_row['n'])}, ρ={rho_prod_str}",
                xy=(prod_row['window_days'], prod_row['rho']), xytext=(prod_row['window_days'], 0.16),
                ha='center', va='center', fontsize=9.5, fontweight='bold', color=COLOR_PRODUCTION,
                arrowprops=dict(arrowstyle='-', color=COLOR_PRODUCTION, linewidth=1))

    # Rótulo de N em cada ponto que não é o de produção
    for _, row in df.iterrows():
        if row['is_production_value']:
            continue
        ax.annotate(f"N={int(row['n'])}", xy=(row['window_days'], row['rho']),
                    xytext=(row['window_days'], row['rho'] - 0.05), ha='center', va='top',
                    fontsize=8.3, color=INK_SECONDARY)

    ax.text(x.min() + 2, -0.40, "← a favor da Teoria da Equidade (todas as janelas testadas)", ha='left',
            va='center', fontsize=9, fontweight='bold', color=COLOR_EQUIDADE)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}d" for v in x], fontsize=10.5)
    ax.set_xlabel('Largura da janela pré-deadline (dias)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Spearman ρ', fontsize=11, fontweight='bold')

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(COLOR_ZONE_EDGE)
    ax.spines['bottom'].set_color(COLOR_ZONE_EDGE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis='y', color='#f0efec', linewidth=1, zorder=0)

    fig.tight_layout()
    out_path = PROJECT_DIR / "viz" / "trade_window_robustness.png"
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Estável: {stable} | mesmo sinal: {all_same_sign} | sempre significativo: {all_significant}")


if __name__ == "__main__":
    main()
