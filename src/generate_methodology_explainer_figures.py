#!/usr/bin/env python3
"""
NBA Salary Inequality x Trades: Methodology & Results Explainer Figures
Author: Data Visualization Specialist

Generates 4 additional figures to make the redesigned methodology and its
results easier to explain, on top of the 3 already produced by
select_inequality_metric.py and analyze_salary_inequality_trades.py:

  1. salary_methodology_flowchart.png  -- visual pipeline of the 5 steps
  2. salary_sample_funnel.png          -- how N goes from 210 to 170 and why
  3. salary_quadrant_analysis.png      -- Tournament vs Equity theory, with
                                          actual data placed in each quadrant
                                          (percentages at 1 decimal place, so
                                          they visibly sum to ~100% instead of
                                          rounding to whole numbers)
  4. salary_gini_pre_post_paired.png   -- paired pre/post Gini per team,
                                          split by direction of the change

All figures are built from the already-saved outputs of
build_trade_activity.py and analyze_salary_inequality_trades.py -- no new
statistics are computed here, this is purely presentation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_data():
    df = pd.read_csv(PROJECT_DIR / "data" / "salary_inequality_trades_analysis.csv")
    df_trade_activity = pd.read_csv(PROJECT_DIR / "data" / "team_trade_activity.csv")
    return df, df_trade_activity


# -----------------------------------------------------------------------------
# 1. METHODOLOGY FLOWCHART
# -----------------------------------------------------------------------------
def generate_flowchart(df: pd.DataFrame, n_total_seasons: int, rho: float, p_val: float):
    n = len(df)
    year_min, year_max = int(df['season_end_year'].min()), int(df['season_end_year'].max())

    fig, ax = plt.subplots(figsize=(15.5, 7), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "METODOLOGIA: DESIGUALDADE SALARIAL E TROCAS DE META-TEMPORADA",
            ha='center', va='center', fontsize=15.5, fontweight='bold', color='#1a252f')
    ax.text(0.5, 0.905, f"Times com troca real no trade deadline, temporadas {year_min-1}-{str(year_min)[2:]} a {year_max-1}-{str(year_max)[2:]} (N={n})",
            ha='center', va='center', fontsize=11, color='#555555')

    p_str = f"{p_val:.4f}".replace('.', ',')
    rho_str = f"{rho:.3f}".replace('.', ',')
    stages = [
        {"num": "1", "title": "Identificar\nTrocas Reais", "color": "#2b5c8f",
         "desc": f"• data/trades/*.csv\n  (transações reais)\n• Janela: 30 dias antes\n  do deadline real\n• {n} de {n_total_seasons} times-\n  temporada qualificam"},
        {"num": "2", "title": "Escolher a\nMétrica de Gini", "color": "#2ca02c",
         "desc": "• 6 métricas candidatas\n• Critério teórico:\n  pondera por minutos\n  jogados\n• Confirmado empiri-\n  camente (não é a\n  regra de escolha)"},
        {"num": "3", "title": "Medir Δ\nDesigualdade", "color": "#ff7f0e",
         "desc": f"• Gini Ponderado por\n  Minutos, pré vs. pós\n  deadline real\n• Só para os {n} times\n  filtrados na Etapa 1"},
        {"num": "4", "title": "Medir Δ\nPerformance", "color": "#d62728",
         "desc": "• Net Rating ajustado\n  pela força do\n  adversário (SRS)\n• Corrige viés de\n  calendário entre as\n  janelas pré/pós"},
        {"num": "5", "title": "Correlacionar\n(Spearman)", "color": "#6f42c1",
         "desc": f"• Matriz Spearman entre\n  Δ Gini, magnitude da\n  troca, Δ performance\n• Teste central:\n  ρ={rho_str}, p={p_str}"},
    ]

    box_w, box_h = 0.163, 0.68
    start_x, y0 = 0.025, 0.10
    gap = 0.018

    for i, st in enumerate(stages):
        x = start_x + i * (box_w + gap)
        rect = patches.FancyBboxPatch((x, y0), box_w, box_h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                       linewidth=2, edgecolor=st["color"], facecolor='#f8f9fa')
        ax.add_patch(rect)
        header = patches.FancyBboxPatch((x + 0.006, y0 + box_h - 0.155), box_w - 0.012, 0.135,
                                         boxstyle="round,pad=0.008,rounding_size=0.018", linewidth=0, facecolor=st["color"])
        ax.add_patch(header)
        ax.text(x + box_w / 2, y0 + box_h - 0.087, f"ETAPA {st['num']}", ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='white')
        ax.text(x + box_w / 2, y0 + box_h - 0.20, st["title"], ha='center', va='top',
                fontsize=11, fontweight='bold', color='#1a252f', linespacing=1.3)
        ax.text(x + 0.014, y0 + box_h - 0.34, st["desc"], ha='left', va='top',
                fontsize=8.7, color='#333333', linespacing=1.45)

        if i < len(stages) - 1:
            ax.annotate('', xy=(x + box_w + gap - 0.003, y0 + box_h / 2), xytext=(x + box_w + 0.003, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=2.2, mutation_scale=14))

    plt.tight_layout()
    fig.savefig(PROJECT_DIR / "viz" / "salary_methodology_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved salary_methodology_flowchart.png")


# -----------------------------------------------------------------------------
# 2. SAMPLE FUNNEL
# -----------------------------------------------------------------------------
def generate_sample_funnel(df: pd.DataFrame, n_total_seasons: int):
    counts_by_season = df.groupby('season_end_year').size()
    all_seasons = sorted(counts_by_season.index.tolist())
    year_min, year_max = all_seasons[0], all_seasons[-1]
    total_per_season = 30  # 30 teams every season in this range

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300, gridspec_kw={'width_ratios': [1, 2.1]})

    # Left: overall funnel
    stages = [f'Times-temporada\nno dataset\n({year_min-1}-{str(year_min)[2:]} a {year_max-1}-{str(year_max)[2:]})', 'Com dados de\nroster completos\n(pré e pós)', 'Com troca real\nno deadline\n(amostra final)']
    values = [n_total_seasons, n_total_seasons, len(df)]
    colors = ['#bdc3c7', '#7f9fc6', '#2b5c8f']
    bars = axes[0].bar(stages, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + n_total_seasons * 0.02, f"N={v}", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_ylim(0, n_total_seasons * 1.12)
    axes[0].set_title("Funil da Amostra", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Times-temporada", fontsize=10.5, fontweight='bold')

    # Right: per-season composition (included vs excluded)
    included = [counts_by_season.get(s, 0) for s in all_seasons]
    excluded = [total_per_season - c for c in included]
    x = np.arange(len(all_seasons))
    axes[1].bar(x, included, color='#2b5c8f', label='Com troca real no deadline (incluído)', edgecolor='black', linewidth=0.6)
    axes[1].bar(x, excluded, bottom=included, color='#dbe4ee', label='Sem troca no deadline (excluído)', edgecolor='black', linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in all_seasons], rotation=45, ha='right')
    axes[1].set_ylim(0, 34)
    axes[1].set_ylabel("Times", fontsize=10.5, fontweight='bold')
    axes[1].set_title("Composição da Amostra por Temporada", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=True, fontsize=9.5)
    for i, v in enumerate(included):
        axes[1].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)

    plt.suptitle(f"Como a Amostra Final (N={len(df)}) foi Construída", fontsize=14.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "viz" / "salary_sample_funnel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved salary_sample_funnel.png")


# -----------------------------------------------------------------------------
# 3. QUADRANT ANALYSIS
# -----------------------------------------------------------------------------
def generate_quadrant_analysis(df: pd.DataFrame):
    x = df['delta_gini']
    y = df['delta_adjusted_perf']
    n = len(df)

    q_torneio_1 = ((x > 0) & (y > 0)).sum()   # mais desigual -> melhora (a favor do Torneio)
    q_equidade = ((x < 0) & (y > 0)).sum()    # menos desigual -> melhora (a favor da Equidade)
    q_torneio_2 = ((x < 0) & (y < 0)).sum()   # menos desigual -> piora (contra a Equidade)
    q_contra_torneio = ((x > 0) & (y < 0)).sum()  # mais desigual -> piora (contra o Torneio)

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)

    xmax = max(abs(x.min()), abs(x.max())) * 1.1
    ymax = max(abs(y.min()), abs(y.max())) * 1.1

    # Shade quadrants
    ax.add_patch(patches.Rectangle((0, 0), xmax, ymax, facecolor='#f8d7da', alpha=0.4, zorder=0))
    ax.add_patch(patches.Rectangle((-xmax, 0), xmax, ymax, facecolor='#d4edda', alpha=0.5, zorder=0))
    ax.add_patch(patches.Rectangle((-xmax, -ymax), xmax, ymax, facecolor='#fff3cd', alpha=0.4, zorder=0))
    ax.add_patch(patches.Rectangle((0, -ymax), xmax, ymax, facecolor='#e2e3e5', alpha=0.5, zorder=0))

    ax.scatter(x, y, s=60, alpha=0.75, color='#2c3e50', edgecolors='white', linewidth=0.5, zorder=3)

    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)

    # Percentuais com 1 casa decimal (30,5% em vez de 30%) -- com arredondamento
    # para inteiro os 4 quadrantes somavam 99% (30+37+17+15) e pareciam não
    # cobrir a amostra toda; com 1 casa fica visível que somam ~100%.
    def pct(count):
        return f"{count / n * 100:.1f}".replace('.', ',')

    label_kwargs = dict(fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(xmax * 0.5, ymax * 0.92, f"A FAVOR DA TEORIA\nDO TORNEIO\n{q_torneio_1} times ({pct(q_torneio_1)}%)",
            color='#721c24', **label_kwargs)
    ax.text(-xmax * 0.5, ymax * 0.92, f"A FAVOR DA TEORIA\nDA EQUIDADE\n{q_equidade} times ({pct(q_equidade)}%)",
            color='#155724', **label_kwargs)
    ax.text(-xmax * 0.5, -ymax * 0.92, f"CONTRA A TEORIA\nDA EQUIDADE\n{q_torneio_2} times ({pct(q_torneio_2)}%)",
            color='#856404', **label_kwargs)
    ax.text(xmax * 0.5, -ymax * 0.92, f"CONTRA A TEORIA\nDO TORNEIO\n{q_contra_torneio} times ({pct(q_contra_torneio)}%)",
            color='#383d41', **label_kwargs)

    ax.set_xlabel('Δ Gini Ponderado por Minutos (Pós - Pré)\n← Desigualdade caiu   |   Desigualdade subiu →', fontsize=11.5, fontweight='bold')
    ax.set_ylabel('← Performance piorou   |   Performance melhorou →\nΔ Performance Ajustada por Adversário (Pós - Pré)', fontsize=11.5, fontweight='bold')
    ax.set_title(f'Times Segundo as Duas Teorias Concorrentes (N={n})', fontsize=14, fontweight='bold', pad=14)

    rho, p_val = stats.spearmanr(x, y)
    ax.text(0.5, -0.13, f"Spearman ρ = {rho:.4f} (p = {p_val:.4f}{'*' if p_val < 0.05 else ''})   |   N = {n}",
            transform=ax.transAxes, ha='center', va='top', fontsize=11.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#555555"))

    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "viz" / "salary_quadrant_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved salary_quadrant_analysis.png")
    return {'q_torneio_1': q_torneio_1, 'q_equidade': q_equidade, 'q_torneio_2': q_torneio_2, 'q_contra_torneio': q_contra_torneio}


# -----------------------------------------------------------------------------
# 4. PAIRED PRE/POST GINI
# -----------------------------------------------------------------------------
def generate_paired_gini(df: pd.DataFrame):
    df = df.copy()
    df['direction'] = np.where(df['delta_gini'] >= 0, 'Aumentou', 'Diminuiu')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})

    # Left: slopegraph sample (subsample for legibility) colored by direction
    rng = np.random.default_rng(42)
    sample_idx = df.index if len(df) <= 60 else rng.choice(df.index, size=60, replace=False)
    df_plot = df.loc[sample_idx]
    # Blue-tone palette (dark navy vs. light blue) instead of red/green -- direction is
    # still readable from lightness/saturation alone, no need for a hue contrast here.
    colors = {'Aumentou': '#1a365d', 'Diminuiu': '#6ba3d6'}
    for _, row in df_plot.iterrows():
        axes[0].plot([0, 1], [row['gini_pre'], row['gini_pos']], color=colors[row['direction']], alpha=0.35, linewidth=1.3)
    for direction, c in colors.items():
        sub = df_plot[df_plot['direction'] == direction]
        axes[0].scatter([0] * len(sub), sub['gini_pre'], color=c, s=18, zorder=3)
        axes[0].scatter([1] * len(sub), sub['gini_pos'], color=c, s=18, zorder=3)
    axes[0].set_xlim(-0.15, 1.15)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Pré-Trade', 'Pós-Trade'], fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Gini Ponderado por Minutos', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Trajetória Individual dos Times\n(amostra de {len(df_plot)} de {len(df)}, para legibilidade)', fontsize=12, fontweight='bold')
    handles = [plt.Line2D([0], [0], color=c, lw=2.5, label=f"{d} (n={len(df[df['direction']==d])})") for d, c in colors.items()]
    axes[0].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True, fontsize=9.5)

    # Right: boxplot pre vs post, full sample
    df_long = pd.melt(df, id_vars=['team_abbreviation'], value_vars=['gini_pre', 'gini_pos'],
                       var_name='periodo', value_name='gini')
    df_long['periodo'] = df_long['periodo'].map({'gini_pre': 'Pré-Trade', 'gini_pos': 'Pós-Trade'})
    sns.boxplot(data=df_long, x='periodo', y='gini', palette={'Pré-Trade': '#2b5c8f', 'Pós-Trade': '#a9cce3'},
                width=0.5, showmeans=True, meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white'), ax=axes[1])
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df['gini_pre'], df['gini_pos'])
    # p-values this small round to "0,0000" at 4 decimals, which reads as literally zero
    # (impossible for a p-value) -- report "p<0,0001" instead once it's below that floor.
    p_label = "p<0,0001" if wilcoxon_p < 0.0001 else f"p={wilcoxon_p:.4f}".replace(".", ",")
    axes[1].set_title(f'Distribuição Completa (N={len(df)})\nWilcoxon pareado: {p_label}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Gini Ponderado por Minutos', fontsize=11, fontweight='bold')

    plt.suptitle('Desigualdade Salarial em Quadra: Antes e Depois do Trade Deadline', fontsize=14.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "viz" / "salary_gini_pre_post_paired.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved salary_gini_pre_post_paired.png")
    return wilcoxon_stat, wilcoxon_p


def main():
    sns.set_theme(style='whitegrid')
    df, df_trade_activity = load_data()

    n_seasons = df['season_end_year'].nunique()
    n_total_seasons = n_seasons * 30  # 30 teams every season in this range
    rho, p_val = stats.spearmanr(df['delta_gini'], df['delta_adjusted_perf'])

    generate_flowchart(df, n_total_seasons, rho, p_val)
    generate_sample_funnel(df, n_total_seasons)
    quadrant_counts = generate_quadrant_analysis(df)
    wilcoxon_stat, wilcoxon_p = generate_paired_gini(df)

    print("\n--- Quadrant counts ---")
    print(quadrant_counts)
    print(f"\n--- Wilcoxon Gini pre vs pos (whole sample) ---\nstat={wilcoxon_stat:.4f} p={wilcoxon_p:.4f}")


if __name__ == "__main__":
    main()
