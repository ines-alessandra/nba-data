#!/usr/bin/env python3
"""
NBA Salary Inequality and Mid-Season Trades: Impact on Performance
Author: Data Scientist

Core research question: among teams that actually transacted around the
real trade deadline of their season, is a change in on-court salary
inequality (Gini weighted by minutes) associated with a change in
performance (opponent-adjusted Net Rating)? And does the SIZE of the trade
activity itself relate to either?

Design (see conversation / methodology proposal for the full rationale):
  - Sample restricted to team-seasons with >=1 real trade in the 30 days
    before that season's real trade deadline (build_trade_activity.py),
    instead of comparing all 210 team-seasons regardless of whether they
    traded.
  - Inequality metric: Gini weighted by minutes played (see
    select_inequality_metric.py for why, and for the deltas of the other
    5 candidate metrics).
  - Performance metric: opponent-adjusted Net Rating, computed separately
    for the pre- and post-deadline windows (adjusted_performance.py),
    instead of raw Net Rating, to control for schedule-strength differences
    between the two windows. Raw Net Rating is kept as a secondary/
    robustness variable.
  - Trade magnitude: number of players moved in the deadline window
    (in + out), a variable the old design didn't have at all.
  - Method: Spearman correlation only, throughout, reported as a single
    correlation matrix across all variable pairs (not one isolated test).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sqlalchemy import create_engine

from trade_deadline_dates import deadline_sql_case
from salary_inequality_metrics import load_roster_snapshots, compute_period_metrics, PRIMARY_METRIC
from adjusted_performance import load_adjusted_performance

DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

VARIABLES = {
    'gini_pre': 'Gini Ponderado (Pré-Trade, nível inicial)',
    'delta_gini': 'Δ Gini Ponderado por Minutos',
    'trade_magnitude': 'Magnitude da Troca (jogadores movimentados)',
    'delta_adjusted_perf': 'Δ Performance (Net Rating ajustado por adversário)',
    'delta_raw_net_rating': 'Δ Net Rating (bruto, robustez)',
}


def build_analysis_dataset(engine) -> pd.DataFrame:
    deadline_expr = deadline_sql_case()

    # Inequality metric (pre/post)
    df_snapshots = load_roster_snapshots(engine, deadline_expr)
    df_metrics = compute_period_metrics(df_snapshots)
    df_pre_m = df_metrics[df_metrics['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    df_pos_m = df_metrics[df_metrics['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])

    df = pd.DataFrame(index=df_pre_m.index.intersection(df_pos_m.index))
    df['gini_pre'] = df_pre_m[PRIMARY_METRIC]
    df['gini_pos'] = df_pos_m[PRIMARY_METRIC]
    df['delta_gini'] = df['gini_pos'] - df['gini_pre']

    # Performance (pre/post, opponent-adjusted + raw)
    df_perf = load_adjusted_performance(engine, deadline_expr)
    perf_pre = df_perf[df_perf['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    perf_pos = df_perf[df_perf['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])
    df['delta_adjusted_perf'] = perf_pos['adjusted_net_rating'] - perf_pre['adjusted_net_rating']
    df['delta_raw_net_rating'] = perf_pos['raw_net_rating'] - perf_pre['raw_net_rating']

    df = df.reset_index()

    # Trade activity (magnitude + filter)
    project_dir = Path(__file__).resolve().parent.parent
    df_trades = pd.read_csv(project_dir / "data" / "team_trade_activity.csv")
    df = pd.merge(df, df_trades[['team_id', 'season_end_year', 'n_trades', 'n_players_moved']],
                  on=['team_id', 'season_end_year'], how='inner')  # inner join = the trade-activity filter
    df = df.rename(columns={'n_players_moved': 'trade_magnitude'})

    # Metadata for readability
    df_meta = pd.read_sql("SELECT DISTINCT team_id, team_abbreviation FROM silver.dim_teams", engine)
    df = pd.merge(df, df_meta, on='team_id', how='left')

    return df.dropna(subset=list(VARIABLES.keys())).reset_index(drop=True)


def run_correlation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(VARIABLES.keys())
    rho_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            rho, p = stats.spearmanr(df[c1], df[c2])
            rho_matrix.loc[c1, c2] = rho
            p_matrix.loc[c1, c2] = p
    return rho_matrix, p_matrix


def main():
    engine = create_engine(DB_URL)
    df = build_analysis_dataset(engine)
    n = len(df)
    print(f"Amostra final (times com atividade real de troca no deadline): N={n}")
    print(df.groupby('season_end_year').size().to_string())

    project_dir = Path(__file__).resolve().parent.parent
    out_csv = project_dir / "data" / "salary_inequality_trades_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved analysis dataset to: {out_csv}")

    rho_matrix, p_matrix = run_correlation_matrix(df)

    print("\n" + "=" * 100)
    print(f" MATRIZ DE CORRELAÇÃO DE SPEARMAN (N={n}) ")
    print("=" * 100)
    display_matrix = rho_matrix.copy()
    display_matrix.index = [VARIABLES[c] for c in display_matrix.index]
    display_matrix.columns = [VARIABLES[c] for c in display_matrix.columns]
    print(display_matrix.round(3).to_string())

    print("\n" + "-" * 100)
    print(" TESTE CENTRAL: Δ Gini Ponderado x Δ Performance Ajustada ")
    print("-" * 100)
    rho_c, p_c = stats.spearmanr(df['delta_gini'], df['delta_adjusted_perf'])
    sig = "SIM (p<0,05)" if p_c < 0.05 else "Não (p>=0,05)"
    print(f"  Spearman rho = {rho_c:.4f}   p-valor = {p_c:.4f}   Significativo? {sig}   N={n}")

    print("\n" + "-" * 100)
    print(" TESTES SECUNDÁRIOS ")
    print("-" * 100)
    for pair, label in [
        (('trade_magnitude', 'delta_adjusted_perf'), "Magnitude da troca x Δ Performance"),
        (('trade_magnitude', 'delta_gini'), "Magnitude da troca x Δ Gini"),
        (('gini_pre', 'delta_adjusted_perf'), "Nível inicial de Gini x Δ Performance"),
        (('delta_gini', 'delta_raw_net_rating'), "Δ Gini x Δ Net Rating bruto (robustez)"),
    ]:
        c1, c2 = pair
        rho, p = stats.spearmanr(df[c1], df[c2])
        sig = "*" if p < 0.05 else ""
        print(f"  {label:<45} rho={rho:+.4f}  p={p:.4f}{sig}")
    print("=" * 100)

    # --- Plots ---
    sns.set_theme(style='whitegrid')

    # 1. Correlation matrix heatmap
    plt.figure(figsize=(9, 7.5))
    short_labels = ['Gini\n(nível pré)', 'Δ Gini', 'Magnitude\nda troca', 'Δ Perf.\n(ajustada)', 'Δ Net Rating\n(bruto)']
    plot_matrix = rho_matrix.copy()
    plot_matrix.index = short_labels
    plot_matrix.columns = short_labels
    # Monochromatic blue diverging colormap (dark navy = negative, white = zero, mid blue =
    # positive) instead of coolwarm's red/blue -- sign is still readable from lightness alone.
    blue_diverging = LinearSegmentedColormap.from_list('blue_diverging', ['#08306b', '#f7fbff', '#2a78d6'])
    sns.heatmap(plot_matrix.astype(float), cmap=blue_diverging, center=0, annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={'label': 'Spearman ρ'})
    # Season range read from the data actually plotted, not hardcoded -- this dataset now
    # spans further back than the 2019-2025 window an earlier iteration of this script used.
    year_min, year_max = int(df['season_end_year'].min()), int(df['season_end_year'].max())
    plt.title(f'Matriz de Correlação de Spearman (N={n})\n'
              f'Times com troca real no deadline, {year_min-1}-{str(year_min)[2:]} a {year_max-1}-{str(year_max)[2:]}',
              fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(project_dir / "viz" / "salary_inequality_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Core scatter with LOESS
    plt.figure(figsize=(10, 6.5))
    plt.scatter(df['delta_gini'], df['delta_adjusted_perf'], s=70, alpha=0.7, color='#4a5568', edgecolors='none')
    lowess = sm.nonparametric.lowess
    z = lowess(df['delta_adjusted_perf'], df['delta_gini'], frac=0.6)
    plt.plot(z[:, 0], z[:, 1], color='#1a365d', linewidth=3, label='Curva LOESS')
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Δ Gini Ponderado por Minutos (Pós - Pré)', fontsize=12, fontweight='bold')
    plt.ylabel('Δ Performance Ajustada por Adversário (Pós - Pré)', fontsize=12, fontweight='bold')
    # Monografia, não slide: sem título embutido na imagem (a legenda da
    # figura no capítulo já identifica o gráfico) e sem caixa de anotação com
    # rho/p (o texto do capítulo já reporta esses números com precisão).
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(project_dir / "viz" / "salary_inequality_core_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved plots to viz/salary_inequality_correlation_matrix.png and viz/salary_inequality_core_scatter.png")


if __name__ == "__main__":
    main()
