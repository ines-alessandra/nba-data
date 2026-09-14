#!/usr/bin/env python3
"""
NBA Salary Inequality Metric Adequacy Evaluation
Author: Data Scientist

Answers "which of the 6 candidate salary-inequality metrics is most adequate
to use in the trade-deadline analysis", using three kinds of evidence -- not
just one, to avoid picking the metric purely because it happens to correlate
best with the outcome (which would be a form of circular specification
search: choosing the metric that produces the result you're about to test):

  1. THEORETICAL adequacy (primary criterion): does the metric measure
     inequality as actually experienced on the court? SD, CV, the Top1/Bottom1
     ratio, the Top-3 share, and the simple Gini all treat a $2M bench player
     who plays 200 minutes the same as a $2M rotation player who plays 2,000
     -- they describe the payroll, not the team. Only the minutes-weighted
     Gini reflects pay inequality as distributed across actual playing time,
     which is the construct this research question is about.

  2. DISTRIBUTIONAL diagnostics (descriptive, not a Pearson/Spearman switch
     this time): Shapiro-Wilk normality and IQR outlier counts on each
     metric's delta, reported to justify using Spearman uniformly rather than
     to pick a "winner".

  3. EMPIRICAL checks (confirmatory, not decisive on their own):
     a. Spearman correlation of each metric's delta against the delta in
        opponent-adjusted performance, restricted to team-seasons with real
        trade-deadline activity (see build_trade_activity.py). Reported for
        transparency, but explicitly NOT used as the sole selection rule.
     b. A Spearman correlation matrix AMONG the 6 metrics' deltas, to check
        whether the minutes-weighted Gini is actually capturing something
        distinct from the simpler payroll-only metrics (if it moved in
        lockstep with all of them, weighting by minutes would add no
        information).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sqlalchemy import create_engine

from trade_deadline_dates import deadline_sql_case
from salary_inequality_metrics import (
    load_roster_snapshots, compute_period_metrics, METRIC_COLUMNS, METRIC_LABELS, PRIMARY_METRIC,
)
from adjusted_performance import load_adjusted_performance

DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)


def build_delta_table(engine) -> pd.DataFrame:
    deadline_expr = deadline_sql_case()

    df_snapshots = load_roster_snapshots(engine, deadline_expr)
    df_metrics = compute_period_metrics(df_snapshots)

    df_perf = load_adjusted_performance(engine, deadline_expr)

    project_dir = Path(__file__).resolve().parent.parent
    df_trade_activity = pd.read_csv(project_dir / "data" / "team_trade_activity.csv")

    df_pre = df_metrics[df_metrics['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    df_pos = df_metrics[df_metrics['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])

    deltas = pd.DataFrame(index=df_pre.index.intersection(df_pos.index))
    for col in METRIC_COLUMNS:
        deltas[f'delta_{col}'] = df_pos[col] - df_pre[col]

    perf_pre = df_perf[df_perf['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    perf_pos = df_perf[df_perf['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])
    deltas['delta_adjusted_perf'] = perf_pos['adjusted_net_rating'] - perf_pre['adjusted_net_rating']
    deltas['delta_raw_net_rating'] = perf_pos['raw_net_rating'] - perf_pre['raw_net_rating']

    deltas = deltas.reset_index()

    # Restrict to team-seasons with real trade-deadline activity
    trade_keys = set(zip(df_trade_activity['team_id'], df_trade_activity['season_end_year']))
    deltas = deltas[deltas.apply(lambda r: (r['team_id'], r['season_end_year']) in trade_keys, axis=1)]

    return deltas.dropna().reset_index(drop=True)


def run_diagnostics(deltas: pd.DataFrame) -> pd.DataFrame:
    results = []
    for col in METRIC_COLUMNS:
        data = deltas[f'delta_{col}'].dropna()

        shapiro_stat, shapiro_p = stats.shapiro(data)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())

        rho, p_val = stats.spearmanr(data, deltas['delta_adjusted_perf'])

        results.append({
            'metrica': col,
            'shapiro_p': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'outliers_iqr': outliers,
            'spearman_rho_vs_perf': rho,
            'spearman_p_vs_perf': p_val,
        })
    return pd.DataFrame(results)


def main():
    engine = create_engine(DB_URL)
    deltas = build_delta_table(engine)
    n = len(deltas)
    print(f"Team-seasons with real trade-deadline activity AND complete metrics: N={n}")

    df_diag = run_diagnostics(deltas)

    print("\n" + "=" * 90)
    print(" CRITÉRIO 2 (descritivo): NORMALIDADE E OUTLIERS DE CADA MÉTRICA ")
    print("=" * 90)
    print(df_diag[['metrica', 'shapiro_p', 'is_normal', 'outliers_iqr']].to_string(index=False))
    print("Nenhuma troca de método Pearson/Spearman é feita aqui -- Spearman é usado")
    print("uniformemente na análise principal; isto é só diagnóstico descritivo.")

    print("\n" + "=" * 90)
    print(" CRITÉRIO 3a (confirmatório, não decisivo sozinho): CORRELAÇÃO COM Δ PERFORMANCE ")
    print(f" (N={n}, apenas times com atividade real de troca no deadline)")
    print("=" * 90)
    df_diag_sorted = df_diag.reindex(df_diag['spearman_rho_vs_perf'].abs().sort_values(ascending=False).index)
    for _, row in df_diag_sorted.iterrows():
        marker = " <- escolhida (critério teórico)" if row['metrica'] == PRIMARY_METRIC else ""
        print(f"  {METRIC_LABELS[row['metrica']]:<32} rho={row['spearman_rho_vs_perf']:+.4f}  p={row['spearman_p_vs_perf']:.4f}{marker}")

    print("\n" + "=" * 90)
    print(" CRITÉRIO 3b: MATRIZ DE CORRELAÇÃO ENTRE AS 6 MÉTRICAS (mostra se o Gini")
    print(" Ponderado captura algo distinto das métricas que ignoram minutos jogados) ")
    print("=" * 90)
    delta_cols = [f'delta_{c}' for c in METRIC_COLUMNS]
    corr_matrix = deltas[delta_cols].corr(method='spearman')
    corr_matrix.index = [METRIC_LABELS[c] for c in METRIC_COLUMNS]
    corr_matrix.columns = [METRIC_LABELS[c] for c in METRIC_COLUMNS]
    print(corr_matrix.round(3).to_string())

    print("\n" + "=" * 90)
    print(f" CONCLUSÃO: métrica primária = {METRIC_LABELS[PRIMARY_METRIC]}")
    print(" Critério 1 (teórico) é decisivo: é a única das 6 que pondera por minutos")
    print(" jogados, i.e. mede a desigualdade que os jogadores efetivamente vivem em")
    print(" quadra, não apenas a folha salarial nominal. Critérios 2 e 3 são reportados")
    print(" para transparência/robustez, não como regra de decisão automática.")
    print("=" * 90)

    # Save diagnostics table
    project_dir = Path(__file__).resolve().parent.parent
    out_csv = project_dir / "data" / "inequality_metric_adequacy.csv"
    df_diag.to_csv(out_csv, index=False)
    print(f"\nSaved diagnostics table to: {out_csv}")

    # Two SEPARATE figures (previously one 1x2 panel, split for legibility and
    # because each half answers a different question and deserves its own
    # caption instead of sharing one suptitle):
    #   1. inequality_metric_correlation_heatmap.png -- Critério 3b (are the 6
    #      metrics redundant with each other, or does minutes-weighting add
    #      information?)
    #   2. inequality_metric_performance_bar.png -- Critério 3a (empirical,
    #      confirmatory correlation with performance -- NOT the selection rule)
    sns.set_theme(style='whitegrid')

    # --- Figure 1: inter-metric correlation heatmap (Critério 3b) ---
    # Monografia, não slide: sem título nem rodapé embutidos na imagem -- a
    # legenda da figura e o texto do capítulo cumprem esse papel. Só o mapa
    # de calor com seus rótulos de eixo e a barra de cor.
    fig1, ax1 = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax1, cbar_kws={'label': "Spearman ρ"})
    ax1.tick_params(axis='x', rotation=35)
    plt.tight_layout()
    out_png1 = project_dir / "viz" / "inequality_metric_correlation_heatmap.png"
    fig1.savefig(out_png1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved plot to: {out_png1}")

    # --- Figure 2: empirical correlation with performance (Critério 3a) ---
    # Monografia, não slide: sem título nem rodapé embutidos na imagem -- a
    # legenda da figura e o texto do capítulo já explicam por que este
    # gráfico é confirmatório, não o critério de escolha.
    fig2, ax2 = plt.subplots(figsize=(9.5, 6.5))
    plot_df = df_diag_sorted.copy()
    plot_df['label'] = plot_df['metrica'].map(METRIC_LABELS)
    colors = ['#d62728' if m == PRIMARY_METRIC else '#bdc3c7' for m in plot_df['metrica']]
    ax2.barh(plot_df['label'], plot_df['spearman_rho_vs_perf'], color=colors, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.invert_yaxis()
    ax2.set_xlabel("Spearman ρ com Δ Performance Ajustada", fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_png2 = project_dir / "viz" / "inequality_metric_performance_bar.png"
    fig2.savefig(out_png2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved plot to: {out_png2}")


if __name__ == "__main__":
    main()
