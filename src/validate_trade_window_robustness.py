#!/usr/bin/env python3
"""
Robustez da Hipótese 1 à largura da janela pré-deadline
Author: Data Scientist

Pergunta: o resultado central da H1 (Spearman rho entre Δ Gini Ponderado por
Minutos e Δ Performance Ajustada, ver analyze_salary_inequality_trades.py)
depende da escolha específica de 30 dias como janela de "atividade de trade
deadline" (build_trade_activity.WINDOW_DAYS_BEFORE_DEADLINE)? Ou o resultado
é estável para larguras de janela vizinhas?

Método: o dataset de Gini/performance pré-pós NÃO depende da largura da
janela (é computado a partir da própria data do deadline, igual para
qualquer janela) -- só a etapa de "quais times-temporada tiveram atividade
de troca qualificante" depende dela. Então: computa-se o dataset completo de
Gini/performance UMA vez (N=390, todos os times-temporada com dados válidos),
e para cada largura de janela candidata, refaz-se só a filtragem por
atividade de troca (reaproveitando build_trade_activity.py) e recalcula-se
Spearman rho/p na amostra resultante.

Saída: data/trade_window_robustness.csv (uma linha por largura de janela) +
resumo impresso no terminal.
"""

import os
from pathlib import Path

import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

import build_trade_activity as bta
from trade_deadline_dates import deadline_sql_case
from salary_inequality_metrics import load_roster_snapshots, compute_period_metrics, PRIMARY_METRIC
from adjusted_performance import load_adjusted_performance

DEFAULT_DB_URL = "postgresql://postgres:root@localhost:5432/nba_pipeline"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# 30 é o valor usado na análise principal; os demais são só para o teste de
# robustez (não substituem o valor de produção em build_trade_activity.py).
WINDOW_WIDTHS_TO_TEST = [14, 21, 30, 45, 60]


def build_full_gini_performance_dataset(engine) -> pd.DataFrame:
    """Igual a analyze_salary_inequality_trades.build_analysis_dataset, MENOS
    o inner-join final com team_trade_activity.csv -- ou seja, todos os
    times-temporada com Gini e performance pré/pós válidos, sem restringir
    ainda a quem teve troca real (essa parte não depende da largura da
    janela, é recalculada uma vez só)."""
    deadline_expr = deadline_sql_case()
    df_snapshots = load_roster_snapshots(engine, deadline_expr)
    df_metrics = compute_period_metrics(df_snapshots)
    df_pre_m = df_metrics[df_metrics['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    df_pos_m = df_metrics[df_metrics['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])

    df = pd.DataFrame(index=df_pre_m.index.intersection(df_pos_m.index))
    df['gini_pre'] = df_pre_m[PRIMARY_METRIC]
    df['gini_pos'] = df_pos_m[PRIMARY_METRIC]
    df['delta_gini'] = df['gini_pos'] - df['gini_pre']

    df_perf = load_adjusted_performance(engine, deadline_expr)
    perf_pre = df_perf[df_perf['periodo'] == 'pre'].set_index(['team_id', 'season_end_year'])
    perf_pos = df_perf[df_perf['periodo'] == 'pos'].set_index(['team_id', 'season_end_year'])
    df['delta_adjusted_perf'] = perf_pos['adjusted_net_rating'] - perf_pre['adjusted_net_rating']
    df['delta_raw_net_rating'] = perf_pos['raw_net_rating'] - perf_pre['raw_net_rating']

    return df.reset_index().dropna(subset=['delta_gini', 'delta_adjusted_perf'])


def qualifying_team_seasons_for_window(df_all_trades: pd.DataFrame, engine, window_days: int) -> pd.DataFrame:
    """Reaproveita build_trade_activity.py inteiro, só trocando a largura da
    janela temporariamente (restaurada no final, via try/finally)."""
    original_window = bta.WINDOW_DAYS_BEFORE_DEADLINE
    bta.WINDOW_DAYS_BEFORE_DEADLINE = window_days
    try:
        df_window = bta.restrict_to_deadline_window(df_all_trades)
    finally:
        bta.WINDOW_DAYS_BEFORE_DEADLINE = original_window
    df_summary = bta.summarize_trade_activity(df_window)
    return bta.attach_team_id(df_summary, engine)


def main():
    engine = create_engine(DB_URL)

    print("Carregando dataset completo de Gini/performance (independe da largura da janela)...")
    df_full = build_full_gini_performance_dataset(engine)
    print(f"N total com dados pré/pós completos: {len(df_full)}")

    print("\nCarregando ledger de trocas reais...")
    df_all_trades = bta.load_all_trades()

    print(f"\n{'Janela (dias)':>14} {'N':>6} {'Spearman rho':>13} {'p-valor':>10}   Significativo?")
    print("-" * 66)
    results = []
    for w in WINDOW_WIDTHS_TO_TEST:
        df_qualifying = qualifying_team_seasons_for_window(df_all_trades, engine, w)
        df_sample = pd.merge(
            df_full,
            df_qualifying[['team_id', 'season_end_year', 'n_trades', 'n_players_moved']],
            on=['team_id', 'season_end_year'], how='inner',
        )
        n = len(df_sample)
        rho, p = stats.spearmanr(df_sample['delta_gini'], df_sample['delta_adjusted_perf'])
        sig = p < 0.05
        results.append({'window_days': w, 'n': n, 'rho': rho, 'p_value': p, 'significant': sig,
                         'is_production_value': w == 30})
        marker = "  <- valor usado na análise" if w == 30 else ""
        print(f"{w:>14} {n:>6} {rho:>+13.4f} {p:>10.4f}   {'sim' if sig else 'não'}{marker}")

    df_results = pd.DataFrame(results)
    project_dir = Path(__file__).resolve().parent.parent
    out_csv = project_dir / "data" / "trade_window_robustness.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSalvo em {out_csv}")

    print("\n" + "=" * 70)
    print("RESUMO DA ROBUSTEZ")
    print("=" * 70)
    rho_min, rho_max = df_results['rho'].min(), df_results['rho'].max()
    all_negative = (df_results['rho'] < 0).all()
    all_significant = bool(df_results['significant'].all())
    print(f"rho varia de {rho_min:.4f} a {rho_max:.4f} entre janelas de "
          f"{min(WINDOW_WIDTHS_TO_TEST)} a {max(WINDOW_WIDTHS_TO_TEST)} dias.")
    print(f"Sinal negativo (a favor da Equidade) em TODAS as janelas testadas? {all_negative}")
    print(f"Estatisticamente significativo (p<0,05) em TODAS as janelas testadas? {all_significant}")


if __name__ == "__main__":
    main()
