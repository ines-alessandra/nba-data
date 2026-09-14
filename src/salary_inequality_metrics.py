#!/usr/bin/env python3
"""
NBA Salary Inequality Metrics (shared formulas)
Author: Data Engineer

Six candidate metrics for "how unequal is a team's salary structure",
computed per team-season-period from a roster snapshot of
(player_id, salary_inflation_adjusted, minutes_played):

  1. Standard Deviation (SD) of salaries
  2. Coefficient of Variation (CV) of salaries
  3. Top1 / Bottom1 salary ratio
  4. Top-3 salary share of total payroll
  5. Simple Gini index of salaries (unweighted)
  6. Gini index of salaries weighted by minutes played (Lorenz curve
     weighted by on-court time instead of roster count)

Metrics 1-5 treat every rostered player the same regardless of whether they
actually play; metric 6 is the only one that reflects pay inequality as
experienced ON THE COURT, which is the reason it was selected as the primary
metric in earlier iterations of this analysis (see select_inequality_metric.py
for the full adequacy evaluation, not just this theoretical argument).
"""

import numpy as np
import pandas as pd


def calculate_gini_simple(S: np.ndarray) -> float:
    n = len(S)
    mean_s = np.mean(S)
    if n == 0 or mean_s == 0:
        return 0.0
    mad = np.abs(np.subtract.outer(S, S)).mean()
    return 0.5 * mad / mean_s


def calculate_gini_weighted_minutes(S: np.ndarray, M: np.ndarray) -> float:
    valid = M > 0
    if not np.any(valid):
        return 0.0
    S_v, M_v = S[valid], M[valid]
    idx = np.argsort(S_v)
    S_sorted, M_sorted = S_v[idx], M_v[idx]
    M_total = np.sum(M_sorted)
    WS = S_sorted * M_sorted
    WS_total = np.sum(WS)
    if M_total == 0 or WS_total == 0:
        return 0.0
    cum_M = np.cumsum(M_sorted)
    cum_WS = np.cumsum(WS)
    p = np.concatenate([[0.0], cum_M / M_total])
    L = np.concatenate([[0.0], cum_WS / WS_total])
    B = np.sum((L[1:] + L[:-1]) / 2.0 * (p[1:] - p[:-1]))
    return 1.0 - 2.0 * B


METRIC_COLUMNS = [
    'desvio_padrao', 'coeficiente_variacao', 'razao_top1_bottom1',
    'proporcao_top3', 'gini_simples', 'gini_ponderado_minutos',
]

METRIC_LABELS = {
    'desvio_padrao': 'Desvio Padrão (SD)',
    'coeficiente_variacao': 'Coeficiente de Variação (CV)',
    'razao_top1_bottom1': 'Razão Top1 / Bottom1',
    'proporcao_top3': 'Proporção Top 3',
    'gini_simples': 'Gini Simples',
    'gini_ponderado_minutos': 'Gini Ponderado por Minutos',
}

PRIMARY_METRIC = 'gini_ponderado_minutos'


def compute_all_metrics(S: np.ndarray, M: np.ndarray) -> dict:
    """Compute all 6 metrics for one roster snapshot (S=salaries, M=minutes)."""
    mean_s = np.mean(S)
    std_s = np.std(S, ddof=1) if len(S) > 1 else 0.0
    cv = std_s / mean_s if mean_s > 0 else np.nan

    max_s = np.max(S)
    min_pos_s = np.min(S[S > 0]) if np.any(S > 0) else np.nan
    ratio_top1_bottom1 = max_s / min_pos_s if min_pos_s and min_pos_s > 0 else np.nan

    top3_sum = np.sum(np.sort(S)[-3:])
    total_sum = np.sum(S)
    prop_top3 = top3_sum / total_sum if total_sum > 0 else np.nan

    return {
        'desvio_padrao': std_s,
        'coeficiente_variacao': cv,
        'razao_top1_bottom1': ratio_top1_bottom1,
        'proporcao_top3': prop_top3,
        'gini_simples': calculate_gini_simple(S),
        'gini_ponderado_minutos': calculate_gini_weighted_minutes(S, M),
    }


def load_roster_snapshots(engine, deadline_expr: str) -> pd.DataFrame:
    """
    Fetch (team_id, season_end_year, periodo, player_id, salary, minutes_played)
    split into 'pre'/'pos' by the real trade-deadline date of each season
    (deadline_expr: SQL CASE expression from trade_deadline_dates.deadline_sql_case()).
    """
    sql_query = f"""
    SELECT
        pg.team_id,
        g.season_end_year,
        CASE WHEN g.game_date < {deadline_expr} THEN 'pre' ELSE 'pos' END AS periodo,
        pg.player_id,
        s.salary_inflation_adjusted,
        SUM(pg.minutes) AS minutes_played
    FROM silver.fact_player_gamelogs pg
    JOIN silver.dim_games g ON pg.game_id = g.game_id
    JOIN silver.fact_player_salaries s
      ON pg.player_id = s.player_id
     AND g.season_end_year = s.season_end_year
    WHERE g.game_type = 'Regular Season'
      AND g.season_end_year BETWEEN 2013 AND 2025
      AND s.salary_inflation_adjusted IS NOT NULL
      AND s.salary_inflation_adjusted > 0
    GROUP BY pg.team_id, g.season_end_year, periodo, pg.player_id, s.salary_inflation_adjusted
    HAVING SUM(pg.minutes) >= 1.0;
    """
    return pd.read_sql(sql_query, engine)


def compute_period_metrics(df_snapshots: pd.DataFrame, min_roster_size: int = 5) -> pd.DataFrame:
    """Group roster snapshots by (team_id, season_end_year, periodo) and compute
    all 6 metrics, dropping periods with fewer than min_roster_size players."""
    records = []
    grouped = df_snapshots.groupby(['team_id', 'season_end_year', 'periodo'])
    for (team_id, season_end_year, periodo), group in grouped:
        S = group['salary_inflation_adjusted'].values
        M = group['minutes_played'].values
        if len(S) < min_roster_size:
            continue
        rec = {'team_id': team_id, 'season_end_year': season_end_year, 'periodo': periodo, 'n_jogadores': len(S)}
        rec.update(compute_all_metrics(S, M))
        records.append(rec)
    return pd.DataFrame(records)
