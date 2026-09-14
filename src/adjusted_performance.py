#!/usr/bin/env python3
"""
NBA Opponent-Adjusted Performance by Period (pre/post trade deadline)
Author: Data Engineer

Computes, per team-season-period, the average Net Rating weighted by each
game's opponent strength (opponent's full-season SRS rating) -- the same
weighting scheme already used for the `adjusted_net_rating` feature in
build_team_features.py, just computed separately for the pre-deadline and
post-deadline windows instead of over the whole season.

Why this instead of raw Net Rating: the pre and post windows face different
schedules (different opponents, different point in the season), so a team's
raw Net Rating can shift between windows purely from strength-of-schedule
differences that have nothing to do with a trade. Weighting each game by
its opponent's season-long SRS controls for that.

The min-SRS shift constant is computed once per season (not separately per
window) so pre and post use the same reference scale.
"""

import numpy as np
import pandas as pd


def load_adjusted_performance(engine, deadline_expr: str) -> pd.DataFrame:
    sql_query = f"""
    SELECT
        f.team_id,
        f.opponent_team_id,
        f.net_rating,
        g.game_date,
        g.season_end_year,
        CASE WHEN g.game_date < {deadline_expr} THEN 'pre' ELSE 'pos' END AS periodo,
        s.srs_rating AS opp_srs_rating
    FROM silver.fact_team_gamelogs f
    JOIN silver.dim_games g ON f.game_id = g.game_id
    LEFT JOIN gold.team_sos s ON f.opponent_team_id = s.team_id AND g.season_year = s.season_year
    WHERE g.game_type = 'Regular Season'
      AND g.season_end_year BETWEEN 2013 AND 2025;
    """
    df = pd.read_sql(sql_query, engine)

    # Shift constant computed once per season (not per period) so pre/post share a scale.
    min_srs_by_season = df.groupby('season_end_year')['opp_srs_rating'].min().to_dict()

    records = []
    grouped = df.groupby(['team_id', 'season_end_year', 'periodo'])
    for (team_id, season_end_year, periodo), group in grouped:
        min_srs = min_srs_by_season.get(season_end_year, 0.0)
        opp_srs = group['opp_srs_rating'].fillna(0.0)
        weights = opp_srs - min_srs + 1.0
        net_ratings = group['net_rating'].fillna(0.0)

        adjusted = float(np.average(net_ratings, weights=weights)) if weights.sum() > 0 else float(net_ratings.mean())
        raw_avg = float(net_ratings.mean())

        records.append({
            'team_id': team_id,
            'season_end_year': season_end_year,
            'periodo': periodo,
            'adjusted_net_rating': adjusted,
            'raw_net_rating': raw_avg,
            'games_played': len(group),
        })

    return pd.DataFrame(records)
