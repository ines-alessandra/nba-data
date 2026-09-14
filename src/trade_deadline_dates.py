#!/usr/bin/env python3
"""
Real NBA trade deadline date per season (season_end_year -> date).

Replaces the fixed "February 15th" cutoff previously hardcoded across the
trade-deadline inequality pipeline. That fixed date is a reasonable
approximation for most seasons (real deadlines fall Feb 6-10), but is
badly wrong for 2020-21: that season was delayed/compressed by COVID-19
(started Dec 22, 2020) and the trade deadline was pushed to March 25, 2021.
Using Feb 15 for that season misclassified ~6 weeks of pre-deadline games
as "post-trade".

Range covers 2013-2025 (season_end_year). 2011-12 (the lockout season,
deadline 2012-03-15) is deliberately excluded: trade activity in the 30-day
pre-deadline window that year is essentially empty (2 teams, 1 trade
league-wide) and isn't a usable observation for the trades-based analysis.

Sources (cross-checked): NBA.com trade trackers, RealGM trade deadline
history, Hoops Rumors "Key In-Season NBA Dates" (2020-21), SI.com report
on the March 25, 2021 deadline, CBS Sports / Bleacher Report / Peachtree
Hoops trade-deadline recaps for 2013-2018.
"""

TRADE_DEADLINES = {
    2013: '2013-02-21',
    2014: '2014-02-20',
    2015: '2015-02-19',
    2016: '2016-02-18',
    2017: '2017-02-23',
    2018: '2018-02-08',
    2019: '2019-02-07',
    2020: '2020-02-06',
    2021: '2021-03-25',  # COVID-shortened 72-game season
    2022: '2022-02-10',
    2023: '2023-02-09',
    2024: '2024-02-08',
    2025: '2025-02-06',
}


def deadline_sql_case(season_column: str = 'g.season_end_year') -> str:
    """SQL CASE expression mapping season_end_year to its real trade-deadline date."""
    clauses = " ".join(f"WHEN {season_column} = {yr} THEN DATE '{d}'" for yr, d in TRADE_DEADLINES.items())
    return f"(CASE {clauses} END)"
