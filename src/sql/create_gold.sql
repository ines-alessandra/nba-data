-- ====================================================================
-- GOLD SCHEMA: ANALYTICAL TABLES & AGGREGATES FOR STUDY
-- ====================================================================

-- Dropping existing tables if any, for clean runs
DROP TABLE IF EXISTS gold.player_trade_value_analysis CASCADE;
DROP TABLE IF EXISTS gold.team_trade_salary_impact CASCADE;
DROP TABLE IF EXISTS gold.player_trade_advanced_espn CASCADE;
DROP TABLE IF EXISTS gold.team_trade_performance CASCADE;
DROP TABLE IF EXISTS gold.player_trade_performance CASCADE;

-- 1. ANALYTICAL TABLE: PLAYER PERFORMANCE PRE VS POST TRADE (TRADITIONAL STATS)
CREATE TABLE gold.player_trade_performance AS
WITH pre_trade_stats AS (
    SELECT 
        t.player_id,
        t.season_end_year,
        t.old_team_id,
        t.new_team_id,
        t.trade_date,
        COUNT(*) AS games_pre,
        AVG(pg.pts) AS pts_pre,
        AVG(pg.ast) AS ast_pre,
        AVG(pg.reb) AS reb_pre,
        AVG(pg.minutes) AS min_pre,
        AVG(pg.plus_minus) AS plus_minus_pre,
        AVG(pg.net_rating) AS net_rating_pre
    FROM silver.fact_player_trades t
    JOIN silver.fact_player_gamelogs pg ON t.player_id = pg.player_id AND t.old_team_id = pg.team_id
    JOIN silver.dim_games g ON pg.game_id = g.game_id
    WHERE g.game_date < t.trade_date AND g.season_end_year = t.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY t.player_id, t.season_end_year, t.old_team_id, t.new_team_id, t.trade_date
),
post_trade_stats AS (
    SELECT 
        t.player_id,
        t.season_end_year,
        t.old_team_id,
        t.new_team_id,
        t.trade_date,
        COUNT(*) AS games_post,
        AVG(pg.pts) AS pts_post,
        AVG(pg.ast) AS ast_post,
        AVG(pg.reb) AS reb_post,
        AVG(pg.minutes) AS min_post,
        AVG(pg.plus_minus) AS plus_minus_post,
        AVG(pg.net_rating) AS net_rating_post
    FROM silver.fact_player_trades t
    JOIN silver.fact_player_gamelogs pg ON t.player_id = pg.player_id AND t.new_team_id = pg.team_id
    JOIN silver.dim_games g ON pg.game_id = g.game_id
    WHERE g.game_date >= t.trade_date AND g.season_end_year = t.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY t.player_id, t.season_end_year, t.old_team_id, t.new_team_id, t.trade_date
)
SELECT 
    p.player_name,
    ot.team_abbreviation AS old_team,
    nt.team_abbreviation AS new_team,
    pre.season_end_year,
    pre.trade_date,
    pre.games_pre,
    post.games_post,
    -- Pre stats
    ROUND(pre.pts_pre::numeric, 2) AS pts_pre,
    ROUND(pre.ast_pre::numeric, 2) AS ast_pre,
    ROUND(pre.reb_pre::numeric, 2) AS reb_pre,
    ROUND(pre.min_pre::numeric, 2) AS min_pre,
    ROUND(pre.plus_minus_pre::numeric, 2) AS plus_minus_pre,
    ROUND(pre.net_rating_pre::numeric, 2) AS net_rating_pre,
    -- Post stats
    ROUND(post.pts_post::numeric, 2) AS pts_post,
    ROUND(post.ast_post::numeric, 2) AS ast_post,
    ROUND(post.reb_post::numeric, 2) AS reb_post,
    ROUND(post.min_post::numeric, 2) AS min_post,
    ROUND(post.plus_minus_post::numeric, 2) AS plus_minus_post,
    ROUND(post.net_rating_post::numeric, 2) AS net_rating_post,
    -- Differences
    ROUND((post.pts_post - pre.pts_pre)::numeric, 2) AS pts_diff,
    ROUND((post.ast_post - pre.ast_pre)::numeric, 2) AS ast_diff,
    ROUND((post.reb_post - pre.reb_pre)::numeric, 2) AS reb_diff,
    ROUND((post.min_post - pre.min_pre)::numeric, 2) AS min_diff,
    ROUND((post.plus_minus_post - pre.plus_minus_pre)::numeric, 2) AS plus_minus_diff,
    ROUND((post.net_rating_post - pre.net_rating_pre)::numeric, 2) AS net_rating_diff
FROM pre_trade_stats pre
JOIN post_trade_stats post ON pre.player_id = post.player_id AND pre.season_end_year = post.season_end_year AND pre.old_team_id = post.old_team_id AND pre.new_team_id = post.new_team_id
JOIN silver.dim_players p ON pre.player_id = p.player_id
JOIN silver.dim_teams ot ON pre.old_team_id = ot.team_id
JOIN silver.dim_teams nt ON pre.new_team_id = nt.team_id
WHERE pre.games_pre >= 5 AND post.games_post >= 5; -- filter out short-term or temporary transfers for statistical validity

-- 2. ANALYTICAL TABLE: TEAM PERFORMANCE PRE VS POST TRADE (WINS & TEAM RATINGS)
CREATE TABLE gold.team_trade_performance AS
WITH team_trades AS (
    SELECT 
        season_end_year,
        trade_date,
        old_team_id AS team_id,
        'Sent Out' AS trade_role,
        player_id
    FROM silver.fact_player_trades
    UNION ALL
    SELECT 
        season_end_year,
        trade_date,
        new_team_id AS team_id,
        'Acquired' AS trade_role,
        player_id
    FROM silver.fact_player_trades
),
team_pre_stats AS (
    SELECT 
        tt.team_id,
        tt.season_end_year,
        tt.trade_date,
        tt.trade_role,
        tt.player_id,
        COUNT(*) AS games_pre,
        SUM(CASE WHEN tg.wl = 'W' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS win_pct_pre,
        AVG(tg.pts) AS pts_scored_pre,
        AVG(tg.pts_allowed) AS pts_allowed_pre,
        AVG(tg.off_rating) AS off_rating_pre,
        AVG(tg.def_rating) AS def_rating_pre,
        AVG(tg.net_rating) AS net_rating_pre
    FROM team_trades tt
    JOIN silver.fact_team_gamelogs tg ON tt.team_id = tg.team_id
    JOIN silver.dim_games g ON tg.game_id = g.game_id
    WHERE g.game_date < tt.trade_date AND g.season_end_year = tt.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY tt.team_id, tt.season_end_year, tt.trade_date, tt.trade_role, tt.player_id
),
team_post_stats AS (
    SELECT 
        tt.team_id,
        tt.season_end_year,
        tt.trade_date,
        tt.trade_role,
        tt.player_id,
        COUNT(*) AS games_post,
        SUM(CASE WHEN tg.wl = 'W' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS win_pct_post,
        AVG(tg.pts) AS pts_scored_post,
        AVG(tg.pts_allowed) AS pts_allowed_post,
        AVG(tg.off_rating) AS off_rating_post,
        AVG(tg.def_rating) AS def_rating_post,
        AVG(tg.net_rating) AS net_rating_post
    FROM team_trades tt
    JOIN silver.fact_team_gamelogs tg ON tt.team_id = tg.team_id
    JOIN silver.dim_games g ON tg.game_id = g.game_id
    WHERE g.game_date >= tt.trade_date AND g.season_end_year = tt.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY tt.team_id, tt.season_end_year, tt.trade_date, tt.trade_role, tt.player_id
)
SELECT 
    t.team_abbreviation AS team,
    p.player_name AS traded_player,
    pre.trade_role,
    pre.season_end_year,
    pre.trade_date,
    pre.games_pre,
    post.games_post,
    -- Win PCT
    ROUND(pre.win_pct_pre::numeric, 3) AS win_pct_pre,
    ROUND(post.win_pct_post::numeric, 3) AS win_pct_post,
    ROUND((post.win_pct_post - pre.win_pct_pre)::numeric, 3) AS win_pct_diff,
    -- Ratings Pre
    ROUND(pre.off_rating_pre::numeric, 2) AS off_rating_pre,
    ROUND(pre.def_rating_pre::numeric, 2) AS def_rating_pre,
    ROUND(pre.net_rating_pre::numeric, 2) AS net_rating_pre,
    -- Ratings Post
    ROUND(post.off_rating_post::numeric, 2) AS off_rating_post,
    ROUND(post.def_rating_post::numeric, 2) AS def_rating_post,
    ROUND(post.net_rating_post::numeric, 2) AS net_rating_post,
    -- Ratings Diff
    ROUND((post.off_rating_post - pre.off_rating_pre)::numeric, 2) AS off_rating_diff,
    ROUND((post.def_rating_post - pre.def_rating_pre)::numeric, 2) AS def_rating_diff,
    ROUND((post.net_rating_post - pre.net_rating_pre)::numeric, 2) AS net_rating_diff
FROM team_pre_stats pre
JOIN team_post_stats post ON pre.team_id = post.team_id AND pre.season_end_year = post.season_end_year AND pre.trade_date = post.trade_date AND pre.trade_role = post.trade_role AND pre.player_id = post.player_id
JOIN silver.dim_teams t ON pre.team_id = t.team_id
JOIN silver.dim_players p ON pre.player_id = p.player_id
WHERE pre.games_pre >= 5 AND post.games_post >= 5; -- ensure sufficient games for comparison

-- 3. ANALYTICAL TABLE: PLAYER PERFORMANCE ADVANCED NET POINTS & WIN PROBABILITY ADDED (ESPN ADVANCED METRICS)
CREATE TABLE gold.player_trade_advanced_espn AS
WITH pre_espn AS (
    SELECT 
        t.player_id,
        t.season_end_year,
        t.old_team_id,
        t.new_team_id,
        t.trade_date,
        COUNT(*) AS games_pre,
        AVG(pb.o_net_pts) AS o_net_pts_pre,
        AVG(pb.d_net_pts) AS d_net_pts_pre,
        AVG(pb.t_net_pts) AS t_net_pts_pre,
        AVG(pb.o_usg) AS o_usg_pre,
        AVG(pb.d_usg) AS d_usg_pre,
        AVG(pb.o_wpa) AS o_wpa_pre,
        AVG(pb.d_wpa) AS d_wpa_pre,
        AVG(pb.t_wpa) AS t_wpa_pre
    FROM silver.fact_player_trades t
    JOIN silver.fact_espn_player_box pb ON t.player_id = pb.player_id AND t.old_team_id = pb.team_id
    JOIN silver.dim_games g ON pb.game_id = g.game_id
    WHERE g.game_date < t.trade_date AND g.season_end_year = t.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY t.player_id, t.season_end_year, t.old_team_id, t.new_team_id, t.trade_date
),
post_espn AS (
    SELECT 
        t.player_id,
        t.season_end_year,
        t.old_team_id,
        t.new_team_id,
        t.trade_date,
        COUNT(*) AS games_post,
        AVG(pb.o_net_pts) AS o_net_pts_post,
        AVG(pb.d_net_pts) AS d_net_pts_post,
        AVG(pb.t_net_pts) AS t_net_pts_post,
        AVG(pb.o_usg) AS o_usg_post,
        AVG(pb.d_usg) AS d_usg_post,
        AVG(pb.o_wpa) AS o_wpa_post,
        AVG(pb.d_wpa) AS d_wpa_post,
        AVG(pb.t_wpa) AS t_wpa_post
    FROM silver.fact_player_trades t
    JOIN silver.fact_espn_player_box pb ON t.player_id = pb.player_id AND t.new_team_id = pb.team_id
    JOIN silver.dim_games g ON pb.game_id = g.game_id
    WHERE g.game_date >= t.trade_date AND g.season_end_year = t.season_end_year AND g.game_type = 'Regular Season'
    GROUP BY t.player_id, t.season_end_year, t.old_team_id, t.new_team_id, t.trade_date
)
SELECT 
    p.player_name,
    ot.team_abbreviation AS old_team,
    nt.team_abbreviation AS new_team,
    pre.season_end_year,
    pre.trade_date,
    pre.games_pre,
    post.games_post,
    -- Offensive Net Points (Points contributed per 100 possessions above league average)
    ROUND(pre.o_net_pts_pre::numeric, 3) AS o_net_pts_pre,
    ROUND(post.o_net_pts_post::numeric, 3) AS o_net_pts_post,
    ROUND((post.o_net_pts_post - pre.o_net_pts_pre)::numeric, 3) AS o_net_pts_diff,
    -- Defensive Net Points (Points saved per 100 possessions above league average)
    ROUND(pre.d_net_pts_pre::numeric, 3) AS d_net_pts_pre,
    ROUND(post.d_net_pts_post::numeric, 3) AS d_net_pts_post,
    ROUND((post.d_net_pts_post - pre.d_net_pts_pre)::numeric, 3) AS d_net_pts_diff,
    -- Total Net Points
    ROUND(pre.t_net_pts_pre::numeric, 3) AS t_net_pts_pre,
    ROUND(post.t_net_pts_post::numeric, 3) AS t_net_pts_post,
    ROUND((post.t_net_pts_post - pre.t_net_pts_pre)::numeric, 3) AS t_net_pts_diff,
    -- Usage rates
    ROUND(pre.o_usg_pre::numeric, 3) AS o_usg_pre,
    ROUND(post.o_usg_post::numeric, 3) AS o_usg_post,
    ROUND(pre.d_usg_pre::numeric, 3) AS d_usg_pre,
    ROUND(post.d_usg_post::numeric, 3) AS d_usg_post,
    -- Total Win Probability Added (overall impact on team win probability)
    ROUND(pre.t_wpa_pre::numeric, 3) AS wpa_pre,
    ROUND(post.t_wpa_post::numeric, 3) AS wpa_post,
    ROUND((post.t_wpa_post - pre.t_wpa_pre)::numeric, 3) AS wpa_diff
FROM pre_espn pre
JOIN post_espn post ON pre.player_id = post.player_id AND pre.season_end_year = post.season_end_year AND pre.old_team_id = post.old_team_id AND pre.new_team_id = post.new_team_id
JOIN silver.dim_players p ON pre.player_id = p.player_id
JOIN silver.dim_teams ot ON pre.old_team_id = ot.team_id
JOIN silver.dim_teams nt ON pre.new_team_id = nt.team_id
WHERE pre.games_pre >= 5 AND post.games_post >= 5;

-- 4. ANALYTICAL TABLE: PLAYER TRADE VALUE VS SALARY (ROAS - RETURN ON ASSET SALARY)
CREATE TABLE gold.player_trade_value_analysis AS
SELECT 
    p.player_name,
    p.old_team,
    p.new_team,
    p.season_end_year,
    p.trade_date,
    p.games_pre,
    p.games_post,
    s.salary,
    s.salary_inflation_adjusted,
    s.ranking AS salary_rank_in_season,
    p.pts_diff,
    p.net_rating_diff,
    -- Calculate metrics per $1M salary
    ROUND((p.pts_diff / (s.salary / 1000000.0))::numeric, 4) AS pts_diff_per_million,
    ROUND((p.net_rating_diff / (s.salary / 1000000.0))::numeric, 4) AS net_rating_diff_per_million
FROM gold.player_trade_performance p
JOIN silver.fact_player_salaries s ON LOWER(TRIM(p.player_name)) = LOWER(TRIM(s.player_name)) AND p.season_end_year = s.season_end_year
WHERE s.salary IS NOT NULL AND s.salary > 0;

-- 5. ANALYTICAL TABLE: TEAM TRADE PERFORMANCE VS TOTAL PAYROLL (TEAM PAYROLL VALUE)
CREATE TABLE gold.team_trade_salary_impact AS
SELECT 
    tp.team,
    tp.traded_player,
    tp.trade_role,
    tp.season_end_year,
    tp.trade_date,
    tp.games_pre,
    tp.games_post,
    s.salary AS team_payroll,
    s.salary_inflation_adjusted AS team_payroll_inflation_adjusted,
    s.ranking AS team_payroll_rank_in_season,
    tp.win_pct_diff,
    tp.net_rating_diff
FROM gold.team_trade_performance tp
JOIN silver.dim_teams dt ON tp.team = dt.team_abbreviation
JOIN silver.fact_team_salaries s ON dt.team_id = s.team_id AND tp.season_end_year = s.season_end_year
WHERE s.salary IS NOT NULL AND s.salary > 0;
