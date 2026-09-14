-- ====================================================================
-- SILVER SCHEMA: CLEANED & MODELLED RELATIONAL TABLES
-- ====================================================================

-- Dropping existing tables if any, with CASCADE to ensure clean recreation
DROP TABLE IF EXISTS silver.fact_player_trades CASCADE;
DROP TABLE IF EXISTS silver.fact_player_season_stats CASCADE;
DROP TABLE IF EXISTS silver.fact_player_gamelogs CASCADE;
DROP TABLE IF EXISTS silver.fact_team_gamelogs CASCADE;
DROP TABLE IF EXISTS silver.fact_espn_player_details CASCADE;
DROP TABLE IF EXISTS silver.fact_espn_player_box CASCADE;
DROP TABLE IF EXISTS silver.fact_espn_team_box CASCADE;
DROP TABLE IF EXISTS silver.fact_espn_four_factors CASCADE;
DROP TABLE IF EXISTS silver.fact_player_salaries CASCADE;
DROP TABLE IF EXISTS silver.fact_team_salaries CASCADE;
DROP TABLE IF EXISTS silver.dim_games CASCADE;
DROP TABLE IF EXISTS silver.dim_players CASCADE;
DROP TABLE IF EXISTS silver.dim_teams CASCADE;

-- 1. DIMENSION: TEAMS
CREATE TABLE silver.dim_teams (
    team_id INT PRIMARY KEY,
    team_abbreviation VARCHAR(10) NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    team_city VARCHAR(100),
    team_tricode VARCHAR(10),
    team_slug VARCHAR(100)
);

-- Populating DIM_TEAMS
INSERT INTO silver.dim_teams (team_id, team_abbreviation, team_name, team_city, team_tricode, team_slug)
WITH all_teams AS (
    SELECT 
        team_id,
        team_abbreviation,
        team_name,
        NULL AS team_city,
        team_abbreviation AS team_tricode,
        NULL AS team_slug,
        1 AS source_priority
    FROM bronze.gamelogs
    WHERE team_id IS NOT NULL
    
    UNION ALL
    
    SELECT 
        "teamId" AS team_id,
        "teamTricode" AS team_abbreviation,
        "teamName" AS team_name,
        "teamCity" AS team_city,
        "teamTricode" AS team_tricode,
        "teamSlug" AS team_slug,
        2 AS source_priority
    FROM bronze.player_game_logs
    WHERE "teamId" IS NOT NULL
    
    UNION ALL
    
    SELECT 
        team_id,
        "tmName" AS team_abbreviation,
        "tmName" AS team_name,
        NULL AS team_city,
        "tmName" AS team_tricode,
        NULL AS team_slug,
        3 AS source_priority
    FROM bronze.espn_team_box
    WHERE team_id IS NOT NULL
    
    UNION ALL
    
    SELECT 
        CAST(team_id AS INT) AS team_id,
        team AS team_abbreviation,
        team AS team_name,
        NULL AS team_city,
        team AS team_tricode,
        NULL AS team_slug,
        4 AS source_priority
    FROM bronze.espn_player_box
    WHERE team_id IS NOT NULL
),
deduped_teams AS (
    SELECT DISTINCT ON (team_id)
        team_id,
        team_abbreviation,
        team_name,
        team_city,
        team_tricode,
        team_slug
    FROM all_teams
    ORDER BY team_id, CASE WHEN team_abbreviation IS NULL OR team_abbreviation = '' THEN 1 ELSE 0 END, source_priority ASC
)
SELECT 
    team_id,
    team_abbreviation,
    team_name,
    team_city,
    team_tricode,
    team_slug
FROM deduped_teams
ON CONFLICT (team_id) DO UPDATE SET
    team_abbreviation = EXCLUDED.team_abbreviation,
    team_name = EXCLUDED.team_name,
    team_city = COALESCE(EXCLUDED.team_city, dim_teams.team_city),
    team_tricode = COALESCE(EXCLUDED.team_tricode, dim_teams.team_tricode),
    team_slug = COALESCE(EXCLUDED.team_slug, dim_teams.team_slug);

-- 2. DIMENSION: PLAYERS
CREATE TABLE silver.dim_players (
    player_id INT PRIMARY KEY,
    player_name VARCHAR(150) NOT NULL,
    first_name VARCHAR(100),
    family_name VARCHAR(100),
    player_slug VARCHAR(150),
    position VARCHAR(10)
);

-- Populating DIM_PLAYERS
INSERT INTO silver.dim_players (player_id, player_name, first_name, family_name, player_slug, position)
WITH all_players AS (
    SELECT 
        "personId" AS player_id,
        COALESCE("firstName" || ' ' || "familyName", '') AS player_name,
        "firstName" AS first_name,
        "familyName" AS family_name,
        "playerSlug" AS player_slug,
        "position" AS position,
        1 AS source_priority
    FROM bronze.player_game_logs
    WHERE "personId" IS NOT NULL
    
    UNION ALL
    
    SELECT 
        player_id,
        player_name,
        SPLIT_PART(player_name, ' ', 1) AS first_name,
        SUBSTRING(player_name FROM POSITION(' ' IN player_name)+1) AS family_name,
        NULL AS player_slug,
        NULL AS position,
        2 AS source_priority
    FROM bronze.playerstats
    WHERE player_id IS NOT NULL
    
    UNION ALL
    
    SELECT 
        player_id,
        name AS player_name,
        SPLIT_PART(name, ' ', 1) AS first_name,
        SUBSTRING(name FROM POSITION(' ' IN name)+1) AS family_name,
        NULL AS player_slug,
        NULL AS position,
        3 AS source_priority
    FROM bronze.espn_player_box
    WHERE player_id IS NOT NULL
    
    UNION ALL
    
    SELECT 
        "playerId" AS player_id,
        name AS player_name,
        SPLIT_PART(name, ' ', 1) AS first_name,
        SUBSTRING(name FROM POSITION(' ' IN name)+1) AS family_name,
        NULL AS player_slug,
        NULL AS position,
        4 AS source_priority
    FROM bronze.espn_player_details
    WHERE "playerId" IS NOT NULL
),
deduped_players AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        player_name,
        first_name,
        family_name,
        player_slug,
        position
    FROM all_players
    ORDER BY player_id, CASE WHEN player_name IS NULL OR player_name = '' THEN 1 ELSE 0 END, source_priority ASC
)
SELECT 
    player_id,
    player_name,
    first_name,
    family_name,
    player_slug,
    position
FROM deduped_players
ON CONFLICT (player_id) DO UPDATE SET
    player_name = EXCLUDED.player_name,
    first_name = COALESCE(EXCLUDED.first_name, dim_players.first_name),
    family_name = COALESCE(EXCLUDED.family_name, dim_players.family_name),
    player_slug = COALESCE(EXCLUDED.player_slug, dim_players.player_slug),
    position = COALESCE(EXCLUDED.position, dim_players.position);

-- 3. DIMENSION: GAMES
CREATE TABLE silver.dim_games (
    game_id VARCHAR(20) PRIMARY KEY,
    game_date DATE NOT NULL,
    season_year VARCHAR(10) NOT NULL,
    season_end_year INT NOT NULL,
    game_type VARCHAR(50) NOT NULL
);

-- Populating DIM_GAMES
INSERT INTO silver.dim_games (game_id, game_date, season_year, season_end_year, game_type)
WITH all_game_ids AS (
    SELECT game_id, CAST(game_date AS DATE) AS game_date, season_year, CAST(SPLIT_PART(season_year, '-', 1) AS INT) + 1 AS season_end_year
    FROM bronze.gamelogs
    WHERE game_id IS NOT NULL
    
    UNION
    
    SELECT "gameId" AS game_id, NULL::DATE AS game_date, NULL::VARCHAR AS season_year, NULL::INT AS season_end_year
    FROM bronze.player_game_logs
    WHERE "gameId" IS NOT NULL
    
    UNION
    
    SELECT "gameId" AS game_id, NULL::DATE AS game_date, NULL::VARCHAR AS season_year, NULL::INT AS season_end_year
    FROM bronze.espn_four_factors
    WHERE "gameId" IS NOT NULL
    
    UNION
    
    SELECT "game_id", NULL::DATE AS game_date, NULL::VARCHAR AS season_year, NULL::INT AS season_end_year
    FROM bronze.espn_team_box
    WHERE "game_id" IS NOT NULL
    
    UNION
    
    SELECT "game_id", NULL::DATE AS game_date, NULL::VARCHAR AS season_year, NULL::INT AS season_end_year
    FROM bronze.espn_player_box
    WHERE "game_id" IS NOT NULL
    
    UNION
    
    SELECT "gameId" AS game_id, NULL::DATE AS game_date, NULL::VARCHAR AS season_year, NULL::INT AS season_end_year
    FROM bronze.espn_player_details
    WHERE "gameId" IS NOT NULL
),
deduped_games AS (
    SELECT DISTINCT ON (game_id)
        game_id,
        game_date,
        season_year,
        season_end_year
    FROM all_game_ids
    ORDER BY game_id, game_date DESC NULLS LAST
)
SELECT
    game_id,
    -- Fallback date calculation for games missing from team gamelogs:
    -- Decodes the year from game_id (e.g. 0022400123 -> season end 2025 -> playoffs in May, play-ins in April)
    COALESCE(
        game_date,
        CASE 
            WHEN SUBSTRING(game_id FROM 1 FOR 3) = '004' THEN TO_DATE((2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT) + 1)::TEXT || '-05-01', 'YYYY-MM-DD')
            WHEN SUBSTRING(game_id FROM 1 FOR 3) = '005' THEN TO_DATE((2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT) + 1)::TEXT || '-04-15', 'YYYY-MM-DD')
            ELSE TO_DATE((2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT) + 1)::TEXT || '-01-01', 'YYYY-MM-DD')
        END
    ) AS game_date,
    COALESCE(
        season_year,
        (2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT))::TEXT || '-' || SUBSTRING((2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT) + 1)::TEXT FROM 3 FOR 2)
    ) AS season_year,
    COALESCE(
        season_end_year,
        2000 + CAST(SUBSTRING(game_id FROM 4 FOR 2) AS INT) + 1
    ) AS season_end_year,
    CASE SUBSTRING(game_id FROM 1 FOR 3)
        WHEN '001' THEN 'Preseason'
        WHEN '002' THEN 'Regular Season'
        WHEN '003' THEN 'All-Star'
        WHEN '004' THEN 'Playoffs'
        WHEN '005' THEN 'Play-In'
        WHEN '006' THEN 'NBA Cup'
        ELSE 'Other'
    END AS game_type
FROM deduped_games
ON CONFLICT (game_id) DO UPDATE SET
    game_date = EXCLUDED.game_date,
    season_year = EXCLUDED.season_year,
    season_end_year = EXCLUDED.season_end_year,
    game_type = EXCLUDED.game_type;

-- 4. FACT TABLE: TEAM GAMELOGS
CREATE TABLE silver.fact_team_gamelogs (
    game_id VARCHAR(20) REFERENCES silver.dim_games(game_id),
    team_id INT REFERENCES silver.dim_teams(team_id),
    opponent_team_id INT REFERENCES silver.dim_teams(team_id),
    is_home BOOLEAN NOT NULL,
    wl CHAR(1),
    pts INT,
    pts_allowed INT,
    possessions FLOAT,
    off_rating FLOAT,
    def_rating FLOAT,
    net_rating FLOAT,
    minutes FLOAT,
    fgm INT,
    fga INT,
    fg_pct FLOAT,
    fg3m INT,
    fg3a INT,
    fg3_pct FLOAT,
    ftm INT,
    fta INT,
    ft_pct FLOAT,
    oreb INT,
    dreb INT,
    reb INT,
    ast INT,
    tov INT,
    stl INT,
    blk INT,
    pf INT,
    plus_minus FLOAT,
    PRIMARY KEY (game_id, team_id)
);

-- Populating FACT_TEAM_GAMELOGS
INSERT INTO silver.fact_team_gamelogs (
    game_id, team_id, opponent_team_id, is_home, wl, pts, pts_allowed, possessions,
    off_rating, def_rating, net_rating, minutes, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
    ftm, fta, ft_pct, oreb, dreb, reb, ast, tov, stl, blk, pf, plus_minus
)
SELECT 
    l.game_id,
    l.team_id,
    opp.team_id AS opponent_team_id,
    CASE WHEN l.matchup LIKE '% vs. %' THEN TRUE ELSE FALSE END AS is_home,
    l.wl,
    l.pts,
    opp.pts AS pts_allowed,
    l.poss AS possessions,
    l.off_rating,
    l.def_rating,
    l.net_rating,
    l.min AS minutes,
    l.fgm,
    l.fga,
    l.fg_pct,
    l.fg3m,
    l.fg3a,
    l.fg3_pct,
    l.ftm,
    l.fta,
    l.ft_pct,
    l.oreb,
    l.dreb,
    l.reb,
    l.ast,
    l.tov,
    l.stl,
    l.blk,
    l.pf,
    l.plus_minus
FROM bronze.gamelogs l
LEFT JOIN bronze.gamelogs opp ON l.game_id = opp.game_id AND l.team_id != opp.team_id
ON CONFLICT (game_id, team_id) DO UPDATE SET
    opponent_team_id = EXCLUDED.opponent_team_id,
    is_home = EXCLUDED.is_home,
    wl = EXCLUDED.wl,
    pts = EXCLUDED.pts,
    pts_allowed = EXCLUDED.pts_allowed,
    possessions = EXCLUDED.possessions,
    off_rating = EXCLUDED.off_rating,
    def_rating = EXCLUDED.def_rating,
    net_rating = EXCLUDED.net_rating,
    minutes = EXCLUDED.minutes,
    fgm = EXCLUDED.fgm,
    fga = EXCLUDED.fga,
    fg_pct = EXCLUDED.fg_pct,
    fg3m = EXCLUDED.fg3m,
    fg3a = EXCLUDED.fg3a,
    fg3_pct = EXCLUDED.fg3_pct,
    ftm = EXCLUDED.ftm,
    fta = EXCLUDED.fta,
    ft_pct = EXCLUDED.ft_pct,
    oreb = EXCLUDED.oreb,
    dreb = EXCLUDED.dreb,
    reb = EXCLUDED.reb,
    ast = EXCLUDED.ast,
    tov = EXCLUDED.tov,
    stl = EXCLUDED.stl,
    blk = EXCLUDED.blk,
    pf = EXCLUDED.pf,
    plus_minus = EXCLUDED.plus_minus;

-- 5. FACT TABLE: PLAYER GAMELOGS
CREATE TABLE silver.fact_player_gamelogs (
    game_id VARCHAR(20) REFERENCES silver.dim_games(game_id),
    player_id INT REFERENCES silver.dim_players(player_id),
    team_id INT REFERENCES silver.dim_teams(team_id),
    minutes FLOAT,
    pts INT,
    ast INT,
    reb INT,
    oreb INT,
    dreb INT,
    stl INT,
    blk INT,
    tov INT,
    pf INT,
    fgm INT,
    fga INT,
    fg_pct FLOAT,
    fg3m INT,
    fg3a INT,
    fg3_pct FLOAT,
    ftm INT,
    fta INT,
    ft_pct FLOAT,
    plus_minus FLOAT,
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    net_rating FLOAT,
    PRIMARY KEY (game_id, player_id)
);

-- Populating FACT_PLAYER_GAMELOGS
INSERT INTO silver.fact_player_gamelogs (
    game_id, player_id, team_id, minutes, pts, ast, reb, oreb, dreb, stl, blk, tov, pf,
    fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus,
    offensive_rating, defensive_rating, net_rating
)
SELECT DISTINCT ON ("gameId", "personId")
    "gameId" AS game_id,
    "personId" AS player_id,
    "teamId" AS team_id,
    -- Safely parsing minutes string "37:28" to fractional float minutes
    CASE 
        WHEN "minutes" IS NULL OR "minutes" = '' OR "minutes" = 'None' THEN 0.0
        WHEN "minutes" LIKE '%:%' THEN 
            CAST(SPLIT_PART("minutes", ':', 1) AS FLOAT) + CAST(SPLIT_PART("minutes", ':', 2) AS FLOAT)/60.0
        ELSE CASE 
                 WHEN CAST("minutes" AS FLOAT) < 0 THEN 0.0
                 ELSE CAST("minutes" AS FLOAT)
             END
    END AS minutes,
    "points" AS pts,
    "assists" AS ast,
    "reboundsTotal" AS reb,
    "reboundsOffensive" AS oreb,
    "reboundsDefensive" AS dreb,
    "steals" AS stl,
    "blocks" AS blk,
    "turnovers" AS tov,
    "foulsPersonal" AS pf,
    "fieldGoalsMade" AS fgm,
    "fieldGoalsAttempted" AS fga,
    "fieldGoalsPercentage" AS fg_pct,
    "threePointersMade" AS fg3m,
    "threePointersAttempted" AS fg3a,
    "threePointersPercentage" AS fg3_pct,
    "freeThrowsMade" AS ftm,
    "freeThrowsAttempted" AS fta,
    "freeThrowsPercentage" AS ft_pct,
    "plusMinusPoints" AS plus_minus,
    "offensiveRating" AS offensive_rating,
    "defensiveRating" AS defensive_rating,
    "netRating" AS net_rating
FROM bronze.player_game_logs
WHERE "personId" IS NOT NULL AND "gameId" IS NOT NULL AND "teamId" IS NOT NULL
ORDER BY "gameId", "personId"
ON CONFLICT (game_id, player_id) DO UPDATE SET
    team_id = EXCLUDED.team_id,
    minutes = EXCLUDED.minutes,
    pts = EXCLUDED.pts,
    ast = EXCLUDED.ast,
    reb = EXCLUDED.reb,
    oreb = EXCLUDED.oreb,
    dreb = EXCLUDED.dreb,
    stl = EXCLUDED.stl,
    blk = EXCLUDED.blk,
    tov = EXCLUDED.tov,
    pf = EXCLUDED.pf,
    fgm = EXCLUDED.fgm,
    fga = EXCLUDED.fga,
    fg_pct = EXCLUDED.fg_pct,
    fg3m = EXCLUDED.fg3m,
    fg3a = EXCLUDED.fg3a,
    fg3_pct = EXCLUDED.fg3_pct,
    ftm = EXCLUDED.ftm,
    fta = EXCLUDED.fta,
    ft_pct = EXCLUDED.ft_pct,
    plus_minus = EXCLUDED.plus_minus,
    offensive_rating = EXCLUDED.offensive_rating,
    defensive_rating = EXCLUDED.defensive_rating,
    net_rating = EXCLUDED.net_rating;

-- 6. FACT TABLE: PLAYER SEASON STATS
CREATE TABLE silver.fact_player_season_stats (
    player_id INT REFERENCES silver.dim_players(player_id),
    team_id INT REFERENCES silver.dim_teams(team_id),
    season_end_year INT,
    age FLOAT,
    gp INT,
    w INT,
    l INT,
    w_pct FLOAT,
    minutes_played FLOAT,
    pts INT,
    pts_per_game FLOAT,
    ast INT,
    ast_per_game FLOAT,
    reb INT,
    reb_per_game FLOAT,
    tov INT,
    stl INT,
    blk INT,
    pf INT,
    fgm INT,
    fga INT,
    fg_pct FLOAT,
    fg3m INT,
    fg3a INT,
    fg3_pct FLOAT,
    ftm INT,
    fta INT,
    ft_pct FLOAT,
    plus_minus FLOAT,
    nba_fantasy_pts FLOAT,
    double_doubles INT,
    triple_doubles INT,
    PRIMARY KEY (player_id, team_id, season_end_year)
);

-- Populating FACT_PLAYER_SEASON_STATS
INSERT INTO silver.fact_player_season_stats (
    player_id, team_id, season_end_year, age, gp, w, l, w_pct, minutes_played, pts,
    pts_per_game, ast, ast_per_game, reb, reb_per_game, tov, stl, blk, pf, fgm, fga,
    fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus, nba_fantasy_pts,
    double_doubles, triple_doubles
)
SELECT
    player_id,
    team_id,
    year AS season_end_year,
    age,
    gp,
    w,
    l,
    w_pct,
    "min" AS minutes_played,
    pts,
    pts_pergame AS pts_per_game,
    ast,
    ast_pergame AS ast_per_game,
    reb,
    reb_pergame AS reb_per_game,
    tov,
    stl,
    blk,
    pf,
    fgm,
    fga,
    fg_pct,
    fg3m,
    fg3a,
    fg3_pct,
    ftm,
    fta,
    ft_pct,
    plus_minus,
    nba_fantasy_pts,
    dd2 AS double_doubles,
    td3 AS triple_doubles
FROM bronze.playerstats
WHERE player_id IS NOT NULL AND team_id IS NOT NULL
ON CONFLICT (player_id, team_id, season_end_year) DO UPDATE SET
    age = EXCLUDED.age,
    gp = EXCLUDED.gp,
    w = EXCLUDED.w,
    l = EXCLUDED.l,
    w_pct = EXCLUDED.w_pct,
    minutes_played = EXCLUDED.minutes_played,
    pts = EXCLUDED.pts,
    pts_per_game = EXCLUDED.pts_per_game,
    ast = EXCLUDED.ast,
    ast_per_game = EXCLUDED.ast_per_game,
    reb = EXCLUDED.reb,
    reb_per_game = EXCLUDED.reb_per_game,
    tov = EXCLUDED.tov,
    stl = EXCLUDED.stl,
    blk = EXCLUDED.blk,
    pf = EXCLUDED.pf,
    fgm = EXCLUDED.fgm,
    fga = EXCLUDED.fga,
    fg_pct = EXCLUDED.fg_pct,
    fg3m = EXCLUDED.fg3m,
    fg3a = EXCLUDED.fg3a,
    fg3_pct = EXCLUDED.fg3_pct,
    ftm = EXCLUDED.ftm,
    fta = EXCLUDED.fta,
    ft_pct = EXCLUDED.ft_pct,
    plus_minus = EXCLUDED.plus_minus,
    nba_fantasy_pts = EXCLUDED.nba_fantasy_pts,
    double_doubles = EXCLUDED.double_doubles,
    triple_doubles = EXCLUDED.triple_doubles;

-- 7. DETECTED MID-SEASON PLAYER TRADES
CREATE TABLE silver.fact_player_trades (
    player_id INT REFERENCES silver.dim_players(player_id),
    season_end_year INT,
    old_team_id INT REFERENCES silver.dim_teams(team_id),
    new_team_id INT REFERENCES silver.dim_teams(team_id),
    trade_date DATE NOT NULL,
    last_game_with_old_team DATE NOT NULL,
    PRIMARY KEY (player_id, season_end_year, old_team_id, new_team_id)
);

-- Populating FACT_PLAYER_TRADES
INSERT INTO silver.fact_player_trades (player_id, season_end_year, old_team_id, new_team_id, trade_date, last_game_with_old_team)
WITH player_games_ordered AS (
    SELECT 
        pg.player_id,
        g.season_end_year,
        pg.team_id,
        g.game_date,
        LAG(pg.team_id) OVER(PARTITION BY pg.player_id, g.season_end_year ORDER BY g.game_date) AS prev_team_id,
        LAG(g.game_date) OVER(PARTITION BY pg.player_id, g.season_end_year ORDER BY g.game_date) AS prev_game_date
    FROM silver.fact_player_gamelogs pg
    JOIN silver.dim_games g ON pg.game_id = g.game_id
    WHERE g.game_type = 'Regular Season'
),
trade_events AS (
    SELECT 
        player_id,
        season_end_year,
        prev_team_id AS old_team_id,
        team_id AS new_team_id,
        game_date AS trade_date,
        prev_game_date AS last_game_with_old_team
    FROM player_games_ordered
    WHERE prev_team_id IS NOT NULL AND prev_team_id != team_id
)
SELECT DISTINCT ON (player_id, season_end_year, old_team_id, new_team_id)
    player_id,
    season_end_year,
    old_team_id,
    new_team_id,
    trade_date,
    last_game_with_old_team
FROM trade_events
ORDER BY player_id, season_end_year, old_team_id, new_team_id, trade_date
ON CONFLICT (player_id, season_end_year, old_team_id, new_team_id) DO UPDATE SET
    trade_date = EXCLUDED.trade_date,
    last_game_with_old_team = EXCLUDED.last_game_with_old_team;

-- ====================================================================
-- ESPN OPTIONAL ADVANCED STATISTICS (SILVER INGESTION)
-- ====================================================================

-- 8. ESPN FACT TABLE: FOUR FACTORS
CREATE TABLE silver.fact_espn_four_factors (
    game_id VARCHAR(20) REFERENCES silver.dim_games(game_id),
    team_abbreviation VARCHAR(10),
    season_end_year INT,
    fg2_o_sc_poss FLOAT,
    fg2_o_poss FLOAT,
    fg2_o_pts_prod FLOAT,
    fg2_o_net_pts FLOAT,
    fg3_o_sc_poss FLOAT,
    fg3_o_poss FLOAT,
    fg3_o_pts_prod FLOAT,
    fg3_o_net_pts FLOAT,
    ft_o_sc_poss FLOAT,
    ft_o_poss FLOAT,
    ft_o_pts_prod FLOAT,
    ft_o_net_pts FLOAT,
    rebound_o_sc_poss FLOAT,
    rebound_o_poss FLOAT,
    rebound_o_pts_prod FLOAT,
    rebound_o_net_pts FLOAT,
    turnover_o_sc_poss FLOAT,
    turnover_o_poss FLOAT,
    turnover_o_pts_prod FLOAT,
    turnover_o_net_pts FLOAT,
    PRIMARY KEY (game_id, team_abbreviation)
);

INSERT INTO silver.fact_espn_four_factors (
    game_id, team_abbreviation, season_end_year,
    fg2_o_sc_poss, fg2_o_poss, fg2_o_pts_prod, fg2_o_net_pts,
    fg3_o_sc_poss, fg3_o_poss, fg3_o_pts_prod, fg3_o_net_pts,
    ft_o_sc_poss, ft_o_poss, ft_o_pts_prod, ft_o_net_pts,
    rebound_o_sc_poss, rebound_o_poss, rebound_o_pts_prod, rebound_o_net_pts,
    turnover_o_sc_poss, turnover_o_poss, turnover_o_pts_prod, turnover_o_net_pts
)
SELECT 
    "gameId" AS game_id,
    "team" AS team_abbreviation,
    "season" AS season_end_year,
    "2pt_oScPoss" AS fg2_o_sc_poss,
    "2pt_oPoss" AS fg2_o_poss,
    "2pt_oPtsProd" AS fg2_o_pts_prod,
    "2pt_oNetPts" AS fg2_o_net_pts,
    "3pt_oScPoss" AS fg3_o_sc_poss,
    "3pt_oPoss" AS fg3_o_poss,
    "3pt_oPtsProd" AS fg3_o_pts_prod,
    "3pt_oNetPts" AS fg3_o_net_pts,
    "freethrow_oScPoss" AS ft_o_sc_poss,
    "freethrow_oPoss" AS ft_o_poss,
    "freethrow_oPtsProd" AS ft_o_pts_prod,
    "freethrow_oNetPts" AS ft_o_net_pts,
    "rebound_oScPoss" AS rebound_o_sc_poss,
    "rebound_oPoss" AS rebound_o_poss,
    "rebound_oPtsProd" AS rebound_o_pts_prod,
    "rebound_oNetPts" AS rebound_o_net_pts,
    "turnover_oScPoss" AS turnover_o_sc_poss,
    "turnover_oPoss" AS turnover_o_poss,
    "turnover_oPtsProd" AS turnover_o_pts_prod,
    "turnover_oNetPts" AS turnover_o_net_pts
FROM bronze.espn_four_factors
ON CONFLICT (game_id, team_abbreviation) DO UPDATE SET
    season_end_year = EXCLUDED.season_end_year,
    fg2_o_sc_poss = EXCLUDED.fg2_o_sc_poss,
    fg2_o_poss = EXCLUDED.fg2_o_poss,
    fg2_o_pts_prod = EXCLUDED.fg2_o_pts_prod,
    fg2_o_net_pts = EXCLUDED.fg2_o_net_pts,
    fg3_o_sc_poss = EXCLUDED.fg3_o_sc_poss,
    fg3_o_poss = EXCLUDED.fg3_o_poss,
    fg3_o_pts_prod = EXCLUDED.fg3_o_pts_prod,
    fg3_o_net_pts = EXCLUDED.fg3_o_net_pts,
    ft_o_sc_poss = EXCLUDED.ft_o_sc_poss,
    ft_o_poss = EXCLUDED.ft_o_poss,
    ft_o_pts_prod = EXCLUDED.ft_o_pts_prod,
    ft_o_net_pts = EXCLUDED.ft_o_net_pts,
    rebound_o_sc_poss = EXCLUDED.rebound_o_sc_poss,
    rebound_o_poss = EXCLUDED.rebound_o_poss,
    rebound_o_pts_prod = EXCLUDED.rebound_o_pts_prod,
    rebound_o_net_pts = EXCLUDED.rebound_o_net_pts,
    turnover_o_sc_poss = EXCLUDED.turnover_o_sc_poss,
    turnover_o_poss = EXCLUDED.turnover_o_poss,
    turnover_o_pts_prod = EXCLUDED.turnover_o_pts_prod,
    turnover_o_net_pts = EXCLUDED.turnover_o_net_pts;

-- 9. ESPN FACT TABLE: TEAM BOX
CREATE TABLE silver.fact_espn_team_box (
    game_id VARCHAR(20) REFERENCES silver.dim_games(game_id),
    team_id INT REFERENCES silver.dim_teams(team_id),
    season_end_year INT,
    is_home BOOLEAN,
    team_abbreviation VARCHAR(10),
    pts INT,
    opp_pts INT,
    win INT,
    tot_poss FLOAT,
    opp_poss FLOAT,
    efg FLOAT,
    fg2p FLOAT,
    fg3p FLOAT,
    ftr FLOAT,
    net_pts_2s FLOAT,
    net_pts_3s FLOAT,
    net_pts_shooting FLOAT,
    net_pts_turnover FLOAT,
    net_pts_rebound FLOAT,
    net_pts_freethrow FLOAT,
    PRIMARY KEY (game_id, team_id)
);

INSERT INTO silver.fact_espn_team_box (
    game_id, team_id, season_end_year, is_home, team_abbreviation, pts, opp_pts, win,
    tot_poss, opp_poss, efg, fg2p, fg3p, ftr, net_pts_2s, net_pts_3s, net_pts_shooting,
    net_pts_turnover, net_pts_rebound, net_pts_freethrow
)
SELECT 
    "game_id",
    "team_id",
    "season" AS season_end_year,
    CASE WHEN "homeTm" = 1 THEN TRUE ELSE FALSE END AS is_home,
    "tmName" AS team_abbreviation,
    "pts",
    "oppPts" AS opp_pts,
    "win",
    "totPoss" AS tot_poss,
    "oppPoss" AS opp_poss,
    "eFG" AS efg,
    "fg2p",
    "fg3p",
    "ftr",
    "netPts2s" AS net_pts_2s,
    "netPts3s" AS net_pts_3s,
    "netPtsShooting" AS net_pts_shooting,
    "netPtsTurnover" AS net_pts_turnover,
    "netPtsRebound" AS net_pts_rebound,
    "netPtsFreethrow" AS net_pts_freethrow
FROM bronze.espn_team_box
ON CONFLICT (game_id, team_id) DO UPDATE SET
    season_end_year = EXCLUDED.season_end_year,
    is_home = EXCLUDED.is_home,
    team_abbreviation = EXCLUDED.team_abbreviation,
    pts = EXCLUDED.pts,
    opp_pts = EXCLUDED.opp_pts,
    win = EXCLUDED.win,
    tot_poss = EXCLUDED.tot_poss,
    opp_poss = EXCLUDED.opp_poss,
    efg = EXCLUDED.efg,
    fg2p = EXCLUDED.fg2p,
    fg3p = EXCLUDED.fg3p,
    ftr = EXCLUDED.ftr,
    net_pts_2s = EXCLUDED.net_pts_2s,
    net_pts_3s = EXCLUDED.net_pts_3s,
    net_pts_shooting = EXCLUDED.net_pts_shooting,
    net_pts_turnover = EXCLUDED.net_pts_turnover,
    net_pts_rebound = EXCLUDED.net_pts_rebound,
    net_pts_freethrow = EXCLUDED.net_pts_freethrow;

-- 10. ESPN FACT TABLE: PLAYER BOX
CREATE TABLE silver.fact_espn_player_box (
    game_id VARCHAR(20) REFERENCES silver.dim_games(game_id),
    player_id INT REFERENCES silver.dim_players(player_id),
    team_id INT REFERENCES silver.dim_teams(team_id),
    season_end_year INT,
    is_home BOOLEAN,
    is_starter BOOLEAN,
    minutes_played FLOAT,
    played BOOLEAN,
    pts INT,
    plus_minus_points INT,
    o_net_pts FLOAT,
    d_net_pts FLOAT,
    t_net_pts FLOAT,
    o_usg FLOAT,
    d_usg FLOAT,
    o_wpa FLOAT,
    d_wpa FLOAT,
    t_wpa FLOAT,
    PRIMARY KEY (game_id, player_id)
);

INSERT INTO silver.fact_espn_player_box (
    game_id, player_id, team_id, season_end_year, is_home, is_starter, minutes_played, played,
    pts, plus_minus_points, o_net_pts, d_net_pts, t_net_pts, o_usg, d_usg, o_wpa, d_wpa, t_wpa
)
SELECT 
    "game_id",
    "player_id",
    CAST("team_id" AS INT) AS team_id,
    "season" AS season_end_year,
    CASE WHEN "home" = 1 THEN TRUE ELSE FALSE END AS is_home,
    CASE WHEN "starter" = 1 THEN TRUE ELSE FALSE END AS is_starter,
    -- Handle minutes_played formatting (MM:SS) to float minutes
    CASE 
        WHEN "minutes_played" IS NULL OR "minutes_played" = '' THEN 0.0
        WHEN "minutes_played" LIKE '%:%' THEN 
            CAST(SPLIT_PART("minutes_played", ':', 1) AS FLOAT) + CAST(SPLIT_PART("minutes_played", ':', 2) AS FLOAT)/60.0
        ELSE CAST("minutes_played" AS FLOAT)
    END AS minutes_played,
    CASE WHEN "played" = 1 THEN TRUE ELSE FALSE END AS played,
    "pts",
    "plusMinusPoints" AS plus_minus_points,
    "oNetPts" AS o_net_pts,
    "dNetPts" AS d_net_pts,
    "tNetPts" AS t_net_pts,
    "oUsg" AS o_usg,
    "dUsg" AS d_usg,
    "oWPA" AS o_wpa,
    "dWPA" AS d_wpa,
    "tWPA" AS t_wpa
FROM bronze.espn_player_box
WHERE "player_id" IS NOT NULL AND "game_id" IS NOT NULL
ON CONFLICT (game_id, player_id) DO UPDATE SET
    team_id = EXCLUDED.team_id,
    season_end_year = EXCLUDED.season_end_year,
    is_home = EXCLUDED.is_home,
    is_starter = EXCLUDED.is_starter,
    minutes_played = EXCLUDED.minutes_played,
    played = EXCLUDED.played,
    pts = EXCLUDED.pts,
    plus_minus_points = EXCLUDED.plus_minus_points,
    o_net_pts = EXCLUDED.o_net_pts,
    d_net_pts = EXCLUDED.d_net_pts,
    t_net_pts = EXCLUDED.t_net_pts,
    o_usg = EXCLUDED.o_usg,
    d_usg = EXCLUDED.d_usg,
    o_wpa = EXCLUDED.o_wpa,
    d_wpa = EXCLUDED.d_wpa,
    t_wpa = EXCLUDED.t_wpa;

-- ====================================================================
-- INDEX CREATION FOR OPTIMIZED SILVER QUERY PERFORMANCE
-- ====================================================================
CREATE INDEX IF NOT EXISTS idx_games_date ON silver.dim_games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON silver.dim_games(season_end_year);
CREATE INDEX IF NOT EXISTS idx_team_gamelogs_team ON silver.fact_team_gamelogs(team_id);
CREATE INDEX IF NOT EXISTS idx_player_gamelogs_player ON silver.fact_player_gamelogs(player_id);
CREATE INDEX IF NOT EXISTS idx_player_gamelogs_team ON silver.fact_player_gamelogs(team_id);
CREATE INDEX IF NOT EXISTS idx_player_trades_player ON silver.fact_player_trades(player_id);
CREATE INDEX IF NOT EXISTS idx_player_trades_date ON silver.fact_player_trades(trade_date);

-- 11. FACT TABLE: PLAYER SALARIES
CREATE TABLE silver.fact_player_salaries (
    player_id INT REFERENCES silver.dim_players(player_id),
    player_name VARCHAR(150) NOT NULL,
    season_end_year INT NOT NULL,
    salary NUMERIC,
    salary_inflation_adjusted NUMERIC,
    ranking INT,
    PRIMARY KEY (player_name, season_end_year)
);

INSERT INTO silver.fact_player_salaries (player_id, player_name, season_end_year, salary, salary_inflation_adjusted, ranking)
SELECT DISTINCT ON (s.player_name, s.season_end_year)
    p.player_id,
    s.player_name,
    s.season_end_year,
    s.salary,
    s.salary_inflation_adjusted,
    s.ranking
FROM bronze.raw_player_salaries s
LEFT JOIN silver.dim_players p ON LOWER(TRIM(p.player_name)) = LOWER(TRIM(s.player_name))
LEFT JOIN silver.fact_player_season_stats ps ON p.player_id = ps.player_id AND s.season_end_year = ps.season_end_year
ORDER BY s.player_name, s.season_end_year, (ps.player_id IS NOT NULL) DESC, s.salary DESC NULLS LAST
ON CONFLICT (player_name, season_end_year) DO UPDATE SET
    player_id = EXCLUDED.player_id,
    salary = EXCLUDED.salary,
    salary_inflation_adjusted = EXCLUDED.salary_inflation_adjusted,
    ranking = EXCLUDED.ranking;

-- 12. FACT TABLE: TEAM SALARIES
CREATE TABLE silver.fact_team_salaries (
    team_id INT REFERENCES silver.dim_teams(team_id),
    team_name VARCHAR(100) NOT NULL,
    season_end_year INT NOT NULL,
    salary NUMERIC,
    salary_inflation_adjusted NUMERIC,
    ranking INT,
    PRIMARY KEY (team_name, season_end_year)
);

INSERT INTO silver.fact_team_salaries (team_id, team_name, season_end_year, salary, salary_inflation_adjusted, ranking)
SELECT DISTINCT ON (s.team_name, s.season_end_year)
    t.team_id,
    s.team_name,
    s.season_end_year,
    s.salary,
    s.salary_inflation_adjusted,
    s.ranking
FROM bronze.raw_team_salaries s
LEFT JOIN silver.dim_teams t ON 
    LOWER(t.team_name) LIKE LOWER(s.team_name) || '%' 
    OR (s.team_name = 'LA Lakers' AND t.team_abbreviation = 'LAL')
    OR (s.team_name = 'LA Clippers' AND t.team_abbreviation = 'LAC')
ORDER BY s.team_name, s.season_end_year, t.team_id NULLS LAST
ON CONFLICT (team_name, season_end_year) DO UPDATE SET
    team_id = EXCLUDED.team_id,
    salary = EXCLUDED.salary,
    salary_inflation_adjusted = EXCLUDED.salary_inflation_adjusted,
    ranking = EXCLUDED.ranking;

-- Indexes for salaries
CREATE INDEX IF NOT EXISTS idx_player_salaries_id ON silver.fact_player_salaries(player_id);
CREATE INDEX IF NOT EXISTS idx_team_salaries_id ON silver.fact_team_salaries(team_id);
