#!/usr/bin/env python3
"""
Scrapes per-player Win Probability Added (WPA) box scores from inpredictable.com
(stats.inpredictable.com/nba/wpBox.php) for a single NBA game.

inpredictable's `gid` URL parameter is the NBA's own game_id (e.g. "0022200121"),
matching the `gameId` column in data/player_game_logs.parquet, and `date` is the
game date (YYYY-MM-DD). So games to scrape can be sourced directly from existing
data (see gamelog_row_to_url()) -- no separate game-ID discovery step is needed.

This is a *validation* scraper: it currently extracts only the player-level WPA
summary table (WPA, eWPA, clWPA, gbWPA, TS%, tsWPA, TO, toWPA, FTA, FT%, ftWPA)
plus game metadata (score, excitement, comeback, MVP, LVP), which is the same
shape of stat produced by ESPN's tWPA field that data/espn/player_box.parquet
(consumed by src/wpa_common.py) is built from. It intentionally does NOT scrape
the full play-by-play win-probability curve.

Usage:
    python src/scrape_inpredictable_wpa.py --game-id 0022200121 --date 2022-11-04 \
        --season 2023 --out data/inpredictable/wpa_player_box_0022200121.csv

Be polite to the source site: this script makes one request per game and is
meant to be run at small scale (single games / manual spot checks) until the
output schema and join-to-NBA-IDs logic have been validated end to end.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import html as lxml_html

BASE_URL = "https://stats.inpredictable.com/nba/wpBox.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; nba-data-research-bot/1.0; "
        "academic use, low volume; contact: ines.alessandra03@gmail.com)"
    )
}

# Column order of the <td>s in each player row of the div#box table, after the
# player-name cell and after the sparkline cell has been filtered out. Note the
# sparkline <td> (a <span class="awayspk"/"homespk">) sits *after* the WPA
# value for away rows but *before* it for home rows -- position alone can't be
# trusted, so it must be filtered out by content, not by index (see below).
_STAT_COLS = [
    "WPA", "eWPA", "clWPA", "gbWPA",
    "FGA", "TS_pct", "tsWPA",
    "TO", "toWPA",
    "FTA", "FT_pct", "ftWPA",
]


def gamelog_row_to_url(game_id: str, date: str, season_end_year: int) -> str:
    """Builds the wpBox.php URL for a game, given the same identifiers already
    present in data/player_game_logs.parquet (gameId, date -> season_end_year)."""
    return (
        f"{BASE_URL}?season={season_end_year}&month=Select+Month"
        f"&date={date}&gid={game_id}"
    )


def fetch_page(game_id: str, date: str, season_end_year: int, timeout: int = 20) -> str:
    url = gamelog_row_to_url(game_id, date, season_end_year)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _team_abbrevs(tree) -> tuple[str, str]:
    """Extracts (away_abbrev, home_abbrev) from the team logo <img> srcs in the
    score header, in away-then-home document order."""
    imgs = tree.xpath('//div[@id="score"]//img[contains(@class,"nbalogo")]/@src')
    abbrevs = [re.search(r"teamlogos/nba/500/([A-Z]+)\.png", src) for src in imgs]
    abbrevs = [m.group(1) for m in abbrevs if m]
    if len(abbrevs) < 2:
        raise ValueError("Could not find both team abbreviations in div#score")
    return abbrevs[0], abbrevs[1]


def _game_meta(tree, away_abbrev: str, home_abbrev: str) -> dict:
    score_cells = tree.xpath('//div[@id="score"]//td')
    text = [c.text_content().strip() for c in score_cells]
    meta = {"away_team": away_abbrev, "home_team": home_abbrev}
    # away score / home score are the first numeric-only cells for each row
    scores = [t for t in text if t.isdigit()]
    if len(scores) >= 2:
        meta["away_score"], meta["home_score"] = int(scores[0]), int(scores[1])
    labels = {"Excitement": "excitement", "Comeback": "comeback", "MVP": "mvp", "LVP": "lvp"}
    for i, t in enumerate(text):
        if t in labels and i + 1 < len(text):
            meta[labels[t]] = text[i + 1]
    return meta


def _player_id_map(tree) -> dict[str, str]:
    """Maps player display name -> NBA person_id, from the on/off table's
    onoff_player.php?pid=<id> links (div#onoff)."""
    mapping = {}
    for a in tree.xpath('//div[@id="onoff"]//a[contains(@href,"pid=")]'):
        m = re.search(r"pid=(\d+)", a.get("href", ""))
        if m:
            mapping[a.text_content().strip()] = m.group(1)
    return mapping


def parse_wpa_box(html_text: str) -> tuple[pd.DataFrame, dict]:
    """Parses the player-WPA tables (div#box) and game meta out of a wpBox.php page.

    Returns (player_df, game_meta_dict).
    """
    tree = lxml_html.fromstring(html_text)

    away_abbrev, home_abbrev = _team_abbrevs(tree)
    meta = _game_meta(tree, away_abbrev, home_abbrev)
    pid_map = _player_id_map(tree)

    # The page's raw HTML nests <table> tags without closing them properly
    # (<table><tbody><table>...), which libxml2's HTML auto-correction resolves
    # inconsistently -- div#id boundaries (and even some table boundaries) can
    # end up swallowing unrelated sections (on/off table, "similar games" list).
    # Anchoring on "which table has a <th>Player</th> header" is more precise,
    # but the wrapper table produced by the broken nesting also matches (it
    # contains the real table as a descendant), so those wrappers must be
    # excluded explicitly: keep only tables with no such table among their
    # descendants (i.e. leaf matches).
    candidates = tree.xpath('//table[.//th[normalize-space(text())="Player"]]')
    player_tables = [
        t for t in candidates
        if not t.xpath('.//table[.//th[normalize-space(text())="Player"]]')
    ]
    if not player_tables:
        raise ValueError("No player WPA table found -- page layout may have changed, or game has no data")

    records = []
    for table in player_tables:
        team_header = table.xpath('.//th[@class="team"]')
        if not team_header:
            continue
        header_tr = team_header[0].getparent()
        team = away_abbrev if "away" in (header_tr.get("class") or "") else home_abbrev

        for tr in table.xpath(".//tr"):
            tds = tr.xpath("./td")
            if not tds or tr.xpath("./th"):
                continue  # team-header row or column-header row (th Player/WPA/...)
            name = tds[0].text_content().strip()
            if not name:
                continue
            record = {"team": team, "player_name": name, "player_id": pid_map.get(name)}
            # Drop the sparkline cell by content (its position relative to the
            # WPA value flips between away/home rows), keeping only scalar cells.
            stat_tds = [td for td in tds[1:] if not td.xpath(".//span[contains(@class,'spk')]")]
            for col, td in zip(_STAT_COLS, stat_tds):
                val = td.text_content().strip()
                if col.endswith("_pct"):
                    record[col] = float(val.rstrip("%")) / 100.0 if val.rstrip("%") else None
                else:
                    try:
                        record[col] = float(val)
                    except ValueError:
                        record[col] = val
            records.append(record)

    df = pd.DataFrame.from_records(records)
    return df, meta


def scrape_game(game_id: str, date: str, season_end_year: int, sleep_s: float = 1.0) -> tuple[pd.DataFrame, dict]:
    html_text = fetch_page(game_id, date, season_end_year)
    time.sleep(sleep_s)  # be polite if this is called in a loop by future batch code
    df, meta = parse_wpa_box(html_text)
    df.insert(0, "game_id", game_id)
    df.insert(1, "game_date", date)
    return df, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-id", required=True, help="NBA game_id, e.g. 0022200121")
    ap.add_argument("--date", required=True, help="Game date, YYYY-MM-DD")
    ap.add_argument("--season", required=True, type=int, help="Season end year, e.g. 2023 for 2022-23")
    ap.add_argument("--out", help="Output CSV path. Defaults to stdout preview only.")
    args = ap.parse_args()

    df, meta = scrape_game(args.game_id, args.date, args.season)

    print(f"Game meta: {meta}", file=sys.stderr)
    print(df.to_string(index=False), file=sys.stderr)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
