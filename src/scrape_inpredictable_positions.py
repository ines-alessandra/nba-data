#!/usr/bin/env python3
"""
Scrapes inpredictable.com's season player-WPA leaderboard (ssnPlayer.php) for its
`Pos` field -- a season-long position label (C, C-F, F, F-C, F-G, G, G-F) with
full-roster coverage (verified for 2022-23: 554 rows vs. 537 players with GP>=1
in data/players_2023.parquet -- inpredictable's count is not lower, so no
meaningful roster gap).

This exists to try to close the positional-data gap left by
build_wpa_team_total_extended.py: the WPA methodology's positional breakdown
(RCP_pos/muP/deltaP) needs a per-player position with full-roster coverage,
which neither this repo's other data nor inpredictable's per-game wpBox pages
have (see that script's docstring for the full explanation).

IMPORTANT CAVEATS vs. ESPN's dAvgPos (the original per-game signal):
  - This is a single SEASON-long label per player, mixing all their games that
    season regardless of team -- NOT per-game, and NOT split per team-stint for
    players traded mid-season. (Verified: Kevin Durant, traded BKN->PHX in
    2022-23, appears as exactly ONE row combining both teams' games.) Applying
    it to a traded player's post-trade stint specifically is an approximation:
    it uses their whole-season role, not the role they specifically played
    after the trade.
  - It's a categorical label (7 buckets), not a continuous 1-5 scale, so
    bucketing to Guards/Forwards/Centers uses a documented rule (see
    POS_TO_BUCKET below) rather than the ESPN script's <=2.5/<=4.5 thresholds.

Output: data/inpredictable/ssn_player_pos_<season_end_year>.parquet
  columns: player_id, player_name, pos, games, season_end_year

Usage:
    python src/scrape_inpredictable_positions.py --seasons 2013-2025 --skip-seasons 2019-2021
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_inpredictable_wpa import HEADERS  # noqa: E402
from scrape_inpredictable_wpa_batch import parse_season_range  # noqa: E402

BASE_URL = "https://stats.inpredictable.com/nba/ssnPlayer.php"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = DATA_DIR / "inpredictable"

# Primary-position rule: bucket by the FIRST letter of inpredictable's hybrid
# label (e.g. "F-C" = primarily Forward -> Forwards). Simplest defensible rule
# for a 2-letter hybrid tag; a player's own literal Pos string is preserved
# in the output too so a different rule can be applied later without rescraping.
POS_TO_BUCKET = {
    "G": "Guards", "G-F": "Guards",
    "F": "Forwards", "F-G": "Forwards", "F-C": "Forwards",
    "C": "Centers", "C-F": "Centers",
    "": None,  # inpredictable leaves Pos blank for some players in older seasons
               # (empirically: ~19% in 2013, ~7% in 2018, 0% by 2025 -- their own
               # classifier's coverage visibly improves over time). Left as NaN
               # here and dropped downstream, exactly as wpa_common.py's own
               # assign_positions()/dropna(subset=["posicao"]) already does for
               # ESPN players with no assignable dAvgPos -- not a new kind of gap.
}


def season_date_bounds(season_end_year: int) -> tuple[str, str]:
    gl = pd.read_parquet(DATA_DIR / f"gamelog_{season_end_year}.parquet", columns=["game_date"])
    dates = pd.to_datetime(gl["game_date"])
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def fetch_page(season_end_year: int, frdt: str, todt: str, grp: int, timeout: int = 20) -> str:
    params = {
        "season": season_end_year, "team": "ALL", "pos": "ALL", "po": 0,
        "frdt": frdt, "todt": todt, "rate": "per", "sort": "sWPA", "order": "DESC", "grp": grp,
    }
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_page(html_text: str) -> pd.DataFrame:
    tree = lxml_html.fromstring(html_text)
    tables = tree.xpath("//table")
    if not tables:
        return pd.DataFrame(columns=["player_id", "player_name", "pos", "games"])
    rows = tables[0].xpath(".//tr")

    records = []
    for tr in rows:
        tds = tr.xpath("./td")
        if len(tds) < 4:
            continue  # header/section rows
        link = tds[1].xpath(".//a[contains(@href,'pid=')]")
        if not link:
            continue
        m = re.search(r"pid=(\d+)", link[0].get("href", ""))
        player_id = m.group(1) if m else None
        records.append({
            "player_id": player_id,
            "player_name": link[0].text_content().strip(),
            "pos": tds[2].text_content().strip(),
            "games": tds[3].text_content().strip(),
        })
    return pd.DataFrame(records)


def scrape_season(season_end_year: int, sleep_s: float, log) -> pd.DataFrame:
    frdt, todt = season_date_bounds(season_end_year)
    all_pages = []
    grp = 1
    while True:
        html_text = fetch_page(season_end_year, frdt, todt, grp)
        page_df = parse_page(html_text)
        if page_df.empty:
            break
        all_pages.append(page_df)
        log(f"[{season_end_year}] page {grp}: {len(page_df)} players")
        grp += 1
        time.sleep(sleep_s)

    df = pd.concat(all_pages, ignore_index=True) if all_pages else pd.DataFrame(
        columns=["player_id", "player_name", "pos", "games"])
    df["games"] = pd.to_numeric(df["games"], errors="coerce").astype("Int64")
    df["season_end_year"] = season_end_year
    unknown = sorted(set(df["pos"].unique()) - set(POS_TO_BUCKET.keys()))
    if unknown:
        raise RuntimeError(f"[{season_end_year}] Unmapped Pos value(s): {unknown} -- add to POS_TO_BUCKET.")
    df["bucket"] = df["pos"].map(POS_TO_BUCKET)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", required=True, help="e.g. '2013-2025' or '2013,2015'")
    ap.add_argument("--skip-seasons", default="", help="Season end years to exclude, same syntax")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    seasons = parse_season_range(args.seasons)
    skip = set(parse_season_range(args.skip_seasons)) if args.skip_seasons else set()
    seasons = [s for s in seasons if s not in skip]

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for season_end_year in seasons:
        df = scrape_season(season_end_year, args.sleep, log)
        out_path = OUT_DIR / f"ssn_player_pos_{season_end_year}.parquet"
        df.to_parquet(out_path, index=False)
        log(f"[{season_end_year}] saved {len(df)} player rows -> {out_path} "
            f"(bucket counts: {df['bucket'].value_counts().to_dict()})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
