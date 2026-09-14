#!/usr/bin/env python3
"""
Batch-scrapes inpredictable.com per-player WPA box scores for every regular
season game in a range of seasons, using scrape_inpredictable_wpa.py's parser
for each individual game page.

Game list source: data/gamelog_<season_end_year>.parquet (already in this repo),
which has one row per team per game -- deduplicated here to one row per game_id
with its date, giving exactly the (game_id, date, season_end_year) triple
scrape_inpredictable_wpa.scrape_game() needs. No separate game-ID discovery step.

Resumable by design: for each season, output is written to
  data/inpredictable/wpa_player_box_<season_end_year>.parquet
  data/inpredictable/wpa_game_meta_<season_end_year>.parquet
On every run, games whose game_id is already present in that season's player-box
parquet are skipped, so interrupting and re-running the same command just
continues where it left off. Progress is flushed to disk every --flush-every
games (default 50), not only at the end of a season, so a kill/crash mid-season
loses at most one flush interval of work.

Failures (HTTP errors, parse errors, games with no inpredictable data e.g. very
early-season edge cases) are appended to data/inpredictable/_failed_games.csv
with the error message, and the batch continues -- one bad game never aborts
the run. Re-run with --retry-failed to re-attempt only those.

Usage:
    # one season, to sanity check before committing to the full range
    python src/scrape_inpredictable_wpa_batch.py --seasons 2013

    # full range, skipping seasons already covered by ESPN's real WPA data
    python src/scrape_inpredictable_wpa_batch.py --seasons 2013-2025 --skip-seasons 2019-2021

    # retry only games that failed on a previous run
    python src/scrape_inpredictable_wpa_batch.py --seasons 2013-2025 --skip-seasons 2019-2021 --retry-failed
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_inpredictable_wpa import fetch_page, parse_wpa_box  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
GAMELOG_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "data" / "inpredictable"
FAILED_PATH = OUT_DIR / "_failed_games.csv"


def parse_season_range(spec: str) -> list[int]:
    """Parses '2013-2025' or '2013' or '2013,2015,2020' into a sorted list of years."""
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            years.update(range(int(lo), int(hi) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def load_game_list(season_end_year: int) -> pd.DataFrame:
    """One row per game_id for the season: game_id, game_date (YYYY-MM-DD str), season_end_year."""
    path = GAMELOG_DIR / f"gamelog_{season_end_year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No gamelog file for season {season_end_year}: {path}")
    df = pd.read_parquet(path, columns=["game_id", "game_date"])
    df = df.drop_duplicates(subset="game_id").copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["season_end_year"] = season_end_year
    return df.sort_values("game_date").reset_index(drop=True)


def load_scraped_game_ids(season_end_year: int) -> set[str]:
    out_path = OUT_DIR / f"wpa_player_box_{season_end_year}.parquet"
    if not out_path.exists():
        return set()
    return set(pd.read_parquet(out_path, columns=["game_id"])["game_id"].unique())


def load_failed_game_ids(season_end_year: int) -> set[str]:
    if not FAILED_PATH.exists():
        return set()
    df = pd.read_csv(FAILED_PATH, dtype={"game_id": str})
    return set(df.loc[df["season_end_year"] == season_end_year, "game_id"])


def log_failure(game_id: str, date: str, season_end_year: int, error: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not FAILED_PATH.exists()
    with open(FAILED_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["game_id", "game_date", "season_end_year", "error", "scraped_at"])
        w.writerow([game_id, date, season_end_year, error, datetime.now(timezone.utc).isoformat()])


def clear_failures_for(game_ids: set[str]) -> None:
    """Drops rows for the given game_ids from the failures log (called after a retry succeeds)."""
    if not FAILED_PATH.exists() or not game_ids:
        return
    df = pd.read_csv(FAILED_PATH, dtype={"game_id": str})
    df = df[~df["game_id"].isin(game_ids)]
    df.to_csv(FAILED_PATH, index=False)


def flush_season(
    season_end_year: int,
    player_buffer: list[pd.DataFrame],
    meta_buffer: list[dict],
) -> None:
    if not player_buffer and not meta_buffer:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    player_path = OUT_DIR / f"wpa_player_box_{season_end_year}.parquet"
    new_players = pd.concat(player_buffer, ignore_index=True) if player_buffer else pd.DataFrame()
    if not new_players.empty:
        if player_path.exists():
            new_players = pd.concat([pd.read_parquet(player_path), new_players], ignore_index=True)
        new_players = new_players.drop_duplicates(subset=["game_id", "player_name", "team"])
        new_players.to_parquet(player_path, index=False)

    meta_path = OUT_DIR / f"wpa_game_meta_{season_end_year}.parquet"
    new_meta = pd.DataFrame(meta_buffer)
    if not new_meta.empty:
        if meta_path.exists():
            new_meta = pd.concat([pd.read_parquet(meta_path), new_meta], ignore_index=True)
        new_meta = new_meta.drop_duplicates(subset=["game_id"])
        new_meta.to_parquet(meta_path, index=False)


def scrape_season(
    season_end_year: int,
    sleep_s: float,
    flush_every: int,
    retry_failed: bool,
    max_retries: int,
    log,
    limit: int | None = None,
) -> None:
    games = load_game_list(season_end_year)
    already_done = load_scraped_game_ids(season_end_year)
    already_failed = set() if retry_failed else load_failed_game_ids(season_end_year)
    skip_ids = already_done | already_failed

    pending = games[~games["game_id"].isin(skip_ids)]
    total = len(games)
    if limit is not None:
        pending = pending.head(limit)
    log(f"[{season_end_year}] {total} games total, {len(already_done)} already scraped, "
        f"{len(already_failed)} previously failed (skipped), {len(pending)} to fetch")

    if pending.empty:
        return

    player_buffer: list[pd.DataFrame] = []
    meta_buffer: list[dict] = []
    retried_ok: set[str] = set()
    t_start = time.time()

    for i, row in enumerate(pending.itertuples(index=False), start=1):
        game_id, date, _ = row.game_id, row.game_date, row.season_end_year
        err = None
        for attempt in range(max_retries + 1):
            try:
                html_text = fetch_page(game_id, date, season_end_year)
                df, meta = parse_wpa_box(html_text)
                if df.empty:
                    raise ValueError("parsed 0 player rows")
                df.insert(0, "game_id", game_id)
                df.insert(1, "game_date", date)
                meta["game_id"] = game_id
                meta["game_date"] = date
                player_buffer.append(df)
                meta_buffer.append(meta)
                if retry_failed:
                    retried_ok.add(game_id)
                err = None
                break
            except Exception as e:  # noqa: BLE001 -- one bad game must not kill the batch
                err = str(e)
                if attempt < max_retries:
                    time.sleep(sleep_s * (2 ** attempt))
        if err is not None:
            log_failure(game_id, date, season_end_year, err)
            log(f"[{season_end_year}] FAILED {game_id} ({date}): {err}")

        if i % flush_every == 0 or i == len(pending):
            flush_season(season_end_year, player_buffer, meta_buffer)
            player_buffer, meta_buffer = [], []
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(pending) - i) / rate / 60 if rate > 0 else float("nan")
            log(f"[{season_end_year}] {i}/{len(pending)} done "
                f"({rate:.2f} games/s, ETA {eta_min:.1f} min)")

        time.sleep(sleep_s)

    if retried_ok:
        clear_failures_for(retried_ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", required=True, help="Season end years, e.g. '2013-2025' or '2013,2015'")
    ap.add_argument("--skip-seasons", default="", help="Season end years to exclude, same syntax")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests (default 1.0)")
    ap.add_argument("--flush-every", type=int, default=50, help="Write progress to disk every N games")
    ap.add_argument("--max-retries", type=int, default=2, help="Retries per game before logging as failed")
    ap.add_argument("--retry-failed", action="store_true",
                     help="Only re-attempt games already logged in _failed_games.csv for these seasons")
    ap.add_argument("--log-file", help="Also append progress lines to this file")
    ap.add_argument("--limit", type=int, help="Cap games per season (for smoke-testing)")
    args = ap.parse_args()

    seasons = parse_season_range(args.seasons)
    skip = set(parse_season_range(args.skip_seasons)) if args.skip_seasons else set()
    seasons = [s for s in seasons if s not in skip]

    log_fh = open(args.log_file, "a") if args.log_file else None

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, file=sys.stderr, flush=True)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    log(f"Starting batch: seasons={seasons} (skipped={sorted(skip)}) sleep={args.sleep}s "
        f"retry_failed={args.retry_failed}")

    for season_end_year in seasons:
        scrape_season(
            season_end_year,
            sleep_s=args.sleep,
            flush_every=args.flush_every,
            retry_failed=args.retry_failed,
            max_retries=args.max_retries,
            log=log,
            limit=args.limit,
        )

    log("Batch complete.")
    if log_fh:
        log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
