#!/usr/bin/env python3
"""
Extends build_wpa_trade_analysis.py's H1 (Preenche Lacuna vs Redundante) and H2
(Objetivo Sazonal) trade-classification pipeline from 3 seasons (2019-2021,
ESPN-only) to 13 seasons (2013-2025), using wpa_extended_common.load_extended_player_box()
in place of wpa_common.load_espn_player_box(). The classification/impact logic
itself (run_analysis(), in build_wpa_trade_analysis.py) is completely unchanged --
this script only supplies a different player_box and a wider season_range.

KNOWN LIMITATION carried over from wpa_extended_common.py: for 2013-2018, a
player's position is a season-static label (not per-game/per-window), so a
trade's pre-trade deltaP for that player's position group uses the same bucket
regardless of how close to the trade date the window starts -- there's no
possibility of detecting an in-season positional shift for those years (there
wasn't one to detect in this data source anyway: see
scrape_inpredictable_positions.py's empirical finding that the label never
changes, trade or no trade). WPA_time (delta_WPA_time, the impact measure) is
NOT affected by this -- it's a real per-game sum in every era.

data/wpa_pretrade_clusters.csv (pre-trade competitive archetype) still only
covers 2019-2021 at time of writing, so `cluster`/`cluster_name` will be NaN
for trade events outside that window in this script's output -- a separate,
not-yet-done follow-up (extending build_wpa_pretrade_clusters.py the same way).

Outputs:
  data/wpa_trade_classification_extended.csv
  data/wpa_trade_impact_analysis_extended.csv

Usage:
    python src/build_wpa_trade_analysis_extended.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import get_engine, PROJECT_DIR  # noqa: E402
from wpa_extended_common import load_extended_player_box  # noqa: E402
from build_wpa_trade_analysis import run_analysis  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
CLASSIFICATION_PATH = DATA_DIR / "wpa_trade_classification_extended.csv"
IMPACT_PATH = DATA_DIR / "wpa_trade_impact_analysis_extended.csv"
PRETRADE_CLUSTERS_PATH = DATA_DIR / "wpa_pretrade_clusters.csv"  # still 2019-2021 only, see docstring

SEASON_RANGE = range(2013, 2026)


def main():
    engine = get_engine()
    print("Loading extended player_box (2013-2025: ESPN-native + hybrid + static-position)...")
    player_box = load_extended_player_box(engine)
    print(f"  {len(player_box):,} player-game rows across "
          f"{sorted(player_box['season_end_year'].unique())}.\n")

    run_analysis(
        engine, player_box, SEASON_RANGE,
        CLASSIFICATION_PATH, IMPACT_PATH, PRETRADE_CLUSTERS_PATH,
        label="(2013-2025, extended)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
