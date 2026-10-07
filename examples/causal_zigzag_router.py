"""Causal ZigZag regime router.

Uses the ZigZag pivots from hs300_index_pivots.json as ground truth, but
applies a STRICT causal discipline:

  bull leg (valley→peak):  regime becomes bull_neutral at leg.start (the
                             valley date is fully realized on that day)
  bear leg (peak→valley):  regime becomes bear at leg.start (the peak
                             date is fully realized on that day)
  default:                  bull_neutral (no completed ZigZag leg yet)

The leg.end (the unconfirmed side — peak/valley that may reverse) is
NOT used as a regime trigger. This means the router keeps regime in
effect until a NEW leg.start appears, which is a conservative causal
approximation:

  - bull_neutral starts at valley, continues until next peak (when a
    new bear leg.start fires)
  - bear starts at peak, continues until next valley

The router does NOT look into future pivots to "shorten" regime. If
the ground truth ZigZag has a bull leg from 2010-07 to 2015-06 (peak
yet to come), the router marks the entire 2010-07 onward window as
bull_neutral — even though we don't know 2015 is the peak yet.

For legs that are < 100 days the router ignores them (same threshold
the ground truth file uses).
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

PIVOTS_PATH = "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/hs300_index_pivots_clean_20260919/hs300_index_pivots.json"
PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
MIN_LEG_DAYS = 100


def load_truth() -> list[tuple[date, date, str]]:
    """Load ZigZag legs as (start_date, end_date, kind) tuples.
    kind = 'bull' if leg is valley→peak, else 'bear'.
    Only legs with duration_days >= MIN_LEG_DAYS are included.
    """
    data = json.loads(Path(PIVOTS_PATH).read_text())
    legs = []
    for leg in data["legs"]:
        if leg["duration_days"] < MIN_LEG_DAYS:
            continue
        start = date.fromisoformat(leg["start"])
        end = date.fromisoformat(leg["end"])
        kind = "bull" if leg["kind"] == "up" else "bear"
        legs.append((start, end, kind))
    legs.sort(key=lambda x: x[0])
    return legs


def build_router() -> dict[date, str]:
    """Map each date to its ZigZag-causal regime.

    Returns dict: {date_iso: 'bear' | 'bull_neutral'}

    Algorithm:
      - Sort legs by start date.
      - For each rebal-relevant date, walk legs and find the most
        recent leg whose START ≤ date. Use that leg's kind.
      - default = bull_neutral (before first valley).
    """
    legs = load_truth()
    panel_daily = (
        pl.scan_parquet(PANEL)
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    regime: dict[date, str] = {}
    leg_idx = 0
    for ts in panel_daily:
        d = ts.date()
        # Advance leg_idx to the rightmost leg with start ≤ d
        while leg_idx + 1 < len(legs) and legs[leg_idx + 1][0] <= d:
            leg_idx += 1
        if leg_idx < len(legs) and legs[leg_idx][0] <= d:
            kind = legs[leg_idx][2]
            regime[d] = "bear" if kind == "bear" else "bull_neutral"
        else:
            # Before first leg start → default bull_neutral
            regime[d] = "bull_neutral"
    return regime


def main() -> None:
    t0 = time.time()
    regime = build_router()
    n_bear = sum(1 for v in regime.values() if v == "bear")
    n_bull = sum(1 for v in regime.values() if v == "bull_neutral")
    print(f"Total dates: {len(regime)}")
    print(f"  bear: {n_bear} ({n_bear/len(regime)*100:.1f}%)")
    print(f"  bull_neutral: {n_bull} ({n_bull/len(regime)*100:.1f}%)")
    print(f"Built in {time.time()-t0:.1f}s")

    # Per-year distribution
    panel_daily = (
        pl.scan_parquet(PANEL)
        .select("trade_date")
        .filter((pl.col("trade_date") >= pl.datetime(2010, 1, 1)) &
                (pl.col("trade_date") <= pl.datetime(2025, 12, 31)))
        .unique()
        .sort("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    yearly: dict[int, dict[str, int]] = {}
    for ts in panel_daily:
        d = ts.date()
        kind = regime.get(d, "bull_neutral")
        y = d.year
        yearly.setdefault(y, {"bear": 0, "bull_neutral": 0})[kind] += 1
    print("\nYearly distribution (causal ZigZag router):")
    print(f"{'year':>6} {'dates':>7} {'bear':>7} {'bull_neutral':>13}")
    for y in sorted(yearly.keys()):
        c = yearly[y]
        total = c["bear"] + c["bull_neutral"]
        print(f"  {y} | {total:>5} | {c['bear']:>3} ({c['bear']/total*100:>5.1f}%) | "
              f"{c['bull_neutral']:>3} ({c['bull_neutral']/total*100:>5.1f}%)")

    # Save router map as JSON for picks builder to consume
    out_path = Path("evidence/causal_zigzag_router_20261006/router_map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {d.isoformat(): r for d, r in regime.items()},
        indent=1
    ))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
