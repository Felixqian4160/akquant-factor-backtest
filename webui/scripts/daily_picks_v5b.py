"""Daily picks CLI for WebUI.

Given a date, computes the 6-factor composite score for every stock in
the v2 panel on that date, and returns top-K picks with their scores.

This is the production use case: every day, run this script to get
today's bull_composite picks.

Fast: < 5 seconds per call.

Output JSON:
{
  "date": "2019-01-03",
  "factors": [...6 names...],
  "top_k": 10,
  "n_stocks_scored": 350,
  "picks": [
    {"rank": 1, "ts_code": "000001.SZ", "score": 0.92, "close": 12.34, "composite_rank": 0.98, ...},
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pick_date = date.fromisoformat(args.date)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    error_path = out_dir / "error.log"

    # Load panel slice for that date.
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .filter(pl.col("trade_date") == pick_date)
        .drop_nulls(subset=BULL_FACTORS)
        .collect()
    )
    if df.shape[0] == 0:
        # Try ± 5 days (in case pick_date is a non-trading day).
        for delta in range(1, 6):
            for sign in (1, -1):
                alt = pick_date.fromordinal(pick_date.toordinal() + sign * delta)
                df2 = (
                    pl.scan_parquet(str(PANEL))
                    .select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"] + BULL_FACTORS)
                    .with_columns(pl.col("trade_date").cast(pl.Date))
                    .filter(pl.col("trade_date") == alt)
                    .drop_nulls(subset=BULL_FACTORS)
                    .collect()
                )
                if df2.shape[0] > 0:
                    df = df2
                    pick_date = alt
                    break
            if df.shape[0] > 0:
                break

    if df.shape[0] == 0:
        err = {"error": f"no data for {args.date} (and ±5 days)"}
        error_path.write_text(json.dumps(err))
        result_path.write_text(json.dumps(err))
        print(f"ERROR: {err}")
        return 1

    n_scored = df.shape[0]
    # Convert to pandas for ranking.
    pdf = df.to_pandas()
    # Cross-section rank each factor, then average.
    mat = pdf[BULL_FACTORS].values.astype(float)
    ranks = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        order = np.argsort(col, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(col))
        ranks[:, j] = r / max(len(col) - 1, 1)
    composite = ranks.mean(axis=1)

    pdf["composite_rank"] = composite
    pdf = pdf.sort_values("composite_rank", ascending=False).reset_index(drop=True)
    top = pdf.head(args.top_k)

    picks = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        picks.append({
            "rank": rank,
            "ts_code": row["ts_code"],
            "score": round(float(row["composite_rank"]), 4),
            "close": round(float(row["close"]), 3),
            "open": round(float(row["open"]), 3),
            "high": round(float(row["high"]), 3),
            "low": round(float(row["low"]), 3),
            "vol": int(float(row["vol"])),
            "factors": {f: round(float(row[f]), 6) for f in BULL_FACTORS},
            "weight_pct": round(100.0 / args.top_k, 2),  # equal weight
        })

    out = {
        "job_id": args.job_id,
        "date_requested": args.date,
        "date_used": pick_date.isoformat(),
        "factors": BULL_FACTORS,
        "top_k": args.top_k,
        "n_stocks_scored": n_scored,
        "picks": picks,
    }
    result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {result_path}: {len(picks)} picks on {pick_date}")
    return 0


if __name__ == "__main__":
    import numpy as np
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)