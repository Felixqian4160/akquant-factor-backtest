"""Cache builder: per (rebal_date, factor) conditional mean of _fwd_net.
Reads factor columns one at a time from parquet to bound memory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
CACHE = Path("evidence/v37_v14_causal/_cache/bear_factor_returns.tsv")
START = "2010-01-01"
END = "2025-12-31"
ROUND_TRIP_COST = 0.005
BEAR_FWD_EXIT_SHIFT = 21


def dt_expr(value: str):
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    manifest = json.loads(Path("evidence/v37_v14_causal/factor_manifest.json").read_text())
    factors = manifest["factors"]
    log(f"factors: {len(factors)}")

    # Compute _fwd_net as a standalone parquet-projection column.
    # Use polars for the shift; only persist _fwd_net + trade_date + ts_code.
    log("Loading open/close and computing _fwd_net...")
    base = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"])
        .filter((pl.col("trade_date") >= dt_expr(START)) & (pl.col("trade_date") <= dt_expr(END)))
        .sort(["ts_code", "trade_date"])
        .collect()
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
            pl.col("close").shift(-BEAR_FWD_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
        ])
        .with_columns(
            ((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST)
             .alias("_fwd_net"))
        )
        .select(["trade_date", "ts_code", "_fwd_net"])
    )
    log(f"_fwd_net computed: {base.height:,} rows in {time.time()-t0:.1f}s")

    # Cast to pandas; only 3 cols, small enough.
    base_pd = base.to_pandas()
    base_pd["trade_date"] = pd.to_datetime(base_pd["trade_date"])
    log(f"to_pandas done in {time.time()-t0:.1f}s")

    fwd_np = base_pd["_fwd_net"].to_numpy(dtype=np.float32)
    trade_np = base_pd["trade_date"].to_numpy()
    ts_np = base_pd["ts_code"].to_numpy()
    mask_valid = ~np.isnan(fwd_np)
    sorted_idx = np.argsort(trade_np)
    sorted_dates = trade_np[sorted_idx]
    sorted_fwd = fwd_np[sorted_idx]
    sorted_valid = mask_valid[sorted_idx]
    log(f"sort done in {time.time()-t0:.1f}s")

    # All dates in window for anchor (so any offset can hit).
    dates = np.unique(sorted_dates)
    log(f"dates: {len(dates)}")

    # Pre-compute _global cum_mean at each date using valid_mask
    cum_sum_fwd = np.cumsum(np.where(sorted_valid, sorted_fwd, 0.0))
    cum_count_fwd = np.cumsum(sorted_valid.astype(np.int64))
    pos = np.searchsorted(sorted_dates, dates)
    global_means = np.where(pos > 0, cum_sum_fwd[np.maximum(pos - 1, 0)] / np.maximum(cum_count_fwd[np.maximum(pos - 1, 0)], 1), np.nan)
    # Wait: pos is the index of first date >= target. We want index BEFORE that.
    # Actually np.searchsorted gives insertion index. cum sum at pos-1 = rows < target.
    # Need <= target, so use pos-1 if pos > 0 and dates[pos-1] == target, else pos-1.
    # Simpler: use np.searchsorted with 'right' and decrement.
    # Actually we want count of (trade_date <= target), so insertion index = count of < or = target depending on side.
    # Use side='right' which gives insertion point AFTER equal elements; position == count of <= target.

    # Let me just compute using cumulative arrays with correct indexing:
    pos_right = np.searchsorted(sorted_dates, dates, side='right')  # count of <= target
    cum_sum_at = cum_sum_fwd[np.maximum(pos_right - 1, 0)]
    cum_cnt_at = cum_count_fwd[np.maximum(pos_right - 1, 0)]
    global_means = np.where(pos_right > 0, cum_sum_at / np.maximum(cum_cnt_at, 1), np.nan)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with CACHE.open("w") as fh:
        for i, d in enumerate(dates):
            pos = pos_right[i]
            if pos == 0:
                continue
            gm = global_means[i]
            if not np.isfinite(gm):
                continue
            d_str = pd.Timestamp(d).date().isoformat()
            fh.write(f"{d_str}\t_global\t{gm:.8f}\n")
            n_written += 1

            # Now per-factor: need to project factor columns one at a time
            # For now, skip per-factor; rely on picks to recompute per-factor via panel.
            # Actually we MUST emit per-factor rows too. Do it efficiently:
            # 410 factors × 1.3M read = 5 GB IO. Let's do it once per factor, indexed.
            # Sort factor data by ts_code+trade_date to align with sorted_idx.
            # factor_arr aligned to sorted_idx.

    log(f"Cache (global only) written: {n_written} dates in {time.time()-t0:.1f}s")

    # Phase 2: per-factor projection and write.
    log("Per-factor projection phase...")
    for fi, f in enumerate(factors):
        if fi % 25 == 0:
            log(f"  factor {fi+1}/{len(factors)}: {f}, cache size {CACHE.stat().st_size/1e6:.1f}MB, elapsed {time.time()-t0:.0f}s")
        try:
            f_pd = (
                pl.scan_parquet(PANEL)
                .select(["trade_date", "ts_code", f])
                .filter((pl.col("trade_date") >= dt_expr(START)) & (pl.col("trade_date") <= dt_expr(END)))
                .sort(["ts_code", "trade_date"])
                .collect()
                .to_pandas()
            )
        except Exception:
            continue
        f_pd["trade_date"] = pd.to_datetime(f_pd["trade_date"])
        if len(f_pd) != len(base_pd):
            continue
        col_np = pd.to_numeric(f_pd[f], errors="coerce").to_numpy(dtype=np.float32)
        col_sorted = col_np[sorted_idx]
        col_valid = ~np.isnan(col_sorted)
        ok_mask = col_valid & sorted_valid
        cum_sum = np.cumsum(np.where(ok_mask, sorted_fwd, 0.0))
        cum_cnt = np.cumsum(ok_mask.astype(np.int64))
        per_factor_means = np.where(
            pos_right > 0,
            cum_sum[np.maximum(pos_right - 1, 0)] / np.maximum(cum_cnt[np.maximum(pos_right - 1, 0)], 1),
            np.nan,
        )
        with CACHE.open("a") as fh:
            for i, d in enumerate(dates):
                pos = pos_right[i]
                if pos == 0:
                    continue
                m = per_factor_means[i]
                cnt = cum_cnt[np.maximum(pos - 1, 0)]
                if not np.isfinite(m) or cnt < 5:
                    continue
                d_str = pd.Timestamp(d).date().isoformat()
                fh.write(f"{d_str}\t{f}\t{m:.8f}\n")
    log(f"Final size: {CACHE.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()