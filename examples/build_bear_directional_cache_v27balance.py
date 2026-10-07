"""Build a causal high/low directional bear-factor score cache.

This replaces the old bear cache's conditional mean (all non-null stocks) with
an executable portfolio statistic:

  for every completed historical bear rebalance date S and factor f:
    high_mean[f,S] = mean(net20 of factor top-10 stocks at S)
    low_mean[f,S]  = mean(net20 of factor bottom-10 stocks at S)

For a current bear rebalance date T, only historical S whose T+21 exit date
is strictly before T are eligible.  The last 30 eligible bear sessions are
used.  The better high/low direction is selected only when its mean net return
exceeds 0.5% and has at least 10 observations.

The cache is versioned and isolated from the historical V37 cache.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
MANIFEST = Path("evidence/v37_v14_causal/factor_manifest.json")
OUT_ROOT = Path("evidence/v37_v14_causal_v27balance_directional_20261006")
CACHE_ROOT = OUT_ROOT / "_cache"
START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
FWD_EXIT_SHIFT = 21
ROUND_TRIP_COST = 0.005
LOOKBACK_SESSIONS = 30
TOP_N = 10
MIN_OBS = 10
MIN_SCORE = 0.005

BEAR_VOTE_THRESHOLD = 2
BEAR_LOOKBACK = 5
BEAR_CUMULATIVE_THRESHOLD = 5


def dt_expr(value: str) -> pl.Expr:
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def router_map(daily: pd.DataFrame) -> dict[str, bool]:
    """V27_BALANCE: 16 signals, 5-day cumulative V2 >= 5."""
    d = daily.copy()
    d["ma5"] = d["idx_close"].rolling(5).mean()
    d["ma10"] = d["idx_close"].rolling(10).mean()
    d["ma20"] = d["idx_close"].rolling(20).mean()
    d["ma60"] = d["idx_close"].rolling(60).mean()
    d["ma120"] = d["idx_close"].rolling(120).mean()
    d["ret10"] = d["idx_close"].pct_change(10)
    d["ret20"] = d["idx_close"].pct_change(20)
    d["ret40"] = d["idx_close"].pct_change(40)
    d["ret60"] = d["idx_close"].pct_change(60)
    d["dist_ma20"] = (d["idx_close"] - d["ma20"]) / d["ma20"]
    d["dist_ma60"] = (d["idx_close"] - d["ma60"]) / d["ma60"]
    d["slope_ma60"] = d["ma60"].diff()
    d["high_60d"] = d["idx_close"].rolling(60).max()
    d["high_120d"] = d["idx_close"].rolling(120).max()
    d["dd_60d"] = d["idx_close"] / d["high_60d"] - 1.0
    d["dd_120d"] = d["idx_close"] / d["high_120d"] - 1.0
    specs = [
        d["idx_close"] < d["ma5"],
        d["idx_close"] < d["ma10"],
        d["idx_close"] < d["ma20"],
        d["idx_close"] < d["ma60"],
        d["idx_close"] < d["ma120"],
        d["ma20"] < d["ma60"],
        d["ma60"] < d["ma120"],
        d["ret10"] < 0,
        d["ret20"] < 0,
        d["ret40"] < 0,
        d["ret60"] < 0,
        d["dist_ma20"] < -0.02,
        d["dist_ma60"] < -0.05,
        d["slope_ma60"] < 0,
        d["dd_60d"] < -0.05,
        d["dd_120d"] < -0.10,
    ]
    votes = pd.concat(specs, axis=1).fillna(False).sum(axis=1)
    v2 = (votes >= BEAR_VOTE_THRESHOLD).astype(int)
    cumulative = v2.rolling(BEAR_LOOKBACK).sum()
    d["is_bear"] = cumulative >= BEAR_CUMULATIVE_THRESHOLD
    return dict(zip(d["date"], d["is_bear"].fillna(False).astype(bool)))


def load_manifest() -> list[str]:
    return json.loads(MANIFEST.read_text())["factors"]


def load_base():
    """Full panel key + causal net20, retaining the sorted row order."""
    base = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"])
        .sort(["ts_code", "trade_date"])
        .collect()
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
            pl.col("close").shift(-FWD_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
        ])
        .with_columns(
            (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_net")
        )
        .select(["trade_date", "ts_code", "_fwd_net"])
    )
    pdf = base.to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])
    dates_np = pdf["trade_date"].to_numpy(dtype="datetime64[D]")
    fwd_np = pdf["_fwd_net"].to_numpy(dtype=np.float32)
    row_order = np.argsort(dates_np, kind="stable")
    dates_sorted = dates_np[row_order]
    fwd_sorted = fwd_np[row_order]
    return pdf, row_order, dates_sorted, fwd_sorted


def load_dates_and_router() -> tuple[list[str], dict[str, bool]]:
    daily = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "idx_close"])
        .filter(pl.col("idx_close").is_not_null())
        .unique("trade_date")
        .sort("trade_date")
        .collect()
        .to_pandas()
    )
    daily["date"] = pd.to_datetime(daily["trade_date"]).dt.date.astype(str)
    all_dates = [d for d in daily["date"].tolist() if START <= d <= END]
    return all_dates, router_map(daily[["date", "idx_close"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebal-offset", type=int, required=True)
    args = parser.parse_args()
    offset = args.rebal_offset
    if offset < 0 or offset >= REBAL_STEP:
        raise ValueError("rebal-offset must be in [0, 19]")

    t0 = time.time()
    factors = load_manifest()
    all_dates, regime = load_dates_and_router()
    rebal_dates = all_dates[offset::REBAL_STEP]
    bear_dates = [d for d in rebal_dates if regime.get(d, False)]
    date_pos = {d: i for i, d in enumerate(all_dates)}
    bear_positions = [date_pos[d] for d in bear_dates]
    # A historical signal S is complete only after its T+21 close date.
    exit_pos = {
        d: (date_pos[d] + FWD_EXIT_SHIFT if date_pos[d] + FWD_EXIT_SHIFT < len(all_dates) else None)
        for d in bear_dates
    }

    log(f"offset={offset}; factors={len(factors)}; rebalances={len(rebal_dates)}; bear_rebalances={len(bear_dates)}")
    base_pdf, row_order, dates_sorted, fwd_sorted = load_base()
    valid_fwd = np.isfinite(fwd_sorted)
    unique_dates = np.array(all_dates, dtype="datetime64[D]")
    # Boundaries in the date-sorted arrays for every panel date.
    left = np.searchsorted(dates_sorted, unique_dates, side="left")
    right = np.searchsorted(dates_sorted, unique_dates, side="right")
    date_to_bounds = {d: (int(left[i]), int(right[i])) for i, d in enumerate(all_dates)}
    log(f"base rows={len(base_pdf):,}; causal net20 ready in {time.time()-t0:.1f}s")

    n_bear = len(bear_dates)
    n_factors = len(factors)
    high_mat = np.full((n_bear, n_factors), np.nan, dtype=np.float32)
    low_mat = np.full((n_bear, n_factors), np.nan, dtype=np.float32)

    # First factor projection establishes exact row alignment.
    key_dates = base_pdf["trade_date"].to_numpy(dtype="datetime64[ns]")
    key_codes = base_pdf["ts_code"].to_numpy()
    alignment_checked = False
    for fi, factor in enumerate(factors):
        try:
            fpdf = (
                pl.scan_parquet(PANEL)
                .select(["trade_date", "ts_code", factor])
                .sort(["ts_code", "trade_date"])
                .collect()
                .to_pandas()
            )
        except Exception as exc:
            log(f"skip factor={factor}: {exc}")
            continue
        if len(fpdf) != len(base_pdf):
            log(f"skip factor={factor}: row count mismatch {len(fpdf)}")
            continue
        if not alignment_checked:
            if not np.array_equal(fpdf["trade_date"].to_numpy(dtype="datetime64[ns]"), key_dates) or not np.array_equal(fpdf["ts_code"].to_numpy(), key_codes):
                raise RuntimeError(f"row alignment mismatch at first factor {factor}")
            alignment_checked = True
        vals = pd.to_numeric(fpdf[factor], errors="coerce").to_numpy(dtype=np.float32)[row_order]
        for bi, signal_date in enumerate(bear_dates):
            l, r = date_to_bounds[signal_date]
            v = vals[l:r]
            rr = fwd_sorted[l:r]
            ok = np.isfinite(v) & np.isfinite(rr)
            valid = np.flatnonzero(ok)
            if len(valid) < TOP_N * 2:
                continue
            vv = v[valid]
            top_idx = valid[np.argpartition(vv, -TOP_N)[-TOP_N:]]
            low_idx = valid[np.argpartition(vv, TOP_N - 1)[:TOP_N]]
            high_mat[bi, fi] = float(np.mean(rr[top_idx]))
            low_mat[bi, fi] = float(np.mean(rr[low_idx]))
        if fi == 0 or (fi + 1) % 50 == 0:
            log(f"factor projections {fi+1}/{n_factors}; elapsed={time.time()-t0:.1f}s")
        del fpdf, vals

    # For each current bear date, aggregate only prior completed sessions.
    out_dir = CACHE_ROOT / f"offset_{offset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "directional_scores.tsv"
    selected_path = out_dir / "selected_directions.tsv"
    selected_rows = []
    score_rows = []
    direction_counter = {"high": 0, "low": 0}
    current_to_bear_index = {d: i for i, d in enumerate(bear_dates)}
    for bi, current_date in enumerate(bear_dates):
        cur_pos = date_pos[current_date]
        eligible = [j for j, d in enumerate(bear_dates) if j < bi and exit_pos.get(d) is not None and exit_pos[d] < cur_pos]
        eligible = eligible[-LOOKBACK_SESSIONS:]
        if not eligible:
            continue
        h = high_mat[eligible, :]
        lo = low_mat[eligible, :]
        with np.errstate(invalid="ignore"):
            h_mean = np.nanmean(h, axis=0)
            l_mean = np.nanmean(lo, axis=0)
        h_n = np.sum(np.isfinite(h), axis=0)
        l_n = np.sum(np.isfinite(lo), axis=0)
        for fi, factor in enumerate(factors):
            high_score = float(h_mean[fi]) if np.isfinite(h_mean[fi]) else np.nan
            low_score = float(l_mean[fi]) if np.isfinite(l_mean[fi]) else np.nan
            score_rows.append((current_date, factor, high_score, low_score, int(h_n[fi]), int(l_n[fi]), len(eligible)))
            candidates = []
            if h_n[fi] >= MIN_OBS and np.isfinite(high_score):
                candidates.append(("high", high_score))
            if l_n[fi] >= MIN_OBS and np.isfinite(low_score):
                candidates.append(("low", low_score))
            if not candidates:
                continue
            direction, score = max(candidates, key=lambda x: x[1])
            if score > MIN_SCORE:
                selected_rows.append((current_date, factor, direction, score, high_score, low_score, int(h_n[fi]), int(l_n[fi]), len(eligible)))
                direction_counter[direction] += 1

    with score_path.open("w") as fh:
        fh.write("current_date\tfactor\thigh_mean\tlow_mean\thigh_n\tlow_n\thistory_n\n")
        for row in score_rows:
            fh.write("\t".join(map(str, row)) + "\n")
    with selected_path.open("w") as fh:
        fh.write("current_date\tfactor\tdirection\tselected_score\thigh_mean\tlow_mean\thigh_n\tlow_n\thistory_n\n")
        for row in selected_rows:
            fh.write("\t".join(map(str, row)) + "\n")
    metadata = {
        "rebal_offset": offset,
        "router": "V27_BALANCE",
        "router_params": {"vote_threshold": BEAR_VOTE_THRESHOLD, "lookback": BEAR_LOOKBACK, "cumulative_threshold": BEAR_CUMULATIVE_THRESHOLD},
        "factor_count": n_factors,
        "rebal_dates": len(rebal_dates),
        "bear_rebal_dates": len(bear_dates),
        "directional_score_rows": len(score_rows),
        "selected_rows": len(selected_rows),
        "selected_direction_counts": direction_counter,
        "lookback_sessions": LOOKBACK_SESSIONS,
        "min_observations": MIN_OBS,
        "min_factor_mean_return": MIN_SCORE,
        "forward_contract": "T+1 raw open -> T+21 close minus 0.5% round-trip cost",
        "leakage_guard": "historical session exit position strictly before current bear date",
        "artifacts": {"directional_scores": str(score_path), "selected_directions": str(selected_path)},
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    log(f"done: selected={len(selected_rows)}, directions={direction_counter}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
