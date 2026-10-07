"""V19 + V32 fusion picks generation (用 V27 16-signal router 替代简单 router).

合同:
- Bear router: V27 16 单日信号 + 10 日累计 ≥ 4
- Bear picks = V19 voting ∪ V32 voting (union with score)
  * V19 voting: factor-return historical ranking (top-10 factors × top-10 stocks)
  * V32 voting: static pool 10 因子 (l_ami/gtja_144/l_size/l_size3/gtja_132/l_dtvm/r_tv/gtja_070/gtja_095/talib_NATR)
- bear date 触发时: V19 ∪ V32 picks, 至少一个 source 投票 >= MIN_VOTES
- MIN_STOCKS=5, MAX_STOCKS=20
- T+21 raw open exit, 0.5% round-trip cost

Output: bear_fusion_picks.json keyed by date string.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")

# V32 static pool
V32_POOL = {
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}
V32_K_NOM = 10
V32_MIN_VOTES = 3
V32_MIN_STOCKS = 5
V32_MAX_STOCKS = 20

# V19 params (best variant from earlier sweep)
V19_LOOKBACK = 30
V19_MIN_FACTOR_MEAN_RETURN = 0.005
V19_TOP_FACTORS = 10
V19_TOP_STOCKS_PER_FACTOR = 10
V19_MIN_VOTES = 3
V19_MIN_STOCKS = 5
V19_MAX_STOCKS = 10
V19_REBAL_STEP = 20  # bear sessions between rebalances

# V27 router
V27_VOTE_THRESHOLD = 2
V27_LOOKBACK = 10
V27_CUMULATIVE_THRESHOLD = 4
BEAR_SIGNAL_SPECS = [
    ("close_lt_MA5", lambda df: df["idx_close"] < df["ma5"]),
    ("close_lt_MA10", lambda df: df["idx_close"] < df["ma10"]),
    ("close_lt_MA20", lambda df: df["idx_close"] < df["ma20"]),
    ("close_lt_MA60", lambda df: df["idx_close"] < df["ma60"]),
    ("close_lt_MA120", lambda df: df["idx_close"] < df["ma120"]),
    ("MA20_lt_MA60", lambda df: df["ma20"] < df["ma60"]),
    ("MA60_lt_MA120", lambda df: df["ma60"] < df["ma120"]),
    ("ret10_lt_0", lambda df: df["ret10"] < 0),
    ("ret20_lt_0", lambda df: df["ret20"] < 0),
    ("ret40_lt_0", lambda df: df["ret40"] < 0),
    ("ret60_lt_0", lambda df: df["ret60"] < 0),
    ("dist_ma20_lt_neg2", lambda df: df["dist_ma20"] < -0.02),
    ("dist_ma60_lt_neg5", lambda df: df["dist_ma60"] < -0.05),
    ("slope_ma60_lt_0", lambda df: df["slope_ma60"] < 0),
    ("dd_60d_lt_neg5", lambda df: df["dd_60d"] < -0.05),
    ("dd_120d_lt_neg10", lambda df: df["dd_120d"] < -0.10),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_v27_router(daily_idx: pl.DataFrame) -> dict:
    """V27 router: 16 signals, daily vote ≥ 2 → V2 flag, 10-day cum ≥ 4 → bear."""
    if daily_idx.height == 0:
        return {}
    df = daily_idx.with_columns([
        pl.col("idx_close").rolling_mean(5).alias("ma5"),
        pl.col("idx_close").rolling_mean(10).alias("ma10"),
        pl.col("idx_close").rolling_mean(20).alias("ma20"),
        pl.col("idx_close").rolling_mean(60).alias("ma60"),
        pl.col("idx_close").rolling_mean(120).alias("ma120"),
        pl.col("idx_close").pct_change(10).alias("ret10"),
        pl.col("idx_close").pct_change(20).alias("ret20"),
        pl.col("idx_close").pct_change(40).alias("ret40"),
        pl.col("idx_close").pct_change(60).alias("ret60"),
    ]).with_columns([
        ((pl.col("idx_close") - pl.col("ma20")) / pl.col("ma20")).alias("dist_ma20"),
        ((pl.col("idx_close") - pl.col("ma60")) / pl.col("ma60")).alias("dist_ma60"),
        pl.col("ma60").diff().alias("slope_ma60"),
        pl.col("idx_close").rolling_max(60).alias("high_60d"),
        pl.col("idx_close").rolling_max(120).alias("high_120d"),
    ]).with_columns([
        (pl.col("idx_close") / pl.col("high_60d") - 1).alias("dd_60d"),
        (pl.col("idx_close") / pl.col("high_120d") - 1).alias("dd_120d"),
    ])
    pdf = df.to_pandas()
    sig_count = pd.DataFrame()
    for name, fn in BEAR_SIGNAL_SPECS:
        sig_count[name] = fn(pdf).fillna(False)
    pdf["v2_vote_count"] = sig_count.sum(axis=1)
    pdf["v2_flag"] = (pdf["v2_vote_count"] >= V27_VOTE_THRESHOLD).astype(int)
    pdf["cum_v2"] = pdf["v2_flag"].rolling(V27_LOOKBACK).sum()
    pdf["is_bear"] = pdf["cum_v2"] >= V27_CUMULATIVE_THRESHOLD
    out = {}
    for d, v in zip(pdf["trade_date"], pdf["is_bear"]):
        out[pd.Timestamp(d).date().isoformat()] = bool(v)
    return out


def get_factor_columns(panel_cols: list[str]) -> list[str]:
    """Get all valid factor columns (exclude idx_*, v10_1_*, raw OHLCV)."""
    EXCLUDE_PREFIXES = ("v10_1_", "idx_")
    EXCLUDE_RAW = {"trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount",
                   "adj_close", "adj_factor", "pct_chg", "circ_cap", "cap", "l_size", "l_size3"}
    cols = []
    for c in panel_cols:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if c in EXCLUDE_RAW:
            continue
        if c.startswith("alpha_alpha_custom_") or c.startswith("mw_") or c.startswith("v_"):
            # Filter very custom or look-ahead-prone
            if "argmax" in c or "argmin" in c:
                continue
        cols.append(c)
    return cols


def build_v19_history(panel: pl.DataFrame, factors: list[str], bear_dates: list,
                      regime_map: dict, current_date: pd.Timestamp) -> dict:
    """For each factor, test high/low direction on top-10/bottom-10 stock portfolio over last N bear sessions.
    Return: {factor: (direction, mean_return)}
    Only bear sessions with exit_date < current_date are eligible (leakage guard).
    """
    FWD_DAYS = 20
    COST = 0.005
    # Filter bear sessions that exit before current_date
    eligible = []
    for d_str in bear_dates:
        d = pd.Timestamp(d_str)
        if d < current_date and (d + pd.Timedelta(days=FWD_DAYS * 2)) < current_date:
            eligible.append(d_str)
    eligible = eligible[-V19_LOOKBACK:]
    if not eligible:
        return {}

    # Build per-bear-date stock rankings
    factor_scores = {}
    for fac in factors:
        if fac not in panel.columns:
            continue
        # For each eligible bear session, get top-10 and bottom-10 stocks by factor value
        # Compute mean fwd return over those portfolios
        high_ret = []
        low_ret = []
        for rd_str in eligible:
            rd = pd.Timestamp(rd_str)
            try:
                day_data = panel.filter(
                    (pl.col("trade_date") == rd.to_pydatetime()) &
                    pl.col(fac).is_not_null()
                ).select(["ts_code", fac])
                if day_data.height < V19_TOP_STOCKS_PER_FACTOR * 2:
                    continue
                # rank
                pdf_day = day_data.to_pandas()
                pdf_day["rk"] = pdf_day[fac].rank(ascending=True, method="ordinal")
                # top-10 (high direction = highest factor values)
                top_high = pdf_day.nlargest(V19_TOP_STOCKS_PER_FACTOR, fac)["ts_code"].tolist()
                top_low = pdf_day.nsmallest(V19_TOP_STOCKS_PER_FACTOR, fac)["ts_code"].tolist()
                # Forward return: open[t+1] / open[t+21] - 1 (approx)
                # Use close[t+1] / close[t+21] - 1 as proxy
                entry_date = rd + pd.Timedelta(days=1)
                exit_date = rd + pd.Timedelta(days=21)
                # Compute mean return for each portfolio
                for sym in top_high:
                    ret_row = panel.filter(
                        (pl.col("ts_code") == sym) &
                        (pl.col("trade_date") >= exit_date.to_pydatetime()) &
                        (pl.col("trade_date") <= (exit_date + pd.Timedelta(days=3)).to_pydatetime())
                    ).select(["close"])
                    entry_row = panel.filter(
                        (pl.col("ts_code") == sym) &
                        (pl.col("trade_date") >= entry_date.to_pydatetime()) &
                        (pl.col("trade_date") <= (entry_date + pd.Timedelta(days=3)).to_pydatetime())
                    ).select(["close"])
                    if ret_row.height == 0 or entry_row.height == 0:
                        continue
                    ret = float(ret_row["close"][0]) / float(entry_row["close"][0]) - 1.0 - COST
                    high_ret.append(ret)
                for sym in top_low:
                    ret_row = panel.filter(
                        (pl.col("ts_code") == sym) &
                        (pl.col("trade_date") >= exit_date.to_pydatetime()) &
                        (pl.col("trade_date") <= (exit_date + pd.Timedelta(days=3)).to_pydatetime())
                    ).select(["close"])
                    entry_row = panel.filter(
                        (pl.col("ts_code") == sym) &
                        (pl.col("trade_date") >= entry_date.to_pydatetime()) &
                        (pl.col("trade_date") <= (entry_date + pd.Timedelta(days=3)).to_pydatetime())
                    ).select(["close"])
                    if ret_row.height == 0 or entry_row.height == 0:
                        continue
                    ret = float(ret_row["close"][0]) / float(entry_row["close"][0]) - 1.0 - COST
                    low_ret.append(ret)
            except Exception:
                continue
        if not high_ret and not low_ret:
            continue
        # mean of high and low
        mh = float(np.mean(high_ret)) if high_ret else 0.0
        ml = float(np.mean(low_ret)) if low_ret else 0.0
        # Choose better direction; require positive mean
        if mh > ml and mh > V19_MIN_FACTOR_MEAN_RETURN:
            factor_scores[fac] = ("high", mh)
        elif ml > mh and ml > V19_MIN_FACTOR_MEAN_RETURN:
            factor_scores[fac] = ("low", ml)
    return factor_scores


def build_v19_picks_at_date(panel: pl.DataFrame, factors: list[str], current_date: pd.Timestamp,
                            bear_dates: list, regime_map: dict) -> dict:
    """Compute V19 picks at current_date: factor-return historical ranking + voting."""
    factor_scores = build_v19_history(panel, factors, bear_dates, regime_map, current_date)
    if not factor_scores:
        return {}

    # Pick top V19_TOP_FACTORS by score
    sorted_factors = sorted(factor_scores.items(), key=lambda x: -x[1][1])[:V19_TOP_FACTORS]
    if not sorted_factors:
        return {}

    # Current date picks: top V19_TOP_STOCKS_PER_FACTOR stocks per factor
    day_data = panel.filter(
        (pl.col("trade_date") == current_date.to_pydatetime())
    ).select(["ts_code"] + [f[0] for f in sorted_factors])
    if day_data.height < V19_TOP_STOCKS_PER_FACTOR * 2:
        return {}

    votes = Counter()
    pdf_day = day_data.to_pandas()
    for fac, (direction, score) in sorted_factors:
        if fac not in pdf_day.columns:
            continue
        if direction == "high":
            top = pdf_day.nlargest(V19_TOP_STOCKS_PER_FACTOR, fac)["ts_code"].tolist()
        else:
            top = pdf_day.nsmallest(V19_TOP_STOCKS_PER_FACTOR, fac)["ts_code"].tolist()
        for sym in top:
            votes[sym] += 1

    if not votes:
        return {}

    # Apply MIN_VOTES
    sel = [s for s, c in votes.items() if c >= V19_MIN_VOTES]
    if len(sel) < V19_MIN_STOCKS:
        sel = [s for s, _ in votes.most_common(V19_MIN_STOCKS)]
    if len(sel) > V19_MAX_STOCKS:
        sel = [s for s, _ in votes.most_common(V19_MAX_STOCKS)]
    return {s: float(votes[s]) for s in sel}


def build_v32_picks_at_date(panel: pl.DataFrame, current_date: pd.Timestamp) -> dict:
    """V32 static pool 10 因子 voting at current_date."""
    day_data = panel.filter(
        pl.col("trade_date") == current_date.to_pydatetime()
    ).select(["ts_code"] + list(V32_POOL.keys()))
    if day_data.height < V32_K_NOM:
        return {}

    votes = Counter()
    pdf_day = day_data.to_pandas()
    for fac, direction in V32_POOL.items():
        if fac not in pdf_day.columns:
            continue
        sub = pdf_day[["ts_code", fac]].dropna()
        if len(sub) < V32_K_NOM:
            continue
        if direction == "hi":
            top = sub.nlargest(V32_K_NOM, fac)["ts_code"].tolist()
        else:
            top = sub.nsmallest(V32_K_NOM, fac)["ts_code"].tolist()
        for sym in top:
            votes[sym] += 1

    if not votes:
        return {}

    sel = [s for s, c in votes.items() if c >= V32_MIN_VOTES]
    if len(sel) < V32_MIN_STOCKS:
        sel = [s for s, _ in votes.most_common(V32_MIN_STOCKS)]
    if len(sel) > V32_MAX_STOCKS:
        sel = [s for s, _ in votes.most_common(V32_MAX_STOCKS)]
    return {s: float(votes[s]) for s in sel}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--out", default="evidence/v19_v32_fusion_picks_v27_20261006/bear_fusion_picks.json")
    parser.add_argument("--regime-out", default="evidence/v19_v32_fusion_picks_v27_20261006/regime_v27.json")
    args = parser.parse_args()

    log(f"加载 panel + 因子...")
    factors_all = []
    with open("evidence/v37_v14_causal/factor_manifest.json") as f:
        manifest = json.load(f)
    if "factors" in manifest:
        factors_all = manifest["factors"]
    else:
        # Fallback: list all columns
        sample = pl.read_parquet(PANEL, n_rows=1)
        factors_all = get_factor_columns(sample.columns)

    # Read needed columns only
    needed = ["trade_date", "ts_code", "open", "high", "low", "close", "vol", "idx_close"] + V32_POOL_KEYS()
    # Actually load full panel with all factors for V19 (heavy)
    df = pl.read_parquet(PANEL)
    log(f"panel loaded: {df.height:,} rows × {df.width} cols")

    # V27 router
    log(f"Building V27 router (16 signals + 10-day cum ≥ 4)...")
    daily_idx = (df.select(["trade_date", "idx_close"])
                 .filter(pl.col("idx_close").is_not_null())
                 .unique("trade_date")
                 .sort("trade_date"))
    regime_map = build_v27_router(daily_idx)
    bear_count = sum(1 for v in regime_map.values() if v)
    bull_count = sum(1 for v in regime_map.values() if not v)
    log(f"  V27: {bear_count} bear dates, {bull_count} bull/neutral dates")

    # Save regime map
    Path(args.regime_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.regime_out, "w") as f:
        json.dump(regime_map, f)

    # Bear sessions for V19 lookback
    bear_dates_sorted = sorted([d for d, v in regime_map.items() if v])

    # Generate picks at every trading date
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    all_dates = (df.filter(
        (pl.col("trade_date") >= start.to_pydatetime()) &
        (pl.col("trade_date") <= end.to_pydatetime())
    ).get_column("trade_date").unique().sort().to_list())

    log(f"Building bear fusion picks across {len(all_dates)} dates...")
    bear_fusion_picks = {}
    v19_picks_log = {}
    v32_picks_log = {}

    for i, rd in enumerate(all_dates):
        rd_str = pd.Timestamp(rd).date().isoformat()
        if not regime_map.get(rd_str, False):
            continue  # bull/neutral — handled by V37 IC voting
        # bear date — compute V19 + V32 picks
        # Rebalance every V19_REBAL_STEP bear sessions
        # Get bear session index
        bear_idx = [j for j, d in enumerate(bear_dates_sorted) if d <= rd_str]
        if not bear_idx:
            continue
        last_bear = bear_dates_sorted[bear_idx[-1]]
        # Only rebalance every V19_REBAL_STEP bear sessions
        if len(bear_idx) % V19_REBAL_STEP != 1:
            # continue holding last picks
            if bear_fusion_picks:
                bear_fusion_picks[rd_str] = bear_fusion_picks.get(last_bear, {})
            continue

        # Compute V19 picks
        v19_picks = build_v19_picks_at_date(df, factors_all, pd.Timestamp(rd), bear_dates_sorted, regime_map)
        # Compute V32 picks
        v32_picks = build_v32_picks_at_date(df, pd.Timestamp(rd))

        # Union with score merge: take max(v19_vote, v32_vote)
        merged = {}
        for s, c in v19_picks.items():
            merged[s] = max(c, v32_picks.get(s, 0))
        for s, c in v32_picks.items():
            if s not in merged:
                merged[s] = c

        # Filter MIN_STOCKS=5, MAX_STOCKS=20
        if len(merged) < V32_MIN_STOCKS:
            # too few — take top by score
            sel = [s for s, _ in sorted(merged.items(), key=lambda x: -x[1])[:V32_MAX_STOCKS]]
        else:
            sel = [s for s, _ in sorted(merged.items(), key=lambda x: -x[1])[:V32_MAX_STOCKS]]
        bear_fusion_picks[rd_str] = {s: float(merged[s]) for s in sel}
        v19_picks_log[rd_str] = v19_picks
        v32_picks_log[rd_str] = v32_picks

        if (i + 1) % 100 == 0:
            log(f"  [{i+1}/{len(all_dates)}] {rd_str}: V19={len(v19_picks)}, V32={len(v32_picks)}, fusion={len(sel)}")

    # Save
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bear_fusion_picks, f)
    log(f"Saved bear fusion picks: {len(bear_fusion_picks)} bear dates -> {args.out}")

    # Summary stats
    sizes = [len(p) for p in bear_fusion_picks.values()]
    log(f"  picks size: min={min(sizes) if sizes else 0}, max={max(sizes) if sizes else 0}, mean={np.mean(sizes) if sizes else 0:.1f}")


def V32_POOL_KEYS():
    return list(V32_POOL.keys())


if __name__ == "__main__":
    main()