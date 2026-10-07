"""ALL_FUSION_DYNAMIC v2 — 基于数学/统计学的投票选股策略.

设计合同 (从之前 sweep 综合):
1. 动态股票池 (无 static pool):
   - bull/neutral: V14 IC voting 选 top-10 因子 (rolling 60d Spearman)
   - bear: V19 historical factor-return ranking 选 top-10 因子
2. Router: V27_BALANCE (vt=2, lb=5, ct=5)
3. 动态仓位: vol_base=0.005 × dd_slope=7.0
4. Cross-section rank (而非 raw value, 抗 outliers)
5. T+1 raw open → T+21 raw open (no look-ahead)
6. 0.5% round-trip cost, lot=100, A 股

数学保证:
- Spearman IC: 秩相关, 对 non-linear 更稳健
- rank(method='ordinal'): 唯一排序, 无 ties
- 投票机制: 多因子共识, 减少单因子噪声
- 滚动窗口: 60d IC, 30 bear sessions, 5d vol
- Z-score: vol factor = 0.005 / vol, 当 vol↑时 factor↓
- DD factor: 1 + dd × 7, 当 dd 深时 factor↓

实施:
- AKQuant 0.3.x run_backtest
- 5-seed (rebal_offset 0/4/8/12/16) 验证稳定性
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
PIVOTS_PATH = "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/hs300_index_pivots_clean_20260919/hs300_index_pivots.json"
OUT_BASE = Path("evidence/all_fusion_dynamic_v2_20261006")
OUT_BASE.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100

# ========== Router params (V27_BALANCE) ==========
BEAR_VOTE_THRESHOLD = 2
BEAR_LOOKBACK = 5
BEAR_CUMULATIVE_THRESHOLD = 5

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

# ========== Bull side (V14 IC voting) ==========
IC_WINDOW_DAYS = 60
CAUSAL_LAG = 21
TOP_FACTORS_BULL = 10
TOP_STOCKS_PER_FACTOR_BULL = 10
MIN_VOTES_BULL = 2
MIN_STOCKS_BULL = 5
MAX_STOCKS_BULL = 10
BULL_REBAL_STEP = 20

# ========== Bear side (V19 historical ranking) ==========
BEAR_LOOKBACK_SESSIONS = 30
TOP_FACTORS_BEAR = 10
TOP_STOCKS_PER_FACTOR_BEAR = 10
MIN_VOTES_BEAR = 3
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10
BEAR_REBAL_SESSIONS = 20
MIN_FACTOR_OBS_BEAR = 10
MIN_FACTOR_MEAN_RETURN = 0.005

# ========== Dynamic target ==========
VOL_BASE = 0.005
DD_SLOPE = 7.0

# ========== Factor exclusion ==========
EXCLUDE_PREFIXES = ("v10_1_", "idx_")
EXCLUDE_RAW = {"ts_code", "trade_date", "open", "high", "low", "close", "vol",
               "amount", "adj_close", "adj_factor", "pct_chg", "circ_cap", "cap"}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_factors(panel_cols: list[str]) -> list[str]:
    """Get usable factor columns (exclude v10_1_*, idx_*, raw OHLCV)."""
    cols = []
    for c in panel_cols:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if c in EXCLUDE_RAW:
            continue
        if "argmax" in c or "argmin" in c:
            continue
        if c.startswith("alpha_alpha_custom_") or c.startswith("mw_") or c.startswith("v_"):
            pass
        cols.append(c)
    return cols


def build_v27_router(daily_idx: pd.DataFrame) -> dict:
    """V27 router: 16 signals + 5d cum >= 5."""
    df = daily_idx.copy()
    sig_count = pd.DataFrame()
    for name, fn in BEAR_SIGNAL_SPECS:
        sig_count[name] = fn(df).fillna(False)
    df["v27_vote_count"] = sig_count.sum(axis=1)
    df["v27_v2_flag"] = (df["v27_vote_count"] >= BEAR_VOTE_THRESHOLD).astype(int)
    df["v27_cum"] = df["v27_v2_flag"].rolling(BEAR_LOOKBACK, min_periods=1).sum()
    df["is_bear"] = (df["v27_cum"] >= BEAR_CUMULATIVE_THRESHOLD).astype(int)
    out = {}
    for d, v in zip(df["date"], df["is_bear"]):
        out[d] = bool(v)
    return out


def compute_bull_ic(panel: pl.DataFrame, factors: list[str], rebal_date: pd.Timestamp,
                    history_dates: list) -> dict:
    """Rolling IC: Spearman corr of factor value vs forward 20d return across stocks.
    Lag = CAUSAL_LAG days (no look-ahead).
    """
    FWD = 20
    # Compute forward return for each (date, stock) ending CAUSAL_LAG days after rebal_date
    # Actually: at rebal_date, we use fwd_ret[t+lag+1 : t+lag+21]
    fwd_start = rebal_date + pd.Timedelta(days=CAUSAL_LAG + 1)
    fwd_end = rebal_date + pd.Timedelta(days=CAUSAL_LAG + FWD)

    # Get forward close
    fwd_close_data = panel.filter(
        (pl.col("trade_date") >= fwd_start.to_pydatetime()) &
        (pl.col("trade_date") <= fwd_end.to_pydatetime())
    ).select(["ts_code", "trade_date", "close"])

    if fwd_close_data.height == 0:
        return {}

    # Compute forward return: close[fwd_end] / close[fwd_start] - 1
    fwd_ret = (fwd_close_data.group_by("ts_code")
                .agg([
                    (pl.col("close").last() / pl.col("close").first() - 1).alias("fwd_ret")
                ])
                .filter(pl.col("fwd_ret").is_not_null()))

    # Get factor values at rebal_date
    rebal_data = panel.filter(
        pl.col("trade_date") == rebal_date.to_pydatetime()
    ).select(["ts_code"] + factors)

    # Per-factor IC across stocks: Spearman corr of factor vs fwd_ret
    # Use pandas for Spearman
    rebal_pdf = rebal_data.to_pandas()
    fwd_pdf = fwd_ret.to_pandas()
    merged = rebal_pdf.merge(fwd_pdf, on="ts_code", how="inner")

    if merged.height if hasattr(merged, 'height') else len(merged) < 20:
        return {}

    ic_scores = {}
    for fac in factors:
        if fac not in merged.columns:
            continue
        sub = merged[[fac, "fwd_ret"]].dropna()
        if len(sub) < 20:
            continue
        # Spearman corr via rank
        try:
            corr = sub[fac].rank().corr(sub["fwd_ret"].rank())
            if not np.isnan(corr):
                ic_scores[fac] = float(corr)
        except Exception:
            continue
    return ic_scores


def build_bull_picks(panel: pl.DataFrame, factors: list[str], current_date: pd.Timestamp,
                     history_dates: list) -> dict:
    """Bull/neutral regime picks via rolling IC voting."""
    ic_scores = compute_bull_ic(panel, factors, current_date, history_dates)
    if not ic_scores:
        return {}

    # Use |IC| as factor strength, sort descending
    sorted_factors = sorted(ic_scores.items(), key=lambda x: -abs(x[1]))[:TOP_FACTORS_BULL]
    if not sorted_factors:
        return {}

    # Current date picks: top stocks per factor by |rank|
    day_data = panel.filter(
        pl.col("trade_date") == current_date.to_pydatetime()
    ).select(["ts_code"] + [f[0] for f in sorted_factors])
    if day_data.height < TOP_STOCKS_PER_FACTOR_BULL * 2:
        return {}

    votes = Counter()
    pdf_day = day_data.to_pandas()
    for fac, ic in sorted_factors:
        if fac not in pdf_day.columns:
            continue
        sub = pdf_day[["ts_code", fac]].dropna()
        if len(sub) < TOP_STOCKS_PER_FACTOR_BULL:
            continue
        # direction = sign(IC), high direction for positive IC
        if ic > 0:
            top = sub.nlargest(TOP_STOCKS_PER_FACTOR_BULL, fac)["ts_code"].tolist()
        else:
            top = sub.nsmallest(TOP_STOCKS_PER_FACTOR_BULL, fac)["ts_code"].tolist()
        for sym in top:
            votes[sym] += 1

    # Apply MIN_VOTES
    sel = [s for s, c in votes.items() if c >= MIN_VOTES_BULL]
    if len(sel) < MIN_STOCKS_BULL:
        sel = [s for s, _ in votes.most_common(MIN_STOCKS_BULL)]
    if len(sel) > MAX_STOCKS_BULL:
        sel = [s for s, _ in votes.most_common(MAX_STOCKS_BULL)]
    return {s: float(votes[s]) for s in sel}


def build_bear_factor_returns(panel: pl.DataFrame, factors: list[str],
                               bear_dates: list, current_date: pd.Timestamp) -> dict:
    """V19 historical factor-return ranking: each factor's top/bot-10 mean net return over last N bear sessions."""
    FWD = 20
    COST = 0.005
    eligible = []
    for d_str in bear_dates:
        d = pd.Timestamp(d_str)
        if d < current_date and (d + pd.Timedelta(days=FWD * 2)) < current_date:
            eligible.append(d_str)
    eligible = eligible[-BEAR_LOOKBACK_SESSIONS:]
    if not eligible:
        return {}

    factor_scores = {}
    for fac in factors:
        if fac not in panel.columns:
            continue
        high_ret = []
        low_ret = []
        for rd_str in eligible:
            rd = pd.Timestamp(rd_str)
            try:
                day_data = panel.filter(
                    (pl.col("trade_date") == rd.to_pydatetime()) &
                    pl.col(fac).is_not_null()
                ).select(["ts_code", fac])
                if day_data.height < TOP_STOCKS_PER_FACTOR_BEAR * 2:
                    continue
                pdf_day = day_data.to_pandas()
                top_high = pdf_day.nlargest(TOP_STOCKS_PER_FACTOR_BEAR, fac)["ts_code"].tolist()
                top_low = pdf_day.nsmallest(TOP_STOCKS_PER_FACTOR_BEAR, fac)["ts_code"].tolist()
                # Forward return: open[t+1] / open[t+21] proxy = close[t+1] / close[t+21]
                entry_date = rd + pd.Timedelta(days=1)
                exit_date = rd + pd.Timedelta(days=FWD)
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
        mh = float(np.mean(high_ret)) if high_ret else 0.0
        ml = float(np.mean(low_ret)) if low_ret else 0.0
        if mh > ml and mh > MIN_FACTOR_MEAN_RETURN:
            factor_scores[fac] = ("high", mh)
        elif ml > mh and ml > MIN_FACTOR_MEAN_RETURN:
            factor_scores[fac] = ("low", ml)
    return factor_scores


def build_bear_picks(panel: pl.DataFrame, factors: list[str], current_date: pd.Timestamp,
                     bear_dates: list) -> dict:
    """V19 picks: factor-return historical ranking + voting."""
    factor_scores = build_bear_factor_returns(panel, factors, bear_dates, current_date)
    if not factor_scores:
        return {}

    sorted_factors = sorted(factor_scores.items(), key=lambda x: -x[1][1])[:TOP_FACTORS_BEAR]
    if not sorted_factors:
        return {}

    day_data = panel.filter(
        pl.col("trade_date") == current_date.to_pydatetime()
    ).select(["ts_code"] + [f[0] for f in sorted_factors])
    if day_data.height < TOP_STOCKS_PER_FACTOR_BEAR * 2:
        return {}

    votes = Counter()
    pdf_day = day_data.to_pandas()
    for fac, (direction, score) in sorted_factors:
        if fac not in pdf_day.columns:
            continue
        sub = pdf_day[["ts_code", fac]].dropna()
        if len(sub) < TOP_STOCKS_PER_FACTOR_BEAR:
            continue
        if direction == "high":
            top = sub.nlargest(TOP_STOCKS_PER_FACTOR_BEAR, fac)["ts_code"].tolist()
        else:
            top = sub.nsmallest(TOP_STOCKS_PER_FACTOR_BEAR, fac)["ts_code"].tolist()
        for sym in top:
            votes[sym] += 1

    sel = [s for s, c in votes.items() if c >= MIN_VOTES_BEAR]
    if len(sel) < MIN_STOCKS_BEAR:
        sel = [s for s, _ in votes.most_common(MIN_STOCKS_BEAR)]
    if len(sel) > MAX_STOCKS_BEAR:
        sel = [s for s, _ in votes.most_common(MAX_STOCKS_BEAR)]
    return {s: float(votes[s]) for s in sel}


def dyn_target(equity_history: list[float]) -> float:
    """Dynamic target: vol_base × dd_factor, clip [0.1, 0.99]."""
    if len(equity_history) < 5:
        return 0.99
    arr = np.array(equity_history[-21:], dtype=np.float64)
    if arr.mean() <= 0:
        return 0.5
    ret = arr[1:] / arr[:-1] - 1.0
    vol = float(ret.std()) if len(ret) > 1 else 0.01
    vol_factor = float(np.clip(VOL_BASE / max(vol, 0.005), 0.2, 1.0))
    peak = arr.max()
    dd = arr[-1] / peak - 1.0 if peak > 0 else 0.0
    dd_factor = float(np.clip(1.0 + dd * DD_SLOPE, 0.25, 1.0))
    target = 0.99 * vol_factor * dd_factor
    return float(np.clip(target, 0.1, 0.99))


def run_backtest(rebal_offset: int = 0) -> dict:
    """Run ALL_FUSION_DYNAMIC v2 with given rebal_offset (0/4/8/12/16)."""
    log(f"=== ALL_FUSION_DYNAMIC v2 | offset={rebal_offset} ===")
    log("Loading panel + factors...")
    sample = pl.read_parquet(PANEL, n_rows=1)
    factors = get_factors(sample.columns)
    log(f"  factors: {len(factors)}")

    # Load all dates
    all_dates_pdf = (pl.read_parquet(PANEL, columns=["trade_date", "idx_close"])
                     .filter(pl.col("idx_close").is_not_null())
                     .unique("trade_date").sort("trade_date").to_pandas())
    all_dates_pdf["date"] = pd.to_datetime(all_dates_pdf["trade_date"]).dt.date

    # Build daily idx signals for V27 router
    daily_idx = all_dates_pdf.copy()
    daily_idx = daily_idx.assign(
        ma5=daily_idx["idx_close"].rolling(5).mean(),
        ma10=daily_idx["idx_close"].rolling(10).mean(),
        ma20=daily_idx["idx_close"].rolling(20).mean(),
        ma60=daily_idx["idx_close"].rolling(60).mean(),
        ma120=daily_idx["idx_close"].rolling(120).mean(),
    )
    daily_idx["ret10"] = daily_idx["idx_close"].pct_change(10)
    daily_idx["ret20"] = daily_idx["idx_close"].pct_change(20)
    daily_idx["ret40"] = daily_idx["idx_close"].pct_change(40)
    daily_idx["ret60"] = daily_idx["idx_close"].pct_change(60)
    daily_idx["dist_ma20"] = (daily_idx["idx_close"] - daily_idx["ma20"]) / daily_idx["ma20"]
    daily_idx["dist_ma60"] = (daily_idx["idx_close"] - daily_idx["ma60"]) / daily_idx["ma60"]
    daily_idx["slope_ma60"] = daily_idx["ma60"].diff()
    daily_idx["high_60d"] = daily_idx["idx_close"].rolling(60).max()
    daily_idx["high_120d"] = daily_idx["idx_close"].rolling(120).max()
    daily_idx["dd_60d"] = daily_idx["idx_close"] / daily_idx["high_60d"] - 1
    daily_idx["dd_120d"] = daily_idx["idx_close"] / daily_idx["high_120d"] - 1

    # V27 router
    regime_map = build_v27_router(daily_idx)
    log(f"  V27 router: {sum(1 for v in regime_map.values() if v)} bear / {sum(1 for v in regime_map.values() if not v)} bull")

    # Bear sessions for V19
    bear_dates_sorted = sorted([d for d, v in regime_map.items() if v])

    # Generate picks at every rebal date
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2025-12-31")
    in_window = daily_idx[
        (daily_idx["date"] >= start_date.date()) &
        (daily_idx["date"] <= end_date.date())
    ]
    all_dates_in_window = sorted(in_window["date"].tolist())
    rebal_dates = all_dates_in_window[::BULL_REBAL_STEP]
    # Apply offset
    rebal_dates = rebal_dates[rebal_offset::1]
    log(f"  rebal dates: {len(rebal_dates)} (offset={rebal_offset})")

    # Generate picks at each rebal date
    picks_dict = {}
    history_dates = [pd.Timestamp(d) for d in all_dates_in_window]
    for rd in rebal_dates:
        rd_pd = pd.Timestamp(rd)
        is_bear = regime_map.get(rd, False)
        # Bear rebal logic: only rebalance every BEAR_REBAL_SESSIONS bear sessions
        if is_bear:
            # find current bear session idx
            bear_idx = [j for j, d in enumerate(bear_dates_sorted) if d <= rd]
            if not bear_idx or (len(bear_idx) % BEAR_REBAL_SESSIONS != 1):
                # carry last bear picks
                last_bear = bear_dates_sorted[bear_idx[-1]] if bear_idx else None
                if last_bear and last_bear in picks_dict:
                    picks_dict[rd] = picks_dict[last_bear]
                continue
            picks = build_bear_picks(pl.read_parquet(PANEL), factors, rd_pd, bear_dates_sorted)
        else:
            picks = build_bull_picks(pl.read_parquet(PANEL), factors, rd_pd, history_dates)
        if picks:
            picks_dict[rd] = picks

    log(f"  picks generated: {len(picks_dict)} dates")

    # Run AKQuant
    universe = sorted({s for p in picks_dict.values() for s in p})
    log(f"  universe: {len(universe)} symbols")

    # Build data dict (minimal cols for AKQuant)
    needed_cols = ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
    data_src = pl.read_parquet(PANEL, columns=needed_cols).filter(
        (pl.col("trade_date") >= (start_date - pd.Timedelta(days=30)).to_pydatetime()) &
        (pl.col("trade_date") <= end_date.to_pydatetime())
    )
    data = {}
    for sym in universe:
        pdf = data_src.filter(pl.col("ts_code") == sym).sort("trade_date").to_pandas().set_index("trade_date").rename_axis("date")
        if pdf.empty:
            continue
        pdf["symbol"] = sym
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1e9
        data[sym] = pdf

    _DAILY_PICKS = {str(k): v for k, v in picks_dict.items()}
    _REBAL_SET = set(_DAILY_PICKS.keys())
    _NAV_HISTORY = [INITIAL_CASH]
    _TARGETS_LOG = []

    class AllFusionStrategy(aq.Strategy):
        warmup = 5

        def on_bar(self, bar):
            pass

        def on_after_trading(self, trading_date, timestamp):
            try:
                eq = float(self.equity)
                if eq > 0:
                    if not _NAV_HISTORY or _NAV_HISTORY[-1] != eq:
                        _NAV_HISTORY.append(eq)
                        if len(_NAV_HISTORY) > 30:
                            _NAV_HISTORY.pop(0)
            except Exception:
                pass

        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET:
                return
            p = _DAILY_PICKS.get(d, {})
            if not p:
                return
            n = len(p)
            target = dyn_target(_NAV_HISTORY)
            self.rebalance_weights(
                target_weights={s: target / n for s in p},
                liquidate_unmentioned=True,
            )
            _TARGETS_LOG.append((d, target, _NAV_HISTORY[-1] if _NAV_HISTORY else 0))

    log(f"  Running AKQuant...")
    t0 = time.time()
    result = aq.run_backtest(
        data=data, strategy=AllFusionStrategy,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        t_plus_one=False,
        fill_policy=aq.NextOpen(),
        lot_size=LOT_SIZE,
    )
    log(f"  AKQuant done in {time.time()-t0:.1f}s")

    metrics = result.metrics_df
    metrics_dict = {}
    for index in metrics.index:
        value = metrics.loc[index, "value"]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        try:
            metrics_dict[index] = float(value)
        except (TypeError, ValueError):
            metrics_dict[index] = str(value)

    # Save
    out_dir = OUT_BASE / f"offset_{rebal_offset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "offset": rebal_offset,
            "contract": {
                "router": "V27_BALANCE (vt=2, lb=5, ct=5)",
                "bull_selector": "V14 IC voting (60d Spearman, lag=21)",
                "bear_selector": "V19 historical factor-return ranking (lookback=30)",
                "dynamic_target": "vol_base=0.005, dd_slope=7.0",
                "rebalance_step": BULL_REBAL_STEP,
                "t_plus_one": False,
                "round_trip_cost": 0.005,
                "lot_size": 100,
            },
            "metrics": metrics_dict,
            "n_picks_dates": len(picks_dict),
            "targets_log_count": len(_TARGETS_LOG),
            "universe_size": len(universe),
        }, f, indent=2, ensure_ascii=False)
    return metrics_dict


def main():
    log("=" * 80)
    log("ALL_FUSION_DYNAMIC v2 — 5-seed sweep")
    log("=" * 80)
    summary = {}
    for offset in [0, 4, 8, 12, 16]:
        try:
            m = run_backtest(offset)
            ret = m.get("total_return_pct", 0)
            sharpe = m.get("sharpe_ratio", 0)
            mdd = m.get("max_drawdown_pct", 0)
            pf = m.get("profit_factor", 0)
            win = m.get("win_rate", 0)
            n_trades = int(m.get("closed_trade_count", 0))
            summary[offset] = {"ret": ret, "sharpe": sharpe, "mdd": mdd,
                               "pf": pf, "win": win, "n_trades": n_trades}
            marker = "✅" if ret > 0 else "❌"
            log(f"  {marker} offset={offset}: +{ret:.2f}% / sharpe {sharpe:.3f} / MDD -{mdd:.2f}% / PF {pf:.3f}")
        except Exception as e:
            log(f"  offset={offset} FAILED: {e}")
            summary[offset] = {"error": str(e)}

    log("\n=== 5-seed sweep summary ===")
    log(f"{'offset':>7} {'ret':>8} {'sharpe':>7} {'MDD':>7} {'PF':>5} {'Win':>5} {'Trades':>7}")
    for offset, m in summary.items():
        if 'error' in m:
            log(f"  {offset:>5} ERROR: {m['error']}")
            continue
        marker = "✅" if m['ret'] > 0 else "❌"
        log(f"  {marker} {offset:>5} {m['ret']:>+7.2f}% {m['sharpe']:>7.3f} {m['mdd']:>6.2f}% {m['pf']:>5.3f} {m['win']:>4.1f}% {m['n_trades']:>7}")

    rets = [m['ret'] for m in summary.values() if 'ret' in m]
    if rets:
        log(f"\n  mean: {sum(rets)/len(rets):+.2f}%")
        log(f"  median: {sorted(rets)[len(rets)//2]:+.2f}%")
        log(f"  spread: {max(rets)-min(rets):.2f}%")
        log(f"  n_pos: {sum(1 for r in rets if r > 0)}/{len(rets)}")

    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()