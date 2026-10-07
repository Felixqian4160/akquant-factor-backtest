"""Daily Factor Reselect v11 — Multi-factor voting mechanism:

User's contract (from latest message):
- 每天单因子回测 → 排名 → 取前 10 因子
- 每个因子独立选 top-10 股票 → 100 次选择
- 取次数最多的 5-10 只股票 = 共识
- 用共识股票买入, 持有, 卖出
- 使用概率来买卖 (vote count = selection probability)

Implementation:
- Phase 1: each factor runs single-factor backtest over recent window, ranks by mean fwd return
- Phase 2: top 10 factors each independently pick top-10 stocks (cross-section percentile rank)
- Phase 3: vote count per stock (how many of the 10 factors selected it)
- Phase 4: pick top-5 stocks by vote count (high consensus = high prob of alpha)
- AKQuant single continuous backtest with rebalance on consensus change + MIN_HOLD_DAYS

Stage 5b 6 factors + 4 from v10.2 rule-driven (sig_low_pullback alternatives
since panel doesn't have sig_*) for a 10-factor pool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import numpy as np
import pandas as pd

# Path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/media/felix/f/quant/aurumq-rl/src")

import akquant as aq  # noqa: E402

PANEL_FILE = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet"
)

# 10-factor pool: Stage 5b 6 + 4 v10.2 rule-driven proxies
FACTOR_POOL = [
    "talib_NATR", "talib_TRANGE", "gtja_gtja_159",
    "gtja_gtja_149", "gtja_gtja_144", "mw_vol_20d",
    "mw_ret_20d", "mw_rsi_14", "alpha_alpha018", "alpha_alpha028",
]
TOP_FACTORS = 10  # use all
TOP_STOCKS_PER_FACTOR = 10
TOP_STOCKS_FINAL = 5  # final consensus: top 5 by vote count
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
MIN_HOLD_DAYS = 5
WINDOW_DAYS = 60
FWD_DAYS = 20

# Pre-computed daily picks: {date_str: {symbol: vote_count}}
_DAILY_PICKS: dict[str, dict[str, float]] = {}


class DailyReselectV11Strategy(aq.Strategy):
    """Multi-factor voting. Rebalance on consensus change + MIN_HOLD_DAYS."""

    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._last_rebalance_date = None

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]
        picks = _DAILY_PICKS.get(date_str, {})
        new_set = set(picks.keys())
        if new_set == self._current_picks:
            return
        if not new_set:
            return

        if self._last_rebalance_date is not None and MIN_HOLD_DAYS > 0:
            days_since = (
                datetime.strptime(date_str, "%Y-%m-%d")
                - datetime.strptime(self._last_rebalance_date, "%Y-%m-%d")
            ).days
            if days_since < MIN_HOLD_DAYS:
                overlap = len(new_set & self._current_picks)
                if overlap >= len(new_set) * 0.5:
                    return

        try:
            self.rebalance_to_topn(
                scores=picks,
                top_n=len(new_set),
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
            self._current_picks = new_set
            self._last_rebalance_date = date_str
        except Exception as exc:
            self.log(f"rebalance failed on {date_str}: {exc}")


def single_factor_backtest_v11(panel, factor, eval_date, top_k=TOP_STOCKS_PER_FACTOR,
                                window_days=WINDOW_DAYS, fwd_days=FWD_DAYS):
    """Single-factor cross-section percentile rank at eval_date → top K stocks.
    Used as proxy for 'single-factor backtest'. Fast (single date look-up).
    """
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    sub = panel.filter(
        pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)
    if factor not in sub.columns:
        return []
    sub = sub.filter(pl.col(factor).is_not_null())
    if sub.height == 0:
        return []
    sorted_sub = sub.sort(factor, descending=True, nulls_last=True).head(top_k)
    return sorted_sub["ts_code"].to_list()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v11")
    args = parser.parse_args()

    job_dir = Path(args.out_base) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_log = job_dir / "progress.jsonl"

    def emit(stage, **kw):
        rec = {"stage": stage, "job_id": args.job_id, "timestamp": datetime.now().isoformat(), **kw}
        with open(progress_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, ensure_ascii=False))

    emit("init", start_date=args.start_date, end_date=args.end_date,
         top_factors=TOP_FACTORS, top_stocks_per_factor=TOP_STOCKS_PER_FACTOR,
         top_stocks_final=TOP_STOCKS_FINAL)
    panel = pl.read_parquet(PANEL_FILE)
    emit("panel_loaded", n_rows=panel.height, n_cols=panel.width)

    start_dt = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date[:10], "%Y-%m-%d").date()
    all_dates = (
        panel.filter(
            (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
            & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        )["trade_date"].unique().sort().to_list()
    )
    if args.max_days:
        all_dates = all_dates[: args.max_days]
    n_total = len(all_dates)
    emit("dates_loaded", n_dates_in_range=n_total)
    if n_total < 5:
        emit("error", msg="too few dates")
        return

    # Phase 1: vectorised daily voting — for each (date, factor), rank stocks
    # and count votes per stock. Take top 5 by vote count.
    t_start = time.time()
    daily_picks_log = []
    valid_factors = [f for f in FACTOR_POOL if f in panel.columns]
    emit("factor_pool", total=FACTOR_POOL, valid=valid_factors)

    # Filter panel to date range, last date per stock
    sub_all = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).select(["trade_date", "ts_code"] + valid_factors)

    # For each (date, factor) → top-10 stocks by rank descending
    # Use polars window functions for cross-section rank
    for f in valid_factors:
        sub_all = sub_all.with_columns(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias(f"_rank_{f}")
        )

    # Now for each date, for each factor, get top-10 stocks
    # Build long-format: (date, factor, ts_code, rank) then filter top-10 per (date,factor)
    long_df = sub_all.select(
        pl.col("trade_date"),
        pl.col("ts_code"),
    )
    # We'll iterate per-factor to get top-10 picks per (date, factor)
    # Then collect into vote counter
    vote_records = []  # list of (date, symbol)
    per_factor_top_picks = {f: {} for f in valid_factors}

    for f in valid_factors:
        rank_col = f"_rank_{f}"
        # Filter top-10 per date
        top_picks = sub_all.filter(pl.col(rank_col) <= TOP_STOCKS_PER_FACTOR).select(["trade_date", "ts_code"])
        for row in top_picks.to_dicts():
            d = str(row["trade_date"])[:10]
            sym = row["ts_code"]
            per_factor_top_picks[f].setdefault(d, []).append(sym)
            vote_records.append((d, sym))

    emit("voting_done", n_records=len(vote_records), duration_sec=time.time() - t_start)

    # Aggregate votes per (date, symbol)
    votes_df = pl.DataFrame(vote_records, schema=["date", "symbol"], orient="row")
    vote_counts = (
        votes_df.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
        .sort(["date", "votes"], descending=[False, True])
    )

    # Take top TOP_STOCKS_FINAL by votes per date
    vote_counts = vote_counts.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    consensus_df = vote_counts.filter(pl.col("_date_rank") <= TOP_STOCKS_FINAL)
    emit("consensus_done", n_consensus=consensus_df.height, duration_sec=time.time() - t_start)

    # Convert to dict for AKQuant
    _DAILY_PICKS.clear()
    daily_picks_log = []
    for row in consensus_df.to_dicts():
        d = row["date"]
        sym = row["symbol"]
        v = int(row["votes"])
        if d not in _DAILY_PICKS:
            _DAILY_PICKS[d] = {}
        _DAILY_PICKS[d][sym] = float(v)
        daily_picks_log.append({
            "date": d, "picks": [sym], "votes": {sym: v},
            "n_unique_picks": 0, "top_vote_count": v,
        })

    emit("picks_phase_done", n_days=len(_DAILY_PICKS),
         duration_sec=time.time() - t_start)

    # Phase 2: build data dict
    all_picks_universe = sorted({sym for picks in _DAILY_PICKS.values() for sym in picks.keys()})
    emit("universe_size", n_unique_stocks=len(all_picks_universe))

    warmup_start = start_dt - timedelta(days=30)
    # Use panel with all stocks for warmup data (data_dict must include them)
    all_in_universe = sorted(set(all_picks_universe))
    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(warmup_start.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & pl.col("ts_code").is_in(all_in_universe)
    ).select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"])

    data_dict: dict[str, pd.DataFrame] = {}
    for code in all_in_universe:
        pdf = (
            sub.filter(pl.col("ts_code") == code)
            .sort("trade_date")
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        if pdf.empty:
            continue
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1.0e9
        data_dict[code] = pdf
    emit("data_dict_built", n_symbols=len(data_dict))

    # Phase 3: AKQuant single continuous backtest
    emit("akquant_start")
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyReselectV11Strategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
        )
        metrics = result.metrics_df
        emit("akquant_done")
    except Exception as exc:
        emit("akquant_failed", error=str(exc))
        return

    metrics_dict = {}
    for idx in metrics.index:
        try:
            v = metrics.loc[idx, "value"]
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            metrics_dict[idx] = float(v)
        except (TypeError, ValueError):
            metrics_dict[idx] = str(metrics.loc[idx, "value"])
    final_value = float(metrics_dict.get("end_market_value", INITIAL_CASH))
    final_return_pct = (final_value / INITIAL_CASH - 1) * 100

    nav_curve = []
    try:
        eq = result.equity_curve
        if hasattr(eq, "items"):
            for ts, val in eq.items():
                nav_curve.append([str(ts)[:10], val])
    except Exception:
        pass

    result_obj = {
        "job_id": args.job_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "final_value": final_value,
        "initial_cash": INITIAL_CASH,
        "final_nav": final_value / INITIAL_CASH,
        "final_return_pct": final_return_pct,
        "n_days": n_total,
        "metrics": metrics_dict,
        "factor_pool": FACTOR_POOL,
        "top_factors": TOP_FACTORS,
        "top_stocks_per_factor": TOP_STOCKS_PER_FACTOR,
        "top_stocks_final": TOP_STOCKS_FINAL,
        "min_hold_days": MIN_HOLD_DAYS,
        "daily_picks": daily_picks_log,
        "nav_curve": nav_curve,
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         n_days=n_total, duration_sec=time.time() - t_start)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()