"""Daily Factor Reselect v13 — TRUE factor backtest:

User's clarified contract:
- 因子回测的标的是 **沪深 300 每只股票的 forward 20d return**
- 因子整体表现 = **所有股票的 forward return 的 mean** (用户合同: 平均数)
- 每天 ranking by mean → top 10 因子
- 每个因子 cross-section rank → top 10 股 → 100 次选择投票
- MIN_VOTES=2 阈值过滤 → Dynamic K (5-10)

V13 vs V12 关键改动:
- 因子 ranking 用 **per-stock forward return mean** (不是 std × null_ratio)
- 每只股票评估: factor_value → forward 20d return
- 每天 ranking 用**所有股票 mean forward return** as factor score
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

EXCLUDE_PREFIXES = (
    "v10_1_",   # zigzag / labels (lookahead)
    "idx_",     # index-level (constant per day)
    "alpha_alpha_custom_argmax_recent",
    "alpha_alpha_custom_argmin_recent",
)

TOP_FACTORS = 10
TOP_STOCKS_PER_FACTOR = 10
MIN_VOTES = 2
MIN_STOCKS = 5
MAX_STOCKS = 10
MIN_HOLD_DAYS = 5
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
FWD_DAYS = 20  # forward return horizon for factor ranking
LOT_SIZE = 100  # A股 1 手 = 100 股

_DAILY_PICKS: dict[str, dict[str, float]] = {}


class DailyReselectV13Strategy(aq.Strategy):
    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._last_rebalance_date = None

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]  # e.g. "2020-01-02 00:00:00" → "2020-01-02"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v13")
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
         min_votes=MIN_VOTES, min_stocks=MIN_STOCKS, max_stocks=MAX_STOCKS,
         min_hold_days=MIN_HOLD_DAYS, fwd_days=FWD_DAYS, lot_size=LOT_SIZE)

    panel = pl.read_parquet(PANEL_FILE)
    base_cols = {
        "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
        "adj_factor", "adj_close", "adj_factor_inferred", "pct_chg",
    }
    candidates = []
    for c in panel.columns:
        if c in base_cols:
            continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        candidates.append(c)
    valid_candidates = [c for c in candidates if c in panel.columns]
    emit("panel_loaded", n_rows=panel.height, n_cols=panel.width,
         n_candidates=len(valid_candidates))

    start_dt = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date[:10], "%Y-%m-%d").date()
    # Need buffer for forward return computation (FWD_DAYS ahead)
    buffer_end = end_dt + timedelta(days=FWD_DAYS + 30)

    # Pre-compute forward returns for ALL stocks: close[t+FWD_DAYS] / close[t] - 1
    # Then for each factor, compute per-(date, factor): mean of fwd_ret across stocks
    # (filtering stocks with valid factor value at date t)
    emit("computing_forward_returns", fwd_days=FWD_DAYS)

    # Filter panel: only include dates we need (up to buffer_end)
    sub_all = panel.filter(
        pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    ).select(["trade_date", "ts_code", "close"] + valid_candidates)

    # Compute per-stock fwd_ret: sort by ts_code, then per (ts_code) shift close by FWD_DAYS
    sub_all = sub_all.sort(["ts_code", "trade_date"])
    # Forward return for each (ts_code, trade_date): close shifted by -FWD_DAYS / close - 1
    sub_all = sub_all.with_columns(
        (pl.col("close").shift(-FWD_DAYS).over("ts_code") / pl.col("close") - 1).alias("_fwd_ret")
    )

    # Filter to dates in our backtest range
    sub_all = sub_all.filter(
        pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    n_total = sub_all["trade_date"].n_unique()
    emit("forward_returns_ready", n_total_dates=n_total,
         duration_sec=time.time() - time.time())

    # Phase 1: per (date, factor) — mean of fwd_ret across stocks
    # (where factor is non-null at that date)
    # For each factor: filter rows where factor is not null, group by date, mean of _fwd_ret
    t_start = time.time()
    factor_scores_daily = {}
    # Process per factor for diagnostic visibility (slow if doing many)
    # Actually for speed: compute one column at a time
    for f in valid_candidates:
        # Where factor is not null, what's the mean fwd_ret per date?
        per_factor = (
            sub_all.filter(pl.col(f).is_not_null() & pl.col("_fwd_ret").is_not_null())
            .group_by("trade_date")
            .agg(pl.col("_fwd_ret").mean().alias(f"_f_{f}"))
        )
        for row in per_factor.to_dicts():
            d = str(row["trade_date"])[:10]
            score = row[f"_f_{f}"]
            if d not in factor_scores_daily:
                factor_scores_daily[d] = {}
            factor_scores_daily[d][f] = score
    emit("factor_scores_done", duration_sec=time.time() - t_start)

    # Phase 2: daily top 10 factors by mean fwd_ret
    # Also per factor: per-day top 10 stocks (cross-section rank)
    # Use the precomputed sub_all for stock ranking
    sub_scored = sub_all
    # Add per-factor rank columns (vectorised)
    rank_exprs = []
    for f in valid_candidates:
        rank_exprs.append(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias(f"_rank_{f}")
        )
    sub_scored = sub_scored.with_columns(rank_exprs)

    # For each (date, factor) → top 10 stocks
    per_factor_top_dfs = []
    for f in valid_candidates:
        rank_col = f"_rank_{f}"
        top_df = (
            sub_scored.filter(pl.col(rank_col) <= TOP_STOCKS_PER_FACTOR)
            .select(pl.col("trade_date").alias("date"), pl.col("ts_code").alias("symbol"))
            .with_columns(pl.lit(f).alias("factor"))
        )
        per_factor_top_dfs.append(top_df)

    votes_long = pl.concat(per_factor_top_dfs)
    vote_counts = (
        votes_long.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
    )
    # Apply MIN_VOTES threshold + top-K per date
    qualified = vote_counts.filter(pl.col("votes") >= MIN_VOTES)
    qualified = qualified.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    qualified = qualified.filter(pl.col("_date_rank") <= MAX_STOCKS)

    # BUT only use picks whose factors are in daily top TOP_FACTORS — restrict to top factors
    # Actually the user said daily top 10 factors, but the votes from all factors are
    # equally weighted. To follow user's contract strictly:
    # - On day d, compute top 10 factors from factor_scores_daily[d]
    # - Use ONLY those factors' top-10 stocks for voting
    # - That requires per-day subsetting

    # Restrict voting to top TOP_FACTORS per day
    emit("filtering_to_top_factors", top_factors=TOP_FACTORS)

    # Build dict: date -> set of top factors
    top_factors_per_day = {}
    for d, scores in factor_scores_daily.items():
        # Sort factors by score descending, take top TOP_FACTORS
        sorted_factors = sorted(scores.items(), key=lambda x: -x[1] if x[1] is not None else 0)
        top = [f for f, _ in sorted_factors[:TOP_FACTORS]]
        top_factors_per_day[d] = top

    # Filter votes_long to only top factors per day
    # This is slow in pure Python; build a map date → set, then filter
    # Simpler: filter votes_long via join with daily factor table
    # Build daily_factor_df: (date, factor, mean_weight)
    daily_factor_records = []
    for d, scores in factor_scores_daily.items():
        for f, s in scores.items():
            daily_factor_records.append({"date": d, "factor": f, "score": s})
    daily_factor_df = pl.DataFrame(daily_factor_records)

    # Cast date to datetime for join
    daily_factor_df = daily_factor_df.with_columns(
        pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )

    # Add per-day rank
    daily_factor_df = daily_factor_df.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("date").alias("_factor_rank")
    )
    # Keep only top TOP_FACTORS factors per day
    top_factors_df = daily_factor_df.filter(pl.col("_factor_rank") <= TOP_FACTORS).select(["date", "factor"])

    # Ensure votes_long's date column matches datetime for the join
    if votes_long.schema["date"] == pl.Utf8:
        votes_long = votes_long.with_columns(
            pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        )

    # Join votes with top factors (keep only votes from top factors)
    filtered_votes = votes_long.join(top_factors_df, on=["date", "factor"], how="inner")
    emit("votes_filtered_to_top_factors", n_votes=filtered_votes.height)

    # Re-aggregate
    qualified = (
        filtered_votes.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
        .filter(pl.col("votes") >= MIN_VOTES)
    )
    qualified = qualified.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    qualified = qualified.filter(pl.col("_date_rank") <= MAX_STOCKS)

    _DAILY_PICKS.clear()
    daily_picks_log = []
    for row in qualified.sort(["date", "votes"], descending=[False, True]).to_dicts():
        d_raw = row["date"]
        # Convert datetime to ISO string for consistency with strategy's date_str
        if hasattr(d_raw, "strftime"):
            d = d_raw.strftime("%Y-%m-%d")
        else:
            d = str(d_raw)[:10]
        sym = row["symbol"]
        v = int(row["votes"])
        if d not in _DAILY_PICKS:
            _DAILY_PICKS[d] = {}
        _DAILY_PICKS[d][sym] = float(v)
        daily_picks_log.append({
            "date": d, "symbol": sym, "votes": v,
        })

    from collections import Counter
    per_day_count = Counter(r["date"] for r in daily_picks_log)
    n_days_with_picks = len(per_day_count)
    avg_stocks_per_day = sum(per_day_count.values()) / max(n_days_with_picks, 1)
    emit("picks_phase_done", n_days=n_days_with_picks,
         avg_stocks_per_day=avg_stocks_per_day)

    # Phase 3: build data dict + AKQuant
    all_picks_universe = sorted({sym for picks in _DAILY_PICKS.values() for sym in picks.keys()})
    emit("universe_size", n_unique_stocks=len(all_picks_universe))

    warmup_start = start_dt - timedelta(days=30)
    sub_data = panel.filter(
        (pl.col("trade_date") >= pl.lit(warmup_start.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & pl.col("ts_code").is_in(all_picks_universe)
    ).select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"])

    data_dict: dict[str, pd.DataFrame] = {}
    for code in all_picks_universe:
        pdf = (
            sub_data.filter(pl.col("ts_code") == code)
            .sort("trade_date").to_pandas()
            .set_index("trade_date").rename_axis("date")
        )
        if pdf.empty:
            continue
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1.0e9
        data_dict[code] = pdf
    emit("data_dict_built", n_symbols=len(data_dict))

    emit("akquant_start")
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyReselectV13Strategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=LOT_SIZE,
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

    # Extract ledger
    def _extract(obj):
        if isinstance(obj, list):
            ledger = []
            for x in obj:
                if isinstance(x, dict):
                    ledger.append({k: (v.isoformat() if hasattr(v, "isoformat") else
                                       (v.item() if hasattr(v, "item") else v))
                                  for k, v in x.items()})
                    continue
                rec = {}
                for attr in dir(x):
                    if attr.startswith("_"):
                        continue
                    try:
                        v = getattr(x, attr)
                    except Exception:
                        continue
                    if callable(v):
                        continue
                    if hasattr(v, "isoformat"):
                        v = v.isoformat()
                    elif hasattr(v, "item"):
                        try:
                            v = v.item()
                        except (ValueError, TypeError):
                            v = str(v)
                    elif str(type(v).__name__) in ("OrderSide", "OrderStatus", "OrderType",
                                                     "OrderRole", "PositionEffect", "TimeInForce"):
                        v = str(v)
                    rec[attr] = v
                ledger.append(rec)
            return ledger
        return []

    trades_ledger = _extract(result.trades)
    orders_ledger = _extract(result.orders)

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
        "factor_candidates": candidates,
        "valid_candidates": valid_candidates,
        "top_factors": TOP_FACTORS,
        "top_stocks_per_factor": TOP_STOCKS_PER_FACTOR,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
        "min_hold_days": MIN_HOLD_DAYS,
        "fwd_days": FWD_DAYS,
        "lot_size": LOT_SIZE,
        "n_days_with_picks": n_days_with_picks,
        "avg_stocks_per_day": avg_stocks_per_day,
        "daily_picks": daily_picks_log,
        "nav_curve": nav_curve,
        "trades_ledger": trades_ledger,
        "orders_ledger": orders_ledger,
        "n_trades": len(trades_ledger),
        "n_orders": len(orders_ledger),
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         n_days=n_total, n_days_with_picks=n_days_with_picks,
         avg_stocks_per_day=avg_stocks_per_day)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()