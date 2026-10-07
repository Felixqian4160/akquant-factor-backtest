"""Daily Factor Reselect v14 — Rolling IC (Information Coefficient) factor ranking

User's contract:
- 因子 ranking 用过去 60 天 rolling IC mean (Spearman rank correlation: 因子值 vs forward 20d return)
- 每天 ranking by rolling IC mean → top 10 因子
- 每个 top 因子 cross-section rank → top 10 股 → 投票
- MIN_VOTES=2 过滤 → Dynamic K (5-10)
- AKQuant single continuous backtest with lot_size=100
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
    "v10_1_",
    "idx_",
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
FWD_DAYS = 20
IC_WINDOW_DAYS = 60  # rolling window for IC computation
LOT_SIZE = 100

_DAILY_PICKS: dict[str, dict[str, float]] = {}
_DAILY_REGIME: dict[str, str] = {}  # date -> "bull" / "bear" / "neutral"
_MIN_HOLD_DAYS: int = 5  # set by main before AKQuant runs
_MIN_CHANGE_RATIO: float = 0.5


class DailyReselectV14Strategy(aq.Strategy):
    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._last_rebalance_date = None

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]
        regime = _DAILY_REGIME.get(date_str, "bull")

        # Regime filter: bear regime → go to cash (no picks)
        if regime == "bear":
            if self._current_picks:
                # Liquidate all positions one-by-one
                for sym in list(self._current_picks):
                    try:
                        self.close_position(sym)
                    except Exception as exc:
                        self.log(f"close {sym} failed: {exc}")
                self._current_picks = set()
                self._last_rebalance_date = date_str
                self.log(f"regime=BEAR → go to cash on {date_str}")
            return

        picks = _DAILY_PICKS.get(date_str, {})
        new_set = set(picks.keys())
        if new_set == self._current_picks:
            return
        if not new_set:
            return

        if self._last_rebalance_date is not None and _MIN_HOLD_DAYS > 0:
            days_since = (
                datetime.strptime(date_str, "%Y-%m-%d")
                - datetime.strptime(self._last_rebalance_date, "%Y-%m-%d")
            ).days
            if days_since < _MIN_HOLD_DAYS:
                # Only rebalance if picks changed significantly
                overlap = len(new_set & self._current_picks)
                change_ratio = (len(new_set) - overlap) / max(len(new_set), 1)
                if change_ratio < _MIN_CHANGE_RATIO:
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


def compute_rolling_ic(sub_panel, factors, fwd_days=20, window=60):
    """Compute rolling IC per (date, factor) — Spearman correlation
    of factor_value vs forward return across all stocks on a given date,
    then take mean over rolling window.

    For each date d:
      - Look back window days
      - For each (factor, day) in window: compute Spearman corr(factor_value, fwd_ret)
      - Mean across window days = rolling IC for factor on date d
    """
    # Forward returns (already computed)
    sub_panel = sub_panel.with_columns(
        (pl.col("close").shift(-fwd_days).over("ts_code") / pl.col("close") - 1).alias("_fwd_ret")
    )

    # Sort by date
    sub_panel = sub_panel.sort(["trade_date", "ts_code"])

    # Compute per-day per-factor IC: for each (date, factor), Spearman corr
    # Use vectorised approach: rank within date, then correlation across stocks
    # For speed: convert to pandas and use pd.DataFrame.corr per group
    # Actually, can vectorise with polars:
    # 1. Add rank columns for factor values per date
    # 2. Add rank column for fwd_ret per date
    # 3. Pearson corr of ranks = Spearman

    rank_exprs = []
    for f in factors:
        rank_exprs.append(
            pl.col(f).rank(method="average").over("trade_date").alias(f"_f_rank_{f}")
        )
    rank_exprs.append(
        pl.col("_fwd_ret").rank(method="average").over("trade_date").alias("_y_rank")
    )
    sub_panel = sub_panel.with_columns(rank_exprs)

    # Now compute per (date, factor) Spearman corr:
    # corr(X_rank, Y_rank) per date
    # Use polars: group_by date, then compute corr
    # But polars corr aggregation needs both columns accessible — use formula
    # corr(x, y) = (mean(xy) - mean(x)*mean(y)) / (std(x) * std(y))
    # For ranks: use Pearson formula on ranks = Spearman

    # Vectorised per (date, factor): use pl.corr in group_by.agg
    ic_records = []
    for f in factors:
        rank_col = f"_f_rank_{f}"
        agg = (
            sub_panel.group_by("trade_date")
            .agg(pl.corr(pl.col(rank_col), pl.col("_y_rank")).alias("_ic"))
            .rename({"_ic": "_factor_ic"})
            .with_columns(pl.lit(f).alias("factor"))
            .select(["trade_date", "factor", "_factor_ic"])
        )
        ic_records.append(agg)

    ic_df = pl.concat(ic_records)

    # Rolling mean of IC per factor
    ic_df = ic_df.sort(["factor", "trade_date"])
    ic_df = ic_df.with_columns(
        pl.col("_factor_ic").rolling_mean(window).over("factor").alias("_rolling_ic")
    )
    return ic_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v14")
    parser.add_argument("--ic-window-days", type=int, default=IC_WINDOW_DAYS)
    parser.add_argument("--min-hold-days", type=int, default=MIN_HOLD_DAYS)
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
         min_votes=MIN_VOTES, ic_window_days=args.ic_window_days,
         min_hold_days=args.min_hold_days,
         fwd_days=FWD_DAYS, lot_size=LOT_SIZE)

    # Set module-level state for strategy
    global _MIN_HOLD_DAYS
    _MIN_HOLD_DAYS = args.min_hold_days

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

    # Need extra buffer for IC window + forward returns
    buffer_start = start_dt - timedelta(days=IC_WINDOW_DAYS + 30)
    buffer_end = end_dt + timedelta(days=FWD_DAYS + 30)

    # Filter panel to buffer range
    sub_all = panel.filter(
        (pl.col("trade_date") >= pl.lit(buffer_start.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(buffer_end.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).select(["trade_date", "ts_code", "close"] + valid_candidates)

    # Compute per-stock forward returns
    sub_all = sub_all.sort(["ts_code", "trade_date"])
    sub_all = sub_all.with_columns(
        (pl.col("close").shift(-FWD_DAYS).over("ts_code") / pl.col("close") - 1).alias("_fwd_ret")
    )

    # Compute rolling IC per factor
    emit("computing_rolling_ic", factors=len(valid_candidates),
         window=args.ic_window_days, fwd_days=FWD_DAYS)
    t0 = time.time()
    ic_df = compute_rolling_ic(sub_all, valid_candidates, FWD_DAYS, args.ic_window_days)
    emit("rolling_ic_done", duration_sec=time.time() - t0)

    # Build daily factor score (rolling_ic)
    # For each date, rank factors by rolling_ic (descending)
    # Then take top TOP_FACTORS

    # Filter to backtest range (after IC warmup)
    warmup_threshold = start_dt + timedelta(days=10)  # need a few days for rolling
    sub_picks_window = (
        ic_df.filter(pl.col("trade_date") >= pl.lit(warmup_threshold.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        .filter(pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        .with_columns(
            pl.col("_rolling_ic").rank(method="ordinal", descending=True).over("trade_date").alias("_factor_rank")
        )
    )
    top_factors_per_day = sub_picks_window.filter(pl.col("_factor_rank") <= TOP_FACTORS)
    emit("top_factors_per_day_built", n_rows=top_factors_per_day.height)

    # Phase 2: For each date, top factor → top 10 stocks (cross-section rank)
    sub_data_window = sub_all.filter(
        pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    sub_data_window = sub_data_window.filter(
        pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )

    # Build votes: for each (date, top factor), get top 10 stocks
    # Vectorised: for each top factor, compute cross-section rank
    rank_exprs = []
    for f in valid_candidates:
        rank_exprs.append(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias(f"_rank_{f}")
        )
    sub_data_window = sub_data_window.with_columns(rank_exprs)

    # Get all top factors × top 10 stocks per day
    # For each factor in top_factors_per_day, filter rows where rank_col <= 10
    per_factor_top_dfs = []
    for f in valid_candidates:
        rank_col = f"_rank_{f}"
        top_df = (
            sub_data_window.filter(pl.col(rank_col) <= TOP_STOCKS_PER_FACTOR)
            .select(pl.col("trade_date").alias("date"), pl.col("ts_code").alias("symbol"))
            .with_columns(pl.lit(f).alias("factor"))
        )
        per_factor_top_dfs.append(top_df)

    votes_long = pl.concat(per_factor_top_dfs)

    # Filter votes to only top factors per day
    # Cast both date columns to datetime for join
    if votes_long.schema["date"] == pl.Utf8:
        votes_long = votes_long.with_columns(
            pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        )

    top_factors_per_day_for_join = top_factors_per_day.rename({
        "trade_date": "date",
        "factor": "factor",
    }).select(["date", "factor"])

    if top_factors_per_day_for_join.schema["date"] == pl.Utf8:
        top_factors_per_day_for_join = top_factors_per_day_for_join.with_columns(
            pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        )

    filtered_votes = votes_long.join(
        top_factors_per_day_for_join, on=["date", "factor"], how="inner"
    )
    emit("votes_filtered", n_votes=filtered_votes.height)

    # Aggregate and threshold
    qualified = (
        filtered_votes.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
        .filter(pl.col("votes") >= MIN_VOTES)
    )
    qualified = qualified.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    qualified = qualified.filter(pl.col("_date_rank") <= MAX_STOCKS)

    # Build _DAILY_PICKS
    _DAILY_PICKS.clear()
    _DAILY_REGIME.clear()
    daily_picks_log = []
    for row in qualified.sort(["date", "votes"], descending=[False, True]).to_dicts():
        d_raw = row["date"]
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

    # Compute daily regime using idx_ret_60d (60-day index return)
    # bull: idx_ret_60d > BULL_THRESHOLD (default 0.05)
    # bear: idx_ret_60d < BEAR_THRESHOLD (default -0.05)
    # else neutral (still trade)
    regime_records = []
    sub_idx = panel.filter(
        pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    ).filter(
        pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    ).group_by("trade_date").agg(pl.col("idx_ret_60d").first()).sort("trade_date")

    BULL_THRESHOLD = 0.05
    BEAR_THRESHOLD = -0.05
    n_bull, n_bear, n_neutral = 0, 0, 0
    for row in sub_idx.to_dicts():
        d = str(row["trade_date"])[:10]
        v = row.get("idx_ret_60d")
        if v is None:
            regime = "neutral"
            n_neutral += 1
        elif v > BULL_THRESHOLD:
            regime = "bull"
            n_bull += 1
        elif v < BEAR_THRESHOLD:
            regime = "bear"
            n_bear += 1
        else:
            regime = "neutral"
            n_neutral += 1
        _DAILY_REGIME[d] = regime
        regime_records.append({"date": d, "regime": regime, "idx_ret_60d": v})
    emit("regime_computed", n_bull=n_bull, n_bear=n_bear, n_neutral=n_neutral,
         bull_threshold=BULL_THRESHOLD, bear_threshold=BEAR_THRESHOLD)

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
            strategy=DailyReselectV14Strategy,
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
        "metrics": metrics_dict,
        "factor_candidates": candidates,
        "valid_candidates": valid_candidates,
        "top_factors": TOP_FACTORS,
        "top_stocks_per_factor": TOP_STOCKS_PER_FACTOR,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
        "min_hold_days": args.min_hold_days,
        "ic_window_days": args.ic_window_days,
        "fwd_days": FWD_DAYS,
        "lot_size": LOT_SIZE,
        "n_days_with_picks": n_days_with_picks,
        "avg_stocks_per_day": avg_stocks_per_day,
        "daily_picks": daily_picks_log,
        "daily_regime": regime_records,
        "n_bull_days": n_bull,
        "n_bear_days": n_bear,
        "n_neutral_days": n_neutral,
        "nav_curve": nav_curve,
        "trades_ledger": trades_ledger,
        "orders_ledger": orders_ledger,
        "n_trades": len(trades_ledger),
        "n_orders": len(orders_ledger),
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         n_days_with_picks=n_days_with_picks,
         avg_stocks_per_day=avg_stocks_per_day)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()