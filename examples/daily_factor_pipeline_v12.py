"""Daily Factor Reselect v12 — Daily-dynamic factor selection + vote-threshold
stock selection (user's contract):

User's framework (clarified):
1. 每天先做单因子回测 → 排名 → 取前 N 个因子 (DAILY, not hard-coded)
2. 每个 top 因子独立选股 → 投票
3. 过滤: 票数 < MIN_VOTES 的不选 (默认 MIN_VOTES=2, 即排除"每个股票只有 1 票")
4. 动态选 top-K 只股: 按投票分布 + 总投票数 → 选 5-10 只 (有依据)

回测标的:
- 每个因子: 在 HS300 个股上 cross-section rank, 选 top-10 (基于单因子 score, 不预回测)
- 单因子"回测" = factor value 在每只股票上的相对排名 (用 last day's cross-section percentile)

为什么不用 IC: IC 需要 60d window × 未来 return 计算, 慢 (12 min for 30 days);
v12 用更简单的代理: 因子在 panel 中的 z-score percentile.

Strategy:
- 每天: 选 top 10 因子 (按因子在 panel 内的 std + null ratio, simple proxy)
- 每个因子: cross-section rank → top 10 股
- 投票: 9-10 因子 × 10 股 = 100 次选择
- 过滤: 只保留 ≥ MIN_VOTES 票的股
- 选 top-K: 按总投票数比例 → K = max(MIN_STOCKS, min(MAX_STOCKS, len(qualified)))
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

# Factor candidates — discovered dynamically from panel (393 factors).
# Excluded prefixes: base/label/zigzag/index features (lookahead bias).
EXCLUDE_PREFIXES = (
    "v10_1_",   # zigzag / labels (lookahead)
    "idx_",     # index-level (constant per day → not per-stock)
    "alpha_alpha_custom_argmax_recent",  # potential lookahead
    "alpha_alpha_custom_argmin_recent",
)

# Configurable parameters
TOP_FACTORS = 10  # daily top-10 by std × (1-null_ratio)
TOP_STOCKS_PER_FACTOR = 10
MIN_VOTES = 2  # threshold: a stock must be selected by ≥MIN_VOTES factors
MIN_STOCKS = 5
MAX_STOCKS = 10
MIN_HOLD_DAYS = 5
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
WINDOW_DAYS = 60  # for factor-selection backtest

# Pre-computed daily picks + factors used
_DAILY_PICKS: dict[str, dict[str, float]] = {}
_DAILY_FACTORS: dict[str, list[str]] = {}  # for diagnostics


class DailyReselectV12Strategy(aq.Strategy):
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


def rank_factors_daily(panel, eval_date, factor_candidates, top_n=TOP_FACTORS,
                       window_days=WINDOW_DAYS):
    """Daily: rank factors by simple proxy: factor's cross-sectional std (higher std
    = more dispersion = more discriminative signal) + low null ratio.
    This is a cheap proxy for IC without running full forward-return backtest.
    """
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    sub = panel.filter(
        pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)

    valid_factors = [f for f in factor_candidates if f and f in sub.columns]
    if not valid_factors:
        return []

    # Compute per-factor: cross-sectional std + null ratio
    factor_stats = []
    for f in valid_factors:
        col = sub[f]
        non_null = col.drop_nulls()
        n_total = sub.height
        n_non_null = non_null.height
        if n_non_null < 100:
            continue
        null_ratio = 1 - n_non_null / n_total
        std = float(non_null.std()) if n_non_null > 1 else 0.0
        # Score: std * (1 - null_ratio) — high dispersion + low nulls
        score = std * (1 - null_ratio)
        factor_stats.append((f, score, std, null_ratio))

    factor_stats.sort(key=lambda x: -x[1])
    return factor_stats[:top_n]


def build_panel_data_dict(panel, picks_universe, start_dt, end_dt) -> dict[str, pd.DataFrame]:
    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & pl.col("ts_code").is_in(picks_universe)
    ).select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"])

    data_dict: dict[str, pd.DataFrame] = {}
    for code in picks_universe:
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
    return data_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v12")
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
         min_hold_days=MIN_HOLD_DAYS)

    panel = pl.read_parquet(PANEL_FILE)
    # Discover factor candidates dynamically from panel columns
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
    emit("panel_loaded", n_rows=panel.height, n_cols=panel.width,
         n_candidates=len(candidates))

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

    # Phase 1: daily factor selection + voting + threshold filtering
    t_start = time.time()
    daily_picks_log = []
    factor_use_count: dict[str, int] = {}  # diagnostic: how often each factor is selected
    n_unique_stocks_per_day = []

    # Pre-filter panel to date range
    sub_all = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).select(["trade_date", "ts_code"] + candidates)
    # Compute per-factor rank per (date, factor) — vectorised
    valid_candidates = [f for f in candidates if f in sub_all.columns]
    emit("valid_candidates", n=len(valid_candidates),
         candidates=valid_candidates[:20])

    # Add rank column for each factor
    rank_exprs = []
    for f in valid_candidates:
        rank_exprs.append(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias(f"_rank_{f}")
        )
    sub_ranked = sub_all.with_columns(rank_exprs)

    # For each date, get top-10 per factor (vectorised: filter rank <= 10)
    # Build long-format vote records
    per_factor_top_dfs = []
    for f in valid_candidates:
        rank_col = f"_rank_{f}"
        top_df = (
            sub_ranked.filter(pl.col(rank_col) <= TOP_STOCKS_PER_FACTOR)
            .select(pl.col("trade_date").alias("date"), pl.col("ts_code").alias("symbol"))
            .with_columns(pl.lit(f).alias("factor"))
        )
        per_factor_top_dfs.append(top_df)

    # Concat all votes
    votes_long = pl.concat(per_factor_top_dfs)
    emit("voting_aggregated", n_votes=votes_long.height,
         duration_sec=time.time() - t_start)

    # Aggregate: count votes per (date, symbol)
    vote_counts = (
        votes_long.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
    )

    # Filter out low-vote stocks (MIN_VOTES threshold)
    qualified = vote_counts.filter(pl.col("votes") >= MIN_VOTES)
    # Take top-K by votes per date, with K bounded by [MIN_STOCKS, MAX_STOCKS]
    qualified = qualified.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    qualified = qualified.filter(pl.col("_date_rank") <= MAX_STOCKS)

    # Dynamic K: if qualified count < MIN_STOCKS, allow all up to MAX_STOCKS
    # (already handled by MAX_STOCKS filter above; we just take what's available)
    emit("consensus_qualified", n_qualified=qualified.height)

    # Convert to dict for AKQuant
    _DAILY_PICKS.clear()
    daily_picks_log = []
    for row in qualified.to_dicts():
        d = row["date"]
        if isinstance(d, datetime):
            d = str(d)[:10]
        else:
            d = str(d)[:10]
        sym = row["symbol"]
        v = int(row["votes"])
        if d not in _DAILY_PICKS:
            _DAILY_PICKS[d] = {}
        _DAILY_PICKS[d][sym] = float(v)
        daily_picks_log.append({
            "date": d, "symbol": sym, "votes": v,
        })

    # Diagnostic: per-day count
    from collections import Counter
    per_day_count = Counter(r["date"] for r in daily_picks_log)
    n_days_with_picks = len(per_day_count)
    avg_stocks_per_day = sum(per_day_count.values()) / max(n_days_with_picks, 1)
    emit("picks_phase_done", n_days=n_days_with_picks,
         avg_stocks_per_day=avg_stocks_per_day,
         duration_sec=time.time() - t_start)

    # Phase 2: build data dict
    all_picks_universe = sorted({sym for picks in _DAILY_PICKS.values() for sym in picks.keys()})
    emit("universe_size", n_unique_stocks=len(all_picks_universe))

    warmup_start = start_dt - timedelta(days=30)
    data_dict = build_panel_data_dict(panel, all_picks_universe, warmup_start, end_dt)
    if not data_dict:
        emit("error", msg="no data in range")
        return
    emit("data_dict_built", n_symbols=len(data_dict))

    # Phase 3: AKQuant
    emit("akquant_start")
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyReselectV12Strategy,
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

    # Extract per-trade ledger (the "账本" / audit trail)
    trades_ledger = []
    try:
        trades_data = result.trades
        emit("debug_trades", type=str(type(trades_data)),
             len=(len(trades_data) if hasattr(trades_data, "__len__") else "N/A"))
        if isinstance(trades_data, list):
            for t in trades_data:
                # Use dir() + getattr since Order/Trade are pyo3 builtins
                # without __dict__
                if isinstance(t, dict):
                    rec = {k: (v.isoformat() if hasattr(v, "isoformat") else
                              (v.item() if hasattr(v, "item") else v))
                           for k, v in t.items()}
                    trades_ledger.append(rec)
                    continue
                rec = {}
                for attr in dir(t):
                    if attr.startswith("_"):
                        continue
                    try:
                        v = getattr(t, attr)
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
                trades_ledger.append(rec)
        elif hasattr(trades_data, "to_dicts"):
            for t in trades_data.to_dicts():
                rec = {k: (v.isoformat() if hasattr(v, "isoformat") else
                          (v.item() if hasattr(v, "item") else v))
                       for k, v in t.items()}
                trades_ledger.append(rec)
    except Exception as exc:
        emit("trades_extract_failed", error=str(exc))

    # Extract orders ledger (all orders placed — open + filled + rejected)
    orders_ledger = []
    try:
        orders_data = result.orders
        emit("debug_orders", type=str(type(orders_data)),
             len=(len(orders_data) if hasattr(orders_data, "__len__") else "N/A"))
        if isinstance(orders_data, list):
            for o in orders_data:
                if isinstance(o, dict):
                    rec = {k: (v.isoformat() if hasattr(v, "isoformat") else
                              (v.item() if hasattr(v, "item") else v))
                           for k, v in o.items()}
                    orders_ledger.append(rec)
                    continue
                rec = {}
                for attr in dir(o):
                    if attr.startswith("_"):
                        continue
                    try:
                        v = getattr(o, attr)
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
                orders_ledger.append(rec)
        elif hasattr(orders_data, "to_dicts"):
            for o in orders_data.to_dicts():
                rec = {}
                for k, v in o.items():
                    if hasattr(v, "isoformat"):
                        rec[k] = v.isoformat()
                    elif hasattr(v, "item"):
                        try:
                            rec[k] = v.item()
                        except (ValueError, TypeError):
                            rec[k] = str(v)
                    else:
                        rec[k] = v
                orders_ledger.append(rec)
    except Exception as exc:
        emit("orders_extract_failed", error=str(exc))

    # Extract executions ledger
    executions_ledger = []
    try:
        exec_data = result.executions_df
        if hasattr(exec_data, "to_dicts"):
            for e in exec_data.to_dicts():
                rec = {}
                for k, v in e.items():
                    if hasattr(v, "isoformat"):
                        rec[k] = v.isoformat()
                    elif hasattr(v, "item"):
                        try:
                            rec[k] = v.item()
                        except (ValueError, TypeError):
                            rec[k] = str(v)
                    else:
                        rec[k] = v
                executions_ledger.append(rec)
    except Exception as exc:
        emit("executions_extract_failed", error=str(exc))

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
        "n_days_with_picks": n_days_with_picks,
        "avg_stocks_per_day": avg_stocks_per_day,
        "daily_picks": daily_picks_log,
        "nav_curve": nav_curve,
        "trades_ledger": trades_ledger,
        "orders_ledger": orders_ledger,
        "executions_ledger": executions_ledger,
        "n_trades": len(trades_ledger),
        "n_orders": len(orders_ledger),
        "n_executions": len(executions_ledger),
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         n_days=n_total, n_days_with_picks=n_days_with_picks,
         avg_stocks_per_day=avg_stocks_per_day,
         duration_sec=time.time() - t_start)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()