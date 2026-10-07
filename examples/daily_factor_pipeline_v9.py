"""Daily Factor Reselect v9 — simplified:
- Hard-code Stage 5b 6 factors (already verified alpha +134% annualised)
- Daily cross-section percentile-rank on those 6 factors → composite → top 10 stocks
- AKQuant single continuous backtest; rebalance only when picks change AND
  MIN_HOLD_DAYS has elapsed
- NO IC computation (was too slow: 30 days took 12+ minutes)

User's contract:
- 每天选股 (use top 6 factors from Stage 5b proven)
- 只在 picks 变化时调仓
- MIN_HOLD_DAYS guard to reduce turnover
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

# Stage 5b 6 factors — verified +134% annualised in v1+v2+v2c workflow
STAGE5B_FACTORS = [
    "talib_NATR", "talib_TRANGE", "gtja_gtja_159",
    "gtja_gtja_149", "gtja_gtja_144", "mw_vol_20d",
]
TOP_STOCKS = 10
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
MIN_HOLD_DAYS = 5

# Pre-computed daily picks: {date_str: picks_dict}
_DAILY_PICKS: dict[str, dict[str, float]] = {}


class DailyReselectV9Strategy(aq.Strategy):
    """Daily: check picks, rebalance only if changed AND min_hold elapsed."""

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
            return  # hold
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
                    return  # small change → hold

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


def select_top_stocks_v9(panel, factors, eval_date, top_k=TOP_STOCKS):
    """Cross-section percentile rank on Stage 5b 6 factors → composite sum → top K stocks."""
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    sub = panel.filter(
        pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)

    valid = [f for f in factors if f in sub.columns]
    if not valid:
        return [], {}

    df = sub.select(["ts_code"] + valid)
    # Vectorised: drop rows with any null in factors
    df = df.filter(~pl.any_horizontal([pl.col(f).is_null() for f in valid]))
    if df.height == 0:
        return [], {}

    score_cols = []
    for f in valid:
        df = df.with_columns(
            (pl.col(f).rank(method="ordinal", descending=True) / pl.col(f).count()).alias(f"_pr_{f}")
        )
        score_cols.append(f"_pr_{f}")
    df = df.with_columns(pl.sum_horizontal(score_cols).alias("_composite"))
    sorted_sub = df.sort("_composite", descending=True, nulls_last=True).head(top_k)
    picks = sorted_sub["ts_code"].to_list()
    scores = {row["ts_code"]: float(row["_composite"]) for row in sorted_sub.to_dicts()}
    return picks, scores


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
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v9")
    args = parser.parse_args()

    job_dir = Path(args.out_base) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_log = job_dir / "progress.jsonl"

    def emit(stage, **kw):
        rec = {"stage": stage, "job_id": args.job_id, "timestamp": datetime.now().isoformat(), **kw}
        with open(progress_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, ensure_ascii=False))

    emit("init", start_date=args.start_date, end_date=args.end_date)
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

    # Phase 1: precompute daily picks — vectorised over all dates at once
    t_start = time.time()
    # Filter panel to dates of interest, then per-date cross-section rank
    valid = [f for f in STAGE5B_FACTORS if f in panel.columns]
    sub_all = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).select(["trade_date", "ts_code"] + valid)
    # Filter out rows where any factor is null
    sub_all = sub_all.filter(~pl.any_horizontal([pl.col(f).is_null() for f in valid]))
    emit("pre_filter_done", n_rows=sub_all.height, elapsed_sec=time.time() - t_start)

    # Per-date cross-section rank for each factor, then sum
    score_exprs = []
    for f in valid:
        score_exprs.append(
            (pl.col(f).rank(method="ordinal", descending=True).over("trade_date")
             / pl.col(f).count().over("trade_date")).alias(f"_pr_{f}")
        )
    sub_scored = sub_all.with_columns(score_exprs)
    sub_scored = sub_scored.with_columns(
        pl.sum_horizontal([f"_pr_{f}" for f in valid]).alias("_composite")
    )
    # Per date: take top-K by composite
    sub_scored = sub_scored.with_columns(
        pl.col("_composite").rank(method="ordinal", descending=True).over("trade_date").alias("_rank")
    )
    top_per_date = sub_scored.filter(pl.col("_rank") <= TOP_STOCKS)
    emit("scoring_done", elapsed_sec=time.time() - t_start)

    daily_picks_log = []
    for row in top_per_date.sort(["trade_date", "_rank"]).to_dicts():
        d = str(row["trade_date"])[:10]
        sym = row["ts_code"]
        if d not in _DAILY_PICKS:
            _DAILY_PICKS[d] = {}
        _DAILY_PICKS[d][sym] = float(row["_composite"])
        daily_picks_log.append({"date": d, "picks": [], "scores": {sym: float(row["_composite"])}})

    emit("picks_phase_done", n_days=len(_DAILY_PICKS), duration_sec=time.time() - t_start)

    # Phase 2: build data dict for ALL stocks that ever appear in picks
    all_picks_universe = sorted({sym for picks in _DAILY_PICKS.values() for sym in picks.keys()})
    emit("universe_size", n_unique_stocks=len(all_picks_universe))

    warmup_start = start_dt - timedelta(days=30)
    data_dict = build_panel_data_dict(panel, all_picks_universe, warmup_start, end_dt)
    if not data_dict:
        emit("error", msg="no data in range")
        return
    emit("data_dict_built", n_symbols=len(data_dict))

    # Phase 3: single AKQuant run_backtest
    emit("akquant_start")
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyReselectV9Strategy,
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

    # Extract equity curve (NAV curve) if available
    nav_curve = []
    try:
        eq = result.equity_curve
        if hasattr(eq, "items"):
            for ts, val in eq.items():
                nav_curve.append([str(ts)[:10], val])
        elif isinstance(eq, list):
            for pt in eq:
                nav_curve.append([str(pt[0])[:10], float(pt[1])])
    except Exception as exc:
        emit("equity_curve_failed", error=str(exc))

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
        "top_factors": STAGE5B_FACTORS,
        "top_stocks": TOP_STOCKS,
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