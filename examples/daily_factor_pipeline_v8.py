"""Daily Factor Reselect v8 — 真正的用户原意 + 修复 v7 的 cycle-overlap bug:

- Single continuous AKQuant backtest over the whole period
- on_cross_section called daily; if picks changed → rebalance_to_topn (the
  new picks) with liquidate_unmentioned=True
- If picks unchanged → hold (no action)
- Per-day NAV tracked via result.metrics_df or by querying account

This avoids the v7 bug where every rebalance opened a NEW 20-day cycle and
multiple cycles overlapped, causing margin starvation.
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

STAGE5B_FACTORS = [
    "talib_NATR", "talib_TRANGE", "gtja_gtja_159",
    "gtja_gtja_149", "gtja_gtja_144", "mw_vol_20d",
]
TOP_FACTORS = 5
TOP_STOCKS = 10
WINDOW_DAYS = 60
FWD_DAYS = 20
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
IC_WINDOW_DAYS = 60
MIN_HOLD_DAYS = 5  # minimum days to hold a position before considering rebalance

# Pre-computed daily picks: {date_str: picks_dict}
_DAILY_PICKS: dict[str, dict[str, float]] = {}
# Pre-computed NAV curve points
_NAV_LOG: list[tuple[str, float]] = []


class DailyReselectV8Strategy(aq.Strategy):
    """Single continuous backtest. Daily on_cross_section:
    - look up today's precomputed picks
    - if they differ from current positions → rebalance
    - else hold
    """
    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._last_rebalance_date = None  # track last rebalance date

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

        # MIN_HOLD_DAYS guard: skip rebalance if last rebalance was too recent
        # UNLESS picks changed significantly (>50% different)
        if self._last_rebalance_date is not None and MIN_HOLD_DAYS > 0:
            days_since = (
                datetime.strptime(date_str, "%Y-%m-%d")
                - datetime.strptime(self._last_rebalance_date, "%Y-%m-%d")
            ).days
            if days_since < MIN_HOLD_DAYS:
                # Only rebalance if >50% of picks changed (significant shift)
                overlap = len(new_set & self._current_picks)
                if overlap >= len(new_set) * 0.5:
                    return  # small change → hold despite MIN_HOLD_DAYS

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


def compute_factor_ic(
    panel: pl.DataFrame, factor: str, as_of_date: str,
    window_days: int = IC_WINDOW_DAYS, fwd_days: int = FWD_DAYS,
) -> float:
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
    window_start = as_of - timedelta(days=window_days)
    fwd_end = as_of + timedelta(days=fwd_days)

    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(window_start.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(fwd_end.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).filter(pl.col(factor).is_not_null())

    if sub.height < 100:
        return 0.0

    as_of_sub = (
        sub.filter(pl.col("trade_date") <= pl.lit(as_of.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        .group_by("ts_code")
        .agg(pl.col(factor).last().alias("factor_val"), pl.col("close").last().alias("px_at_eval"))
    )
    future_target_date = pl.lit((as_of + timedelta(days=fwd_days)).date().isoformat()).str.strptime(
        pl.Datetime("ms"), "%Y-%m-%d"
    )
    future_sub = sub.filter(pl.col("trade_date") >= future_target_date).group_by("ts_code").agg(
        pl.col("close").first().alias("px_future")
    )
    joined = as_of_sub.join(future_sub, on="ts_code", how="inner").with_columns(
        (pl.col("px_future") / pl.col("px_at_eval") - 1).alias("fwd_ret")
    )
    joined = joined.filter(pl.col("fwd_ret").is_not_null() & pl.col("factor_val").is_not_null())
    if joined.height < 50:
        return 0.0
    arr = joined.select(["factor_val", "fwd_ret"]).to_numpy()
    if arr.shape[0] < 50:
        return 0.0
    f_rank = np.argsort(np.argsort(arr[:, 0]))
    r_rank = np.argsort(np.argsort(arr[:, 1]))
    if np.std(f_rank) == 0 or np.std(r_rank) == 0:
        return 0.0
    ic = float(np.corrcoef(f_rank, r_rank)[0, 1])
    return ic if np.isfinite(ic) else 0.0


def rank_factors_by_ic(panel, eval_date, factors):
    ics = [(f, compute_factor_ic(panel, f, eval_date)) for f in factors]
    ics.sort(key=lambda x: -abs(x[1]))
    return ics


def select_top_stocks_ic_weighted(panel, factors_with_ic, eval_date, top_k=TOP_STOCKS):
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    sub = panel.filter(pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)

    valid = [(f, ic) for f, ic in factors_with_ic if f in sub.columns]
    if not valid:
        return [], {}
    df = sub.select(["ts_code"] + [f for f, _ in valid]).drop_nulls()
    if df.height == 0:
        return [], {}
    score_cols = []
    for f, ic in valid:
        w = abs(ic)
        if ic >= 0:
            df = df.with_columns((pl.col(f).rank(method="ordinal", descending=True) * w).alias(f"_sc_{f}"))
        else:
            df = df.with_columns((pl.col(f).rank(method="ordinal", descending=False) * w).alias(f"_sc_{f}"))
        score_cols.append(f"_sc_{f}")
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
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v8")
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
    if n_total < 30:
        emit("error", msg="too few dates")
        return

    # Phase 1: precompute daily picks
    t_start = time.time()
    daily_picks_log = []
    for i, eval_date in enumerate(all_dates):
        eval_date_str = str(eval_date)[:10]
        factors_with_ic = rank_factors_by_ic(panel, eval_date_str, STAGE5B_FACTORS)
        top_factors = [f for f, _ in factors_with_ic[:TOP_FACTORS]]
        top_ics = [(f, ic) for f, ic in factors_with_ic[:TOP_FACTORS]]
        picks, scores = select_top_stocks_ic_weighted(panel, top_ics, eval_date_str, top_k=TOP_STOCKS)
        _DAILY_PICKS[eval_date_str] = scores
        daily_picks_log.append({
            "date": eval_date_str,
            "top_factors": top_factors,
            "factor_ics": dict(top_ics),
            "picks": picks,
        })
        if i % 20 == 0:
            emit("picks_computed", date=eval_date_str, day=i, n_total=n_total,
                 elapsed_sec=time.time() - t_start)
    emit("picks_phase_done", n_days=n_total, n_with_picks=len(_DAILY_PICKS),
         duration_sec=time.time() - t_start)

    # Phase 2: build data dict for ALL stocks that ever appear in picks
    all_picks_universe = sorted({sym for picks in _DAILY_PICKS.values() for sym in picks.keys()})
    emit("universe_size", n_unique_stocks=len(all_picks_universe))

    # Need to extend data dict slightly before start_dt for warmup
    warmup_start = start_dt - timedelta(days=30)
    data_dict = build_panel_data_dict(panel, all_picks_universe, warmup_start, end_dt)
    if not data_dict:
        emit("error", msg="no data in range")
        return
    emit("data_dict_built", n_symbols=len(data_dict))

    # Phase 3: single AKQuant run_backtest over the entire period
    emit("akquant_start")
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyReselectV8Strategy,
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

    # Extract final NAV
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

    # Try to extract NAV curve if available
    nav_curve = []
    if hasattr(result, "nav_curve"):
        for pt in result.nav_curve:
            nav_curve.append([str(pt.get("date", pt.get("trade_date", "")))[:10], float(pt.get("nav", 1.0))])

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
        "top_factors": STAGE5B_FACTORS[:TOP_FACTORS],
        "top_stocks": TOP_STOCKS,
        "daily_picks": daily_picks_log,
        "nav_curve": nav_curve,
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         duration_sec=time.time() - t_start)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()