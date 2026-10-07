"""Daily Factor Reselect v10 — 加 bull/bear regime router:

User's contract (regime-aware):
- Bull regime (idx_mom_60 > +5%): use Stage 5b 6 trend factors (talib_NATR, ...)
- Bear regime (idx_mom_60 < -5%): use reversal factors (low drawdown + mean revert)
- Sideways regime: hold cash / use mean-revert signals

This addresses v9's failure: Stage 5b trend factors picked large-cap bank stocks
which underperformed in 2020 bull market. By regime-routing, we hope to:
- In bull: ride trend (alpha-positive factors)
- In bear: avoid buying bank stocks, look for rebound candidates
- In sideways: stay flat

Same backtest engine (AKQuant) as v9.
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

# Bull regime factors (Stage 5b proven trend-following)
BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE", "gtja_gtja_159",
    "gtja_gtja_149", "gtja_gtja_144", "mw_vol_20d",
]

# Bear regime factors — favour mean-revert / low-vol / momentum-reversal
# These were v10.2 verified reversal-style signals
BEAR_FACTORS = [
    "mw_ret_5d", "mw_ret_20d", "mw_rsi_14",
    "mw_vol_20d", "alpha_alpha018", "alpha_alpha028",
]

# Sideways regime — mild trend or hold-flat
SIDEWAYS_FACTORS = [
    "talib_BBANDS_upper", "talib_BBANDS_lower", "mw_rsi_14",
    "mw_vol_20d", "alpha_alpha018", "alpha_alpha005",
]

# Fallback list (used if regime-specific factor not in panel)
FALLBACK_FACTORS = BULL_FACTORS

REGIME_THRESHOLDS = {
    "bull_thresh": 0.05,      # idx_mom_60 > +5%
    "bear_thresh": -0.05,     # idx_mom_60 < -5%
}

TOP_STOCKS = 10
INITIAL_CASH = 100_000_000.0
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}
MIN_HOLD_DAYS = 5

# Pre-computed daily picks + regime: {date_str: {"picks": dict, "regime": str}}
_DAILY_PICKS: dict[str, dict] = {}


class DailyReselectV10Strategy(aq.Strategy):
    """Regime-aware daily picks. Rebalance only when picks change AND MIN_HOLD_DAYS."""

    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._last_rebalance_date = None

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]
        daily = _DAILY_PICKS.get(date_str, {})
        picks = daily.get("picks", {})
        new_set = set(picks.keys())
        if new_set == self._current_picks:
            return
        if not new_set:
            return  # regime router said "hold cash"

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


def select_top_stocks_v10(panel, factors, eval_date, top_k=TOP_STOCKS):
    """Cross-section percentile rank → composite sum → top K stocks."""
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


def classify_regime(panel, eval_date):
    """Classify regime based on HS300 idx_mom_60 (60-day momentum of index).

    Returns 'bull' | 'bear' | 'sideways'.
    """
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    # Panel trade_date is Datetime(ms); use idx_close from INDEX_PANEL
    # Simpler: use last close of "ts_code" == "000300.SH" from factor panel
    # But factor panel doesn't have idx_close. Use a proxy: cross-sectional mean of
    # mw_ret_60d over all stocks on eval_date
    sub = panel.filter(
        pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    )
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)
    if "mw_ret_60d" in sub.columns:
        # Cross-sectional mean of mw_ret_60d as regime proxy
        proxy_mean = float(sub["mw_ret_60d"].drop_nulls().mean())
    else:
        proxy_mean = 0.0

    if proxy_mean > REGIME_THRESHOLDS["bull_thresh"]:
        return "bull", proxy_mean
    elif proxy_mean < REGIME_THRESHOLDS["bear_thresh"]:
        return "bear", proxy_mean
    else:
        return "sideways", proxy_mean


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
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v10")
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

    # Phase 1: precompute daily picks + regime
    t_start = time.time()
    daily_picks_log = []
    regime_counts = {"bull": 0, "bear": 0, "sideways": 0}

    for i, eval_date in enumerate(all_dates):
        eval_date_str = str(eval_date)[:10]
        regime, proxy_mean = classify_regime(panel, eval_date_str)
        regime_counts[regime] += 1
        if regime == "bull":
            factors = BULL_FACTORS
        elif regime == "bear":
            factors = BEAR_FACTORS
        else:
            factors = SIDEWAYS_FACTORS

        picks, scores = select_top_stocks_v10(panel, factors, eval_date_str, top_k=TOP_STOCKS)
        _DAILY_PICKS[eval_date_str] = {"picks": scores, "regime": regime, "proxy_mean": proxy_mean}
        daily_picks_log.append({
            "date": eval_date_str, "picks": picks, "scores": scores,
            "regime": regime, "proxy_mean": proxy_mean,
        })

        if i % 50 == 0:
            emit("picks_computed", date=eval_date_str, day=i, n_total=n_total,
                 regime=regime, proxy_mean=proxy_mean,
                 elapsed_sec=time.time() - t_start)

    emit("picks_phase_done", n_days=n_total, regime_counts=regime_counts,
         duration_sec=time.time() - t_start)

    # Phase 2: build data dict for ALL stocks that ever appear in picks
    all_picks_universe = sorted(
        {sym for d in _DAILY_PICKS.values() for sym in d["picks"].keys()}
    )
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
            strategy=DailyReselectV10Strategy,
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

    # Extract equity curve
    nav_curve = []
    try:
        eq = result.equity_curve
        if hasattr(eq, "items"):
            for ts, val in eq.items():
                nav_curve.append([str(ts)[:10], val])
        elif isinstance(eq, list):
            for pt in eq:
                nav_curve.append([str(pt[0])[:10], float(pt[1])])
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
        "top_factors": {"bull": BULL_FACTORS, "bear": BEAR_FACTORS, "sideways": SIDEWAYS_FACTORS},
        "top_stocks": TOP_STOCKS,
        "min_hold_days": MIN_HOLD_DAYS,
        "regime_thresholds": REGIME_THRESHOLDS,
        "regime_counts": regime_counts,
        "daily_picks": daily_picks_log,
        "nav_curve": nav_curve,
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_value=final_value, final_return_pct=final_return_pct,
         n_days=n_total, regime_counts=regime_counts,
         duration_sec=time.time() - t_start)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()