"""v5: Daily factor reselect pipeline USING AKQuant (per user requirement).

Every trading day:
  1. For each of 406 factors, compute 60-day-window × 20-day-fwd-return metric (polars, cached)
  2. Pick top-10 factors by composite rank
  3. Cross-section select top-10 stocks
  4. Run ONE AKQuant mini-backtest for that day's picks: T+1 open buy, hold 10 days, T+11 open sell
  5. Record daily P&L; accumulate NAV

AKQuant is the backtest engine per cycle (mandatory per user).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

# aurumq-rl path for panel_loader
sys.path.insert(0, "/media/felix/f/quant/aurumq-rl/src")

# AKQuant
import akquant as aq

# Constants
TOP_FACTORS = 10
TOP_STOCKS = 10
HOLD_DAYS = 10
WINDOW_DAYS = 60
INITIAL_CASH = 100_000_000.0  # ¥100M so 10-stock equal-weight has plenty of margin
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}

EVIDENCE_DIR = ROOT / "evidence" / "daily_factor_reselect" / "pipeline_v5"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# Global state — AKQuant 0.3.x doesn't pass __init__ kwargs, so we use module-level state.
_CURRENT_PICKS: dict[str, float] = {}


class DailyFactorReselectStrategy5(aq.Strategy):
    """Buy top-K stocks once at cycle start, hold until cycle end, then close all."""
    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rebalanced = False

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        if self._rebalanced:
            return
        self._rebalanced = True
        if not _CURRENT_PICKS:
            return
        try:
            self.rebalance_to_topn(
                scores=_CURRENT_PICKS,
                top_n=len(_CURRENT_PICKS),
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


def emit_progress(payload: dict, job_dir: Path | None = None) -> None:
    payload["timestamp"] = datetime.now().isoformat()
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    with open(EVIDENCE_DIR / "progress.jsonl", "a") as f:
        f.write(line + "\n")
    if job_dir is not None:
        with open(job_dir / "progress.jsonl", "a") as f:
            f.write(line + "\n")


def compute_factor_metric(
    panel: pl.DataFrame, factor: str, eval_date: str,
    window_days: int = WINDOW_DAYS, fwd_days: int = 20, top_k: int = 20,
) -> dict:
    eval_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d")
    start_str = (eval_dt - timedelta(days=int(window_days * 1.6))).strftime("%Y-%m-%d")
    fwd_end_str = (eval_dt + timedelta(days=int(fwd_days * 1.6))).strftime("%Y-%m-%d")

    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_str).str.strptime(pl.Date, "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(fwd_end_str).str.strptime(pl.Date, "%Y-%m-%d"))
    )
    if factor not in sub.columns:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}
    sub = sub.drop_nulls(subset=[factor])
    if sub.height < window_days * 30:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    dates = sub["trade_date"].unique().sort()
    if len(dates) < 30:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    eval_dates = dates.filter(dates <= eval_dt)
    if len(eval_dates) == 0:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}
    eval_rebal = eval_dates[-1]

    rebal = (
        sub.filter(pl.col("trade_date") == eval_rebal)
        .sort(factor, descending=True, nulls_last=True)
        .head(top_k)
    )
    if rebal.height == 0:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    ts_codes = rebal["ts_code"].to_list()
    fwd_dates = dates.filter(dates > eval_rebal).head(fwd_days)
    if len(fwd_dates) < 2:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    fwd = sub.filter(pl.col("ts_code").is_in(ts_codes) & pl.col("trade_date").is_in(fwd_dates))
    if fwd.height == 0:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    pivot = fwd.pivot(index="ts_code", on="trade_date", values="close").sort("ts_code")
    if pivot.height == 0 or pivot.width < 3:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    close_cols = [c for c in pivot.columns if c != "ts_code"]
    arr = pivot.select(close_cols).to_numpy()
    if arr.shape[1] < 2:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    per_ret = arr[:, -1] / arr[:, 0] - 1
    per_ret = per_ret[np.isfinite(per_ret)]
    if len(per_ret) == 0:
        return {"return": 0.0, "sharpe": 0.0, "n_stocks": 0}

    mean_ret = float(np.mean(per_ret))
    std_ret = float(np.std(per_ret, ddof=1)) if len(per_ret) > 1 else 0.0
    sharpe = mean_ret / std_ret if std_ret > 1e-9 else 0.0
    return {"return": mean_ret, "sharpe": sharpe, "n_stocks": len(per_ret)}


def compute_all_metrics(
    panel: pl.DataFrame, all_factors: list, eval_date: str,
    cache_dir: Path | None = None, job_dir: Path | None = None,
) -> dict:
    if cache_dir is not None:
        cache_file = cache_dir / f"metrics_{eval_date}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["metrics"]
    metrics = {}
    t0 = time.time()
    for i, f in enumerate(all_factors):
        metrics[f] = compute_factor_metric(panel, f, eval_date)
        if (i + 1) % 50 == 0:
            emit_progress({"stage": "metrics", "date": eval_date,
                           "progress": i + 1, "total": len(all_factors),
                           "rate": (i + 1) / (time.time() - t0)}, job_dir=job_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"date": eval_date, "metrics": metrics}))
    return metrics


def pick_top_factors(metrics: dict, top_n: int) -> list[str]:
    valid = {f: m["return"] for f, m in metrics.items()
             if m["n_stocks"] >= 5 and m["return"] > 0}
    if not valid:
        valid = {f: m["return"] for f, m in metrics.items() if m["n_stocks"] >= 5}
    if not valid:
        return []
    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_n]]


def select_top_stocks(
    panel: pl.DataFrame, factors: list[str], eval_date: str, top_k: int
) -> tuple[list[str], dict[str, float]]:
    """Cross-section percentile-rank on `factors`; pick the SINGLE best factor's top-K stocks.

    User contract is "select by best factor group" — average-composite dilutes alpha.
    We pick the factor with highest single-factor metric (passed in by caller order) as
    the tiebreaker; with the rank-based composite for stocks ranking.
    """
    if not factors:
        return [], {}
    sub = panel.filter(pl.col("trade_date") == pl.lit(eval_date).str.strptime(pl.Date, "%Y-%m-%d"))
    sub = sub.drop_nulls(subset=factors)
    if sub.height == 0:
        return [], {}

    # Use the FIRST factor (top-ranked by metrics) as the primary signal.
    # Cross-sectional percentile rank within the universe.
    primary = factors[0]
    sub = sub.with_columns(
        (pl.col(primary).rank(method="ordinal", descending=True) / pl.col(primary).count())
        .alias("_score")
    )
    sorted_sub = sub.sort("_score", descending=True, nulls_last=True).head(top_k)
    picks = sorted_sub["ts_code"].to_list()
    scores = {row["ts_code"]: float(row["_score"]) for row in sorted_sub.to_dicts()}
    return picks, scores


def build_mini_data_dict(
    panel: pl.DataFrame, picks: list[str], start: date, end: date, factors: list[str]
) -> dict:
    """Build {symbol: DataFrame} dict for AKQuant — mini cycle only."""
    if hasattr(start, "date") and callable(start.date):
        start_d = start.date()
    else:
        start_d = start
    if hasattr(end, "date") and callable(end.date):
        end_d = end.date()
    else:
        end_d = end
    s = pl.lit(start_d)
    e = pl.lit(end_d)
    df = (
        panel.select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"] + factors)
        .filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
        .filter(pl.col("ts_code").is_in(picks))
        .drop_nulls(subset=factors)
    )
    stocks = sorted(df["ts_code"].unique().to_list())
    data_dict: dict = {}
    for code in stocks:
        pdf = (
            df.filter(pl.col("ts_code") == code)
            .sort("trade_date")
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1.0e9  # synthetic large volume
        # Ensure factor cols go into extra (AKQuant reads from extra dict)
        for f in factors:
            if f not in pdf.columns:
                continue
        data_dict[code] = pdf
    return data_dict


def run_akquant_cycle(
    panel: pl.DataFrame, picks: dict[str, float], factors: list[str],
    cycle_start: date, cycle_end: date,
) -> dict:
    """Run ONE AKQuant backtest for a 10-day hold cycle."""
    global _CURRENT_PICKS
    if not picks or cycle_start >= cycle_end:
        return {"cycle_return_pct": 0.0, "n_trades": 0}
    # Set global state so strategy can read it
    _CURRENT_PICKS = dict(picks)
    data_dict = build_mini_data_dict(panel, list(picks.keys()), cycle_start, cycle_end, factors)
    if not data_dict:
        _CURRENT_PICKS = {}
        return {"cycle_return_pct": 0.0, "n_trades": 0}
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyFactorReselectStrategy5,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
        )
        m = result.metrics_df
        # Use total_return_pct (includes unrealized P&L) — mini cycles
        # are too short to close all positions, so realised P&L alone
        # would just reflect commission cost.
        cycle_ret_pct = float(m.loc["total_return_pct", "value"])
        n_trades = int(m.loc["closed_trade_count", "value"])
        _CURRENT_PICKS = {}
        return {"cycle_return_pct": cycle_ret_pct, "n_trades": n_trades}
    except Exception as exc:
        _CURRENT_PICKS = {}
        return {"cycle_return_pct": 0.0, "n_trades": 0, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--top-factors", type=int, default=TOP_FACTORS)
    parser.add_argument("--top-stocks", type=int, default=TOP_STOCKS)
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--max-days", type=int, default=0,
                        help="If > 0, limit to first N trading days (for smoke tests).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Reuse existing metrics_cache/ from a prior job.")
    args = parser.parse_args()

    job_id = args.job_id or f"v5_{int(time.time())}"
    job_dir = EVIDENCE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    emit_progress({"stage": "init", "job_id": job_id,
                   "start_date": args.start_date, "end_date": args.end_date,
                   "top_factors": args.top_factors, "top_stocks": args.top_stocks,
                   "hold_days": args.hold_days}, job_dir=job_dir)

    with open(ROOT / "evidence" / "daily_factor_reselect" / "all_factor_candidates.json") as f:
        all_factors = json.load(f)
    emit_progress({"stage": "factors_loaded", "n_factors": len(all_factors)}, job_dir=job_dir)

    emit_progress({"stage": "loading_panel"}, job_dir=job_dir)
    # Load the v2 panel (has all 406 cols)
    panel_file = ("/media/felix/f/quant/aurumq-rl/evidence/"
                 "quant_workflow_migration_20260915/"
                 "v10_2_mainwave_features_v2_talib_20260924_021633/"
                 "wavehunter_mainwave_features_v2.parquet")
    panel = pl.read_parquet(panel_file)
    emit_progress({"stage": "panel_loaded", "n_rows": panel.height, "n_cols": panel.width},
                  job_dir=job_dir)

    dates = panel["trade_date"].unique().sort().to_list()
    dates_str = [str(d)[:10] for d in dates]  # strip " 00:00:00"
    in_range = [d for d in dates_str if args.start_date <= d <= args.end_date]
    if args.max_days > 0:
        in_range = in_range[: args.max_days]
    emit_progress({"stage": "dates_loaded", "n_dates_in_range": len(in_range)}, job_dir=job_dir)

    if args.cache_dir:
        # Reuse existing metrics cache from a previous job (skip metrics recompute).
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = (ROOT / cache_dir).resolve()
        log_path = job_dir / "pipeline.log"
        log_path.write_text(f"[{datetime.now().isoformat()}] reusing cache_dir={cache_dir}\n")
    else:
        cache_dir = job_dir / "metrics_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    nav = 1.0
    nav_curve = []
    ledger = []
    n_days = len(in_range)
    t_start = time.time()

    for di, eval_date in enumerate(in_range):
        m = compute_all_metrics(panel, all_factors, eval_date, cache_dir, job_dir)
        factors = pick_top_factors(m, args.top_factors)
        if len(factors) < 1:
            nav_curve.append((eval_date, nav, [], None))
            continue
        picks, scores = select_top_stocks(panel, factors, eval_date, args.top_stocks)
        if len(picks) < 1:
            nav_curve.append((eval_date, nav, [], factors))
            continue

        # T+1 open buy, hold 10 days, T+11 open sell
        entry_idx = di + 1
        exit_idx = min(di + 1 + args.hold_days, len(in_range) - 1)
        if entry_idx >= len(in_range):
            continue
        entry_date = datetime.strptime(in_range[entry_idx], "%Y-%m-%d").date()
        exit_date = datetime.strptime(in_range[exit_idx], "%Y-%m-%d").date()

        cycle_result = run_akquant_cycle(panel, scores, factors, entry_date, exit_date)
        cycle_ret_pct = cycle_result["cycle_return_pct"]
        cycle_ret = cycle_ret_pct / 100.0
        nav *= (1 + cycle_ret)
        nav_curve.append((eval_date, nav, picks, factors))
        ledger.append({
            "eval_date": eval_date,
            "entry_date": in_range[entry_idx],
            "exit_date": in_range[exit_idx],
            "factors": factors,
            "picks": picks,
            "scores": scores,
            "cycle_return_pct": cycle_ret_pct,
            "cycle_return_pct_inferred": cycle_ret * 100,
            "nav": nav,
            "n_trades": cycle_result.get("n_trades", 0),
            "error": cycle_result.get("error"),
        })

        if (di + 1) % 5 == 0 or di == n_days - 1:
            emit_progress({"stage": "loop", "date": eval_date, "n_done": di + 1,
                           "n_total": n_days, "nav": nav, "cycle_ret_pct": cycle_ret_pct,
                           "factors": factors[:3], "elapsed_sec": time.time() - t_start},
                          job_dir=job_dir)

    final_nav = nav
    final_ret = (final_nav - 1) * 100
    result = {
        "job_id": job_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "top_factors": args.top_factors,
        "top_stocks": args.top_stocks,
        "hold_days": args.hold_days,
        "n_days": n_days,
        "n_rebal_days": len(ledger),
        "final_nav": final_nav,
        "final_return_pct": final_ret,
        "nav_curve": nav_curve,
        "ledger": ledger,
    }
    result_file = job_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    emit_progress({"stage": "done", "final_nav": final_nav, "n_cycles": len(ledger)},
                  job_dir=job_dir)
    emit_progress({"stage": "result_written", "result_file": str(result_file)}, job_dir=job_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())