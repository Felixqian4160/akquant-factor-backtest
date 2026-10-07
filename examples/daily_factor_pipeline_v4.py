"""v4: Daily factor reselect pipeline.

Every trading day:
  1. For each of 406 factors, compute 60-day-window × 20-day-fwd-return metric (close-to-close proxy)
  2. Pick top-10 factors by composite rank(return + 0.01 * sharpe)
  3. Cross-section percentile-rank on those 10 factors; average = composite
  4. Buy top-10 stocks at next day's raw open; sell at T+11 raw open (10-day hold)
  5. Record daily P&L; accumulate NAV

Returns: ledger + nav_curve + factor_history.

Usage:
  /usr/bin/python3.12 examples/daily_factor_pipeline_v4.py \
    --start-date 2010-01-01 --end-date 2025-12-31 \
    --top-factors 10 --top-stocks 10 --hold-days 10 --window-days 60 \
    --job-id v4_2010_2025
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

from aurumq_rl.data_loader import load_panel

# Constants
TOP_FACTORS = 10
TOP_STOCKS = 10
HOLD_DAYS = 10
WINDOW_DAYS = 60
EVIDENCE_DIR = ROOT / "evidence" / "daily_factor_reselect" / "pipeline_v4"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def emit_progress(payload: dict, job_dir: Path | None = None) -> None:
    """Write progress event to stdout + global progress.jsonl + per-job progress.jsonl."""
    payload["timestamp"] = datetime.now().isoformat()
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    global_progress = EVIDENCE_DIR / "progress.jsonl"
    with open(global_progress, "a") as f:
        f.write(line + "\n")
    if job_dir is not None:
        job_progress = job_dir / "progress.jsonl"
        with open(job_progress, "a") as f:
            f.write(line + "\n")


def compute_factor_metric(
    panel: pl.DataFrame, factor: str, eval_date: str,
    window_days: int = WINDOW_DAYS, fwd_days: int = 20, top_k: int = 20,
) -> dict:
    """Single-factor close-to-close backtest: long top-K stocks by factor, hold fwd_days."""
    eval_dt = datetime.strptime(eval_date, "%Y-%m-%d")
    start_str = (eval_dt - timedelta(days=int(window_days * 1.6))).strftime("%Y-%m-%d")
    fwd_end_dt = eval_dt + timedelta(days=int(fwd_days * 1.6))
    fwd_end_str = fwd_end_dt.strftime("%Y-%m-%d")

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

    # Eval at eval_date (or last date <= eval_date)
    eval_dates = dates.filter(dates <= eval_dt.strftime("%Y-%m-%d"))
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
    fwd_dates = dates.filter((dates > eval_rebal))
    fwd_dates = fwd_dates.head(fwd_days)
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

    # Per-stock return from first to last close in the forward window
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
    """Compute metrics for all factors on eval_date. Cache to disk if cache_dir provided."""
    if cache_dir is not None:
        cache_file = cache_dir / f"metrics_{eval_date}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["metrics"]
    metrics = {}
    t0 = time.time()
    for i, f in enumerate(all_factors):
        metrics[f] = compute_factor_metric(panel, f, eval_date)
        if (i + 1) % 50 == 0:
            emit_progress(
                {"stage": "metrics", "date": eval_date, "progress": i + 1, "total": len(all_factors),
                 "rate": (i + 1) / (time.time() - t0)},
                job_dir=job_dir,
            )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"date": eval_date, "metrics": metrics}))
    return metrics


def pick_top_factors(metrics: dict, top_n: int) -> list[str]:
    """Pick top-N factors ranked by return (proxy for momentum alpha)."""
    # Filter out zero-metric factors
    valid = {f: m["return"] for f, m in metrics.items() if m["n_stocks"] >= 5 and m["return"] > 0}
    if not valid:
        # If all zero, fall back to top by absolute return
        valid = {f: m["return"] for f, m in metrics.items() if m["n_stocks"] >= 5}
    if not valid:
        return []
    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_n]]


def select_top_stocks(
    panel: pl.DataFrame, factors: list[str], eval_date: str, top_k: int
) -> tuple[list[str], dict[str, float]]:
    """Cross-section percentile-rank on `factors`; average = composite; pick top-K stocks."""
    if not factors:
        return [], {}
    sub = panel.filter(pl.col("trade_date") == pl.lit(eval_date).str.strptime(pl.Date, "%Y-%m-%d"))
    sub = sub.drop_nulls(subset=factors)
    if sub.height == 0:
        return [], {}

    # Percentile rank for each factor (1.0 = best)
    for f in factors:
        sub = sub.with_columns(
            (pl.col(f).rank(method="ordinal", descending=True) / pl.col(f).count()).alias(f"_pr_{f}")
        )

    pr_cols = [f"_pr_{f}" for f in factors]
    sub = sub.with_columns(
        pl.sum_horizontal(pr_cols).truediv(len(pr_cols)).alias("_composite")
    )
    sorted_sub = sub.sort("_composite", descending=True, nulls_last=True).head(top_k)
    picks = sorted_sub["ts_code"].to_list()
    scores = {row["ts_code"]: float(row["_composite"]) for row in sorted_sub.to_dicts()}
    return picks, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--top-factors", type=int, default=TOP_FACTORS)
    parser.add_argument("--top-stocks", type=int, default=TOP_STOCKS)
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--job-id", type=str, default=None)
    args = parser.parse_args()

    job_id = args.job_id or f"v4_{int(time.time())}"
    job_dir = EVIDENCE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    emit_progress({"stage": "init", "job_id": job_id,
                   "start_date": args.start_date, "end_date": args.end_date,
                   "top_factors": args.top_factors, "top_stocks": args.top_stocks,
                   "hold_days": args.hold_days},
                  job_dir=job_dir)

    # Load factors list
    with open(ROOT / "evidence" / "daily_factor_reselect" / "all_factor_candidates.json") as f:
        all_factors = json.load(f)
    emit_progress({"stage": "factors_loaded", "n_factors": len(all_factors)}, job_dir=job_dir)

    # Load panel
    emit_progress({"stage": "loading_panel"}, job_dir=job_dir)
    panel = load_panel()
    emit_progress({"stage": "panel_loaded", "n_rows": panel.height, "n_cols": panel.width}, job_dir=job_dir)

    # Get trading dates
    dates = panel["trade_date"].unique().sort().to_list()
    dates_str = [str(d) for d in dates]
    in_range_idx = [i for i, d in enumerate(dates_str) if args.start_date <= d <= args.end_date]
    in_range_dates = [dates_str[i] for i in in_range_idx]
    emit_progress({"stage": "dates_loaded", "n_dates_in_range": len(in_range_dates)}, job_dir=job_dir)

    # Day-by-day loop
    cache_dir = job_dir / "metrics_cache"
    nav = 1.0
    nav_curve = []  # [(date, nav, picks, factors)]
    ledger = []  # [(date, picks, entry_prices, exit_prices, ret)]
    n_days = len(in_range_dates)
    t_start = time.time()

    for di, eval_date in enumerate(in_range_dates):
        # 1. compute metrics (cached per date)
        m = compute_all_metrics(panel, all_factors, eval_date, cache_dir, job_dir)
        # 2. pick top-N factors
        factors = pick_top_factors(m, args.top_factors)
        if len(factors) < 1:
            nav_curve.append((eval_date, nav, [], None))
            continue
        # 3. cross-section select top-K stocks
        picks, scores = select_top_stocks(panel, factors, eval_date, args.top_stocks)
        if len(picks) < 1:
            nav_curve.append((eval_date, nav, [], factors))
            continue

        # 4. compute return: T+1 open to T+11 open (or end if past)
        entry_idx = di + 1  # T+1
        exit_idx = min(di + 1 + args.hold_days, len(in_range_dates) - 1)  # T+1+hold
        if entry_idx >= len(in_range_dates):
            continue
        entry_date = in_range_dates[entry_idx]
        exit_date = in_range_dates[exit_idx]

        # Get entry open prices
        eod_entry = panel.filter(
            (pl.col("trade_date") == pl.lit(entry_date).str.strptime(pl.Date, "%Y-%m-%d"))
            & pl.col("ts_code").is_in(picks)
        ).select(["ts_code", "open"])
        exit_close = panel.filter(
            (pl.col("trade_date") == pl.lit(exit_date).str.strptime(pl.Date, "%Y-%m-%d"))
            & pl.col("ts_code").is_in(picks)
        ).select(["ts_code", "close"])

        merged = eod_entry.join(exit_close, on="ts_code", how="inner")
        if merged.height == 0:
            continue
        merged = merged.with_columns(
            (pl.col("close") / pl.col("open") - 1).alias("ret")
        )
        # Equal-weight
        cycle_ret = float(merged["ret"].mean()) if merged.height > 0 else 0.0
        # Apply costs (round-trip 0.5%)
        cycle_ret_net = cycle_ret - 0.005

        nav *= (1 + cycle_ret_net)
        nav_curve.append((eval_date, nav, picks, factors))
        ledger.append({
            "eval_date": eval_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "factors": factors,
            "picks": picks,
            "scores": scores,
            "cycle_return_gross": cycle_ret,
            "cycle_return_net": cycle_ret_net,
            "nav": nav,
        })

        if (di + 1) % 20 == 0 or di == n_days - 1:
            emit_progress(
                {"stage": "loop", "date": eval_date, "n_done": di + 1, "n_total": n_days,
                 "nav": nav, "cycle_ret": cycle_ret_net, "factors": factors[:3],
                 "elapsed_sec": time.time() - t_start},
                job_dir=job_dir,
            )

    # Output final
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
    emit_progress({"stage": "result_written", "result_file": str(result_file)},
                  job_dir=job_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())