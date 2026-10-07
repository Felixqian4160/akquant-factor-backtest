"""
Daily Factor Reselect Pipeline v3 — 使用 AKQuant 真实回测引擎.

设计合同 (用户明确):
  1. 每天跑全 406 因子 short-window backtest → metrics (return, sharpe)
  2. 每次选 N ∈ {5,6,7,8,9,10} → 因子组合
  3. 因子按收益排名组合 + 前一天必须正收益 (hard filter)
  4. 用 AKQuant 跑 rebal cycle → top-K picks (10-20 stocks) → 持仓 rebal_period
  5. 因子不变 → 股票池不变 → 继续持有
  6. 因子变了 → 调仓换股
  7. 第一次跑历史, 后面 incremental 只跑 (today-1)
  8. 输出 NAV 曲线 + ledger

时间预估 (实测):
  - 单因子 metrics 算 (polars): 14s / date × 30 dates = 7 min/batch
  - AKQuant rebal cycle: ~30s / cycle × 196 cycles = ~98 min
  - 完整 2010-2025: 132 batches × 7 min + 196 × 30s = ~16 hours

WebUI integration: emit_progress() 写到 stdout JSON 行, 同时写 progress.jsonl 给 webui poll
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

import akquant as aq
from akquant import Bar, Strategy

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_021633/"
    "wavehunter_mainwave_features_v2.parquet"
)

OUT_DIR = ROOT / "evidence" / "daily_factor_reselect" / "pipeline_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── Config ─────────
REBAL_DAYS = 20  # 默认 rebal period
TOP_K = 20  # 默认 top-K picks (10-20)
WINDOW_DAYS = 60  # factor metric short-window
N_OPTIONS = [5, 6, 7, 8, 9, 10]  # 候选 N
COST_BPS_PER_SIDE = 25
SLIPPAGE = 0.001  # 10 bps
INITIAL_CASH = 1_000_000.0


def emit_progress(payload: dict, job_dir: Path | None = None) -> None:
    """Write progress event to stdout (JSON line) + progress.jsonl for webui.

    Writes to:
      - stdout: for live log streaming
      - OUT_DIR/progress.jsonl: global file for /api/daily_factor/latest_progress
      - job_dir/progress.jsonl (if provided): per-job file for /api/daily_factor/progress/{job_id}
    """
    payload["timestamp"] = datetime.now().isoformat()
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    global_progress = OUT_DIR / "progress.jsonl"
    with open(global_progress, "a") as f:
        f.write(line + "\n")
    if job_dir is not None:
        job_progress = job_dir / "progress.jsonl"
        with open(job_progress, "a") as f:
            f.write(line + "\n")


def load_panel() -> pl.DataFrame:
    return pl.read_parquet(str(PANEL))


def get_trading_dates(panel: pl.DataFrame, start_date: str, end_date: str) -> list:
    return panel.filter(
        (pl.col("trade_date") >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d"))
    )["trade_date"].unique().sort().to_list()


def compute_factor_metric(
    panel: pl.DataFrame, factor: str, eval_date: str, window_days: int = WINDOW_DAYS, top_k: int = 20
) -> dict:
    """Compute single-factor short-window metric. NOT a backtest — just metric computation."""
    eval_dt = datetime.strptime(eval_date, "%Y-%m-%d")
    start_str = (eval_dt - timedelta(days=window_days * 2)).strftime("%Y-%m-%d")

    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(start_str).str.strptime(pl.Date, "%Y-%m-%d"))
        & (pl.col("trade_date") < pl.lit(eval_date).str.strptime(pl.Date, "%Y-%m-%d"))
    )
    if factor not in sub.columns:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}
    sub = sub.drop_nulls(subset=[factor])
    if sub.height < window_days * 50:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}

    dates = sub["trade_date"].unique().sort()
    if len(dates) < 30:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}

    last_idx = max(0, len(dates) - 21)
    last_rebal = dates[last_idx]
    rebal = (
        sub.filter(pl.col("trade_date") == last_rebal)
        .sort(factor, descending=True, nulls_last=True)
        .head(top_k)
    )
    if rebal.height == 0:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}

    ts_codes = rebal["ts_code"].to_list()
    fwd_dates = dates[(last_idx + 1) : (last_idx + 21)]
    fwd = sub.filter(pl.col("ts_code").is_in(ts_codes) & pl.col("trade_date").is_in(fwd_dates))
    if fwd.height == 0:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}

    pivot = fwd.pivot(index="ts_code", on="trade_date", values="close").sort("ts_code")
    if pivot.height == 0 or pivot.width < 3:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}
    close_cols = [c for c in pivot.columns if c != "ts_code"]
    arr = pivot.select(close_cols).to_numpy()
    if arr.shape[1] < 2:
        return {"return": 0.0, "sharpe": 0.0, "ann": 0.0}
    per_ret = arr[:, -1] / arr[:, 0] - 1
    avg = float(np.nanmean(per_ret))
    std = float(np.nanstd(per_ret)) if len(per_ret) > 1 else 0.01
    n_d = arr.shape[1]
    sharpe = avg / max(std, 0.01) * np.sqrt(252 / max(n_d, 1))
    ann = (1 + avg) ** (252 / max(n_d, 1)) - 1
    return {"return": avg, "sharpe": float(sharpe), "ann": float(ann)}


def compute_all_metrics(
    panel: pl.DataFrame, all_factors: list, date_str: str, cache_dir: Path, job_dir: Path | None = None
) -> dict:
    """Compute metrics for all factors on one date. Cache to disk."""
    cache_file = cache_dir / f"metrics_{date_str}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["metrics"]
    metrics = {}
    t0 = time.time()
    for i, f in enumerate(all_factors):
        metrics[f] = compute_factor_metric(panel, f, date_str)
        if (i + 1) % 100 == 0:
            emit_progress(
                {
                    "stage": "metrics",
                    "date": date_str,
                    "progress": i + 1,
                    "total": len(all_factors),
                    "rate": (i + 1) / (time.time() - t0),
                },
                job_dir=job_dir,
            )
    cache_file.write_text(json.dumps({"date": date_str, "metrics": metrics}))
    return metrics


def pick_factors(prev_metrics: dict, all_factors: list, N: int) -> list:
    """Hard filter: prev day return > 0. Rank by return+sharpe. Top-N."""
    candidates = [
        (f, prev_metrics[f]) for f in all_factors if prev_metrics[f]["return"] > 0
    ]
    if len(candidates) < N:
        return []
    candidates.sort(key=lambda x: x[1]["return"] + x[1]["sharpe"] * 0.01, reverse=True)
    return [c[0] for c in candidates[:N]]


class DailyFactorReselectStrategy(Strategy):
    """AKQuant Strategy that picks top-K stocks based on pre-selected N factors.

    The factor selection happens BEFORE run_backtest (in run_rebal_cycle),
    and the chosen factors + top_k are injected via class-level attributes
    that AKQuant will read when instantiating.
    """

    warmup = 5
    factors: list = []  # class-level, set via monkey-patch before each cycle
    top_k: int = TOP_K
    rebal_days: int = REBAL_DAYS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._last_rebal_date = None

    @classmethod
    def configure(cls, factors: list, top_k: int, rebal_days: int) -> None:
        """Inject factor selection — modifies class attrs."""
        cls.factors = list(factors)
        cls.top_k = top_k
        cls.rebal_days = rebal_days

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
        vals = []
        ok = True
        for f in self.factors:
            v = bar.extra.get(f)
            if v is None:
                ok = False
                break
            vals.append(float(v))
        if ok:
            self._factor_store[d][bar.symbol] = vals

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date
        if d not in self._factor_store:
            return
        if self._last_rebal_date is not None:
            gap = (d - self._last_rebal_date).days
            if gap < self.rebal_days:
                return
        self._last_rebal_date = d

        raw = self._factor_store[d]
        if len(raw) < self.top_k:
            return

        syms = list(raw.keys())
        mat = np.array([raw[s] for s in syms])
        ranks = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            order = np.argsort(col, kind="mergesort")
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(col))
            ranks[:, j] = r / max(len(col) - 1, 1)
        composite = ranks.mean(axis=1)
        scores = {syms[i]: float(composite[i]) for i in range(len(syms))}
        try:
            self.rebalance_to_topn(
                scores=scores,
                top_n=self.top_k,
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


def build_data_dict(
    panel: pl.DataFrame, factors: list, start, end, top_k: int
) -> tuple:
    """Build {symbol: DataFrame} dict for AKQuant. Include factor columns in 'extra'.

    `start`/`end` may be either datetime.date or datetime.datetime — normalise to date
    and pass as native Python date (polars auto-converts to Date literal).
    """
    if hasattr(start, "date") and callable(start.date):
        start = start.date()
    if hasattr(end, "date") and callable(end.date):
        end = end.date()
    s = pl.lit(start)
    e = pl.lit(end)
    df = (
        panel.select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"] + factors)
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
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
        data_dict[code] = pdf
    return data_dict, df.shape[0], len(stocks)


def run_rebal_cycle_akquant(
    panel: pl.DataFrame,
    factors: list,
    cycle_start: date,
    cycle_end: date,
    top_k: int,
    rebal_days: int,
    cycle_idx: int,
) -> dict:
    """Run one rebal cycle using AKQuant engine."""
    data_dict, n_rows, n_stocks = build_data_dict(
        panel, factors, cycle_start, cycle_end, top_k
    )
    if n_stocks < top_k:
        return {
            "error": "too_few_stocks",
            "cycle_idx": cycle_idx,
            "n_stocks": n_stocks,
            "factors": factors,
            "top_k": top_k,
        }

    t0 = time.time()
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=DailyFactorReselectStrategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
        )
        elapsed = time.time() - t0
        m = result.metrics_df
        total_ret = float(m.loc["total_return_pct", "value"])
        closed_ret = float(
            (float(m.loc["total_pnl", "value"]) - float(m.loc["unrealized_pnl", "value"]))
            / float(m.loc["initial_market_value", "value"])
            * 100
        )
        sharpe = float(m.loc["sharpe_ratio", "value"])
        mdd = float(m.loc["max_drawdown_pct", "value"])
        n_trades = int(m.loc["closed_trade_count", "value"])
        end_value = float(m.loc["end_market_value", "value"])

        return {
            "cycle_idx": cycle_idx,
            "cycle_start": cycle_start.isoformat(),
            "cycle_end": cycle_end.isoformat(),
            "factors": factors,
            "top_k": top_k,
            "n_stocks": n_stocks,
            "n_rows": n_rows,
            "elapsed_sec": elapsed,
            "total_return_pct": total_ret,
            "closed_only_return_pct": closed_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": mdd,
            "n_trades": n_trades,
            "end_value": end_value,
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "cycle_idx": cycle_idx,
            "cycle_start": cycle_start.isoformat(),
            "cycle_end": cycle_end.isoformat(),
            "factors": factors,
            "top_k": top_k,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2024-03-31")
    parser.add_argument("--rebal-days", type=int, default=REBAL_DAYS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--n-options", type=str, default=",".join(map(str, N_OPTIONS)))
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Reuse an existing metrics_cache/ dir (skip recomputing metrics).",
    )
    args = parser.parse_args()

    n_options = [int(x) for x in args.n_options.split(",")]
    job_id = args.job_id or f"daily_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    log_file = job_dir / "pipeline.log"
    if args.cache_dir:
        # Reuse existing metrics cache from a previous job (skip metrics recompute).
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = (ROOT / cache_dir).resolve()
        log_file.write_text(
            f"[{datetime.now().isoformat()}] reusing cache_dir={cache_dir}\n"
        )
    else:
        cache_dir = job_dir / "metrics_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        f"[{datetime.now().isoformat()}] start: {args.start_date} ~ {args.end_date}, "
        f"rebal={args.rebal_days}, top_k={args.top_k}, n_options={n_options}\n"
    )

    emit_progress(
        {
            "stage": "init",
            "job_id": job_id,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "rebal_days": args.rebal_days,
            "top_k": args.top_k,
            "n_options": n_options,
        },
        job_dir=job_dir,
    )

    with open(ROOT / "evidence" / "daily_factor_reselect" / "all_factor_candidates.json") as f:
        all_factors = json.load(f)

    emit_progress({"stage": "loading_panel"}, job_dir=job_dir)
    panel = load_panel()
    emit_progress({"stage": "panel_loaded", "n_rows": panel.height, "n_cols": panel.width}, job_dir=job_dir)

    all_dates = get_trading_dates(panel, args.start_date, args.end_date)
    if len(all_dates) < 30:
        log_file.write_text(f"ERROR: panel too short ({len(all_dates)} days)\n")
        return 1

    # Extended range for prev day lookups
    extra_days = args.window_days * 4
    extended_start = (datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=extra_days)).strftime("%Y-%m-%d")
    extended_dates = get_trading_dates(panel, extended_start, args.end_date)

    # Step 1: compute metrics cache for extended range
    metrics_by_date: dict = {}
    for di, d in enumerate(extended_dates):
        date_str = d.strftime("%Y-%m-%d")
        m = compute_all_metrics(panel, all_factors, date_str, cache_dir, job_dir=job_dir)
        metrics_by_date[date_str] = m
        if (di + 1) % 5 == 0 or di == len(extended_dates) - 1:
            emit_progress(
                {
                    "stage": "metrics_done",
                    "date": date_str,
                    "n_dates_done": di + 1,
                    "n_dates_total": len(extended_dates),
                    "pct": (di + 1) / len(extended_dates) * 100,
                },
                job_dir=job_dir,
            )

    # Step 2: rebal cycles (every args.rebal_days)
    in_range_dates = get_trading_dates(panel, args.start_date, args.end_date)
    rebal_indices = list(range(0, len(in_range_dates), args.rebal_days))
    if rebal_indices[-1] != len(in_range_dates) - 1:
        rebal_indices.append(len(in_range_dates) - 1)
    rebal_dates = [in_range_dates[i] for i in rebal_indices]

    emit_progress({"stage": "rebal_starting", "n_cycles": len(rebal_dates)}, job_dir=job_dir)

    # Step 3: per-rebal pick factors + run AKQuant cycle
    nav = 1.0
    nav_curve = []
    ledger = []
    prev_factors = None

    for ri, rebal_date in enumerate(rebal_dates):
        # Find prev trading day (in metrics cache)
        prev_str = None
        for d in extended_dates:
            if d < rebal_date:
                prev_str = d.strftime("%Y-%m-%d")
            else:
                break
        if prev_str is None or prev_str not in metrics_by_date:
            emit_progress({"stage": "rebal_skip", "reason": "no_prev_metrics", "rebal_date": rebal_date.strftime("%Y-%m-%d")}, job_dir=job_dir)
            continue
        prev_metrics = metrics_by_date[prev_str]

        # For each N option, pick factors
        all_picks_per_n = {}
        for N in n_options:
            factors = pick_factors(prev_metrics, all_factors, N)
            if not factors:
                continue
            all_picks_per_n[N] = factors

        if not all_picks_per_n:
            emit_progress({"stage": "rebal_skip", "reason": "no_valid_N", "rebal_date": rebal_date.strftime("%Y-%m-%d")}, job_dir=job_dir)
            continue

        # Choose N=default first, but if multiple have valid picks, prefer N with most candidates
        chosen_N = max(all_picks_per_n.keys(), key=lambda n: (len(all_picks_per_n[n]), -n))
        factors = all_picks_per_n[chosen_N]

        # Detect rebalance: factor set changed?
        rebalance = (factors != prev_factors) if prev_factors is not None else True

        # Run AKQuant cycle: cycle_start = rebal_date, cycle_end = min(rebal_date + rebal_days, end_date)
        cycle_end_idx = min(ri + 1, len(rebal_indices) - 1)
        cycle_end = in_range_dates[rebal_indices[cycle_end_idx]]

        cycle_result = run_rebal_cycle_akquant(
            panel=panel,
            factors=factors,
            cycle_start=rebal_date,
            cycle_end=cycle_end,
            top_k=args.top_k,
            rebal_days=args.rebal_days,
            cycle_idx=ri,
        )

        cycle_result["nav"] = nav
        if "closed_only_return_pct" in cycle_result:
            cycle_return = cycle_result["closed_only_return_pct"] / 100
            nav *= 1 + cycle_return
            cycle_result["nav_after"] = nav
            nav_curve.append((rebal_date.strftime("%Y-%m-%d"), nav, len(factors), chosen_N, factors[:3]))
        cycle_result["rebalance"] = rebalance
        cycle_result["prev_factors"] = prev_factors
        cycle_result["new_factors"] = [f for f in factors if f not in (prev_factors or [])]
        cycle_result["dropped_factors"] = [f for f in (prev_factors or []) if f not in factors]
        ledger.append(cycle_result)

        prev_factors = factors

        emit_progress(
            {
                "stage": "rebal_done",
                "cycle_idx": ri,
                "rebal_date": rebal_date.strftime("%Y-%m-%d"),
                "N": chosen_N,
                "n_factors": len(factors),
                "rebalance": rebalance,
                "cycle_return_pct": cycle_result.get("closed_only_return_pct"),
                "nav": nav,
                "n_cycles_done": ri + 1,
                "n_cycles_total": len(rebal_dates),
            },
            job_dir=job_dir,
        )

    # Output final
    emit_progress({"stage": "done", "final_nav": nav, "n_cycles": len(ledger)}, job_dir=job_dir)

    result = {
        "job_id": job_id,
        "config": vars(args),
        "n_rebal_cycles": len(ledger),
        "final_nav": nav,
        "final_return_pct": (nav - 1) * 100,
        "nav_curve": nav_curve,
        "ledger": ledger,
    }
    result_file = job_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    emit_progress({"stage": "result_written", "result_file": str(result_file)}, job_dir=job_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())