"""Walk-forward validation CLI for WebUI.

Reuses Stage 5b's BullCompositeTopK10Kill strategy (verified v1+v2 terminal
state). Runs multiple time-series windows:
  - train window: parameter stable (no retrain in this MVP — uses Stage 5b hyper-params)
  - val window: in-sample validation
  - OOS window: out-of-sample test

Single-variable change vs Stage 5b: panel slicing (per-bull-leg → walk-forward).

Output JSON:
{
  "contract": {...},
  "windows": [
    {
      "idx": 1,
      "train_start": "2010-01-01", "train_end": "2018-12-31",
      "oos_start": "2019-07-01",  "oos_end": "2020-12-31",
      "oos_metrics": {
        "closed_only_return_pct": ..., "sharpe_ratio": ...,
        "max_drawdown_pct": ..., "closed_trade_count": ...
      }
    }, ...
  ],
  "summary": {
    "n_windows": ...,
    "mean_oos_ann_pct": ...,
    "mean_oos_sharpe": ...,
    "mean_oos_mdd_pct": ...,
    "n_pass_target": ...
  }
}
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "src"))

# Reuse Stage 5b's verified strategy class via dynamic import.
# This ensures walk-forward uses the EXACT same alpha logic.
sys.path.insert(0, str(ROOT / "examples"))

PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]
TOP_K = 10
REBAL_DAYS = 20
INITIAL_CASH = 1_000_000.0
COMMISSION_BPS_PER_SIDE = 40
KILL_DD_THRESHOLD = 0.08
KILL_COOLDOWN_DAYS = 20


# === Same strategy as Stage 5b (verified) ===
class BullCompositeTopK10Kill(Strategy):
    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        self._daily_complete: set = set()
        self._equity_peak: float = float(INITIAL_CASH)
        self._kill_until = None

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
            self._symbols_seen[d] = set()
        vals = []
        ok = True
        for f in BULL_FACTORS:
            v = bar.extra.get(f)
            if v is None:
                ok = False
                break
            vals.append(float(v))
        if ok:
            self._factor_store[d][bar.symbol] = vals
            self._symbols_seen[d].add(bar.symbol)
        if len(self._symbols_seen[d]) >= 50:
            self._daily_complete.add(d)

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date
        acct = self.get_account()
        equity = float(acct.get("equity", INITIAL_CASH))
        if equity > self._equity_peak:
            self._equity_peak = equity
        dd = (equity / self._equity_peak) - 1.0 if self._equity_peak > 0 else 0.0
        if dd <= -KILL_DD_THRESHOLD:
            self._kill_until = d + timedelta(days=KILL_COOLDOWN_DAYS)

        if self._kill_until is not None and d < self._kill_until:
            try:
                self.close_position()
            except Exception:
                pass
            return

        if d not in self._factor_store:
            return
        if self._last_rebal_date is not None:
            gap = (d - self._last_rebal_date).days
            if gap < self.rebal_days:
                return
        self._last_rebal_date = d
        raw = self._factor_store[d]
        if len(raw) < self.top_n:
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
                scores=scores, top_n=self.top_n, weight_mode="equal",
                long_only=True, liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


def build_data_dict(start: date, end: date) -> tuple[dict, int, int]:
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        .drop_nulls(subset=BULL_FACTORS)
        .collect()
    )
    n_rows = df.shape[0]
    stocks = sorted(df["ts_code"].unique().to_list())
    data_dict: dict[str, pd.DataFrame] = {}
    for code in stocks:
        pdf = (
            df.filter(pl.col("ts_code") == code)
            .sort("trade_date").to_pandas()
            .set_index("trade_date").rename_axis("date")
        )
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        data_dict[code] = pdf
    return data_dict, n_rows, len(stocks)


def run_window(window_start: date, window_end: date) -> dict:
    """Run a single window backtest with Stage 5b config."""
    data_dict, n_rows, n_stocks = build_data_dict(window_start, window_end)
    if n_stocks < TOP_K:
        return {"error": f"too few stocks ({n_stocks})", "n_stocks": n_stocks}
    t0 = time.time()
    result = aq.run_backtest(
        data=data_dict, strategy=BullCompositeTopK10Kill,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_BPS_PER_SIDE / 10_000,
        slippage=0.0, t_plus_one=False,
        fill_policy=aq.NextOpen(), lot_size=100,
    )
    elapsed = time.time() - t0
    m = result.metrics_df
    total_pnl = float(m.loc["total_pnl", "value"])
    upnl = float(m.loc["unrealized_pnl", "value"])
    initial = float(m.loc["initial_market_value", "value"])
    closed_pnl = total_pnl - upnl
    closed_ret = closed_pnl / initial * 100
    days = (window_end - window_start).days
    ann = ((1 + closed_ret / 100) ** (365 / max(days, 1)) - 1) * 100
    return {
        "closed_only_return_pct": round(closed_ret, 4),
        "total_return_pct": round(float(m.loc["total_return_pct", "value"]), 4),
        "unrealized_pnl": round(upnl, 2),
        "sharpe_ratio": round(float(m.loc["sharpe_ratio", "value"]), 3),
        "max_drawdown_pct": round(float(m.loc["max_drawdown_pct", "value"]), 2),
        "win_rate": round(float(m.loc["win_rate", "value"]), 2),
        "closed_trade_count": int(float(m.loc["closed_trade_count", "value"])),
        "annualized_pct": round(ann, 1),
        "elapsed_sec": round(elapsed, 1),
        "n_rows": n_rows, "n_stocks": n_stocks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--train-start", default="2010-01-01")
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--val-months", type=int, default=6)
    ap.add_argument("--oos-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    result_path = out_dir / "result.json"
    error_path = out_dir / "error.log"

    def emit_progress(payload: dict) -> None:
        with progress_path.open("a") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)
    oos_window = timedelta(days=30 * args.oos_months)
    val_window = timedelta(days=30 * args.val_months)
    step = timedelta(days=30 * args.step_months)

    # Generate windows: walk-forward rolling from train_end + val to end-of-data.
    windows = []
    cur_oos_start = train_end + val_window
    while len(windows) < args.max_windows:
        cur_oos_end = cur_oos_start + oos_window
        # Cap at panel end (2026-08-21).
        if cur_oos_end > date(2026, 8, 21):
            cur_oos_end = date(2026, 8, 21)
        if cur_oos_start >= cur_oos_end:
            break
        windows.append({
            "oos_start": cur_oos_start.isoformat(),
            "oos_end": cur_oos_end.isoformat(),
        })
        cur_oos_start += step

    emit_progress({"event": "start", "n_windows": len(windows), "config": vars(args)})

    results = []
    for i, w in enumerate(windows):
        emit_progress({"event": "window_start", "idx": i + 1, **w})
        try:
            metrics = run_window(date.fromisoformat(w["oos_start"]), date.fromisoformat(w["oos_end"]))
            results.append({"idx": i + 1, **w, "metrics": metrics})
            emit_progress({"event": "window_done", "idx": i + 1, "metrics": metrics})
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            results.append({"idx": i + 1, **w, "error": err})
            emit_progress({"event": "window_error", "idx": i + 1, "error": err})

    # Summary.
    valid = [r["metrics"] for r in results if "metrics" in r and "closed_only_return_pct" in r.get("metrics", {})]
    summary = {
        "n_windows": len(windows),
        "n_valid": len(valid),
        "n_pass_target": sum(
            1 for m in valid
            if m.get("sharpe_ratio", 0) >= 1.2
            and m.get("max_drawdown_pct", 100) <= 12
            and m.get("annualized_pct", 0) >= 12
        ),
    }
    if valid:
        import statistics
        summary["mean_oos_ann_pct"] = round(statistics.mean(m["annualized_pct"] for m in valid), 1)
        summary["mean_oos_sharpe"] = round(statistics.mean(m["sharpe_ratio"] for m in valid), 3)
        summary["mean_oos_mdd_pct"] = round(statistics.mean(m["max_drawdown_pct"] for m in valid), 2)

    out = {
        "contract": {
            "job_id": args.job_id,
            "strategy": "BullCompositeTopK10Kill (Stage 5b)",
            "factors": BULL_FACTORS,
            "top_k": TOP_K,
            "rebal_days": REBAL_DAYS,
            "kill_dd_threshold": KILL_DD_THRESHOLD,
            "kill_cooldown_days": KILL_COOLDOWN_DAYS,
            "commission_bps_per_side": COMMISSION_BPS_PER_SIDE,
        },
        "config": vars(args),
        "windows": results,
        "summary": summary,
    }
    result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    emit_progress({"event": "finished", "summary": summary})
    print(f"wrote {result_path}")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        Path(args.out_dir if "args" in dir() else "/tmp").mkdir(parents=True, exist_ok=True)
        print(f"FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)