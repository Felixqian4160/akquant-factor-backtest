"""v1 Stage 3: per-regime multi-symbol composite verification.

Reuses the existing BullCompositeTopK strategy (which has been verified
on the full 2004-2026 panel with +451% closed-only), but instead of
running the full panel, slices into bull/bear legs and runs ONE backtest
per leg. This is the per-regime alpha audit for Stage 3.

Goal: per-regime closed_only_return + sharpe + MDD, to verify whether
multi-symbol composite alpha is robust per leg (not averaged across
regimes). Stage 3 single-variable: panel slicing (full -> per-leg).

Stage 3 is independent of Stage 2a cross-check. It directly measures
per-bull-leg alpha. Bear legs use a separate strategy (BearReversionTopK)
to validate the bear-side has its own working edge.
"""
from __future__ import annotations

import json
import sys
import time
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import akquant as aq
from akquant import Bar, Strategy

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

from akquant_factor_backtest.panel_loader import (  # noqa: E402
    load_panel_for_factor,
)

OUT_DIR = ROOT / "evidence" / f"factor_research_workflow_v1_stage3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)
BULL_LEGS = json.loads(Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_index_leg_splits_100d_20260920/"
    "bull_legs.json"
).read_text())
BEAR_LEGS = json.loads(Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_index_leg_splits_100d_20260920/"
    "bear_legs.json"
).read_text())

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]
TOP_K = 20
REBAL_DAYS = 20
INITIAL_CASH = 1_000_000.0
COMMISSION = 0.0025


class BullCompositeTopK(Strategy):
    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._seen = 0
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        self._daily_complete: set = set()

    def on_bar(self, bar: Bar) -> None:
        self._seen += 1
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


def build_leg_panel(leg: dict) -> pl.DataFrame:
    """Filter the full panel to a single leg's date range, drop null factors."""
    s = date.fromisoformat(leg["start_date"])
    e = date.fromisoformat(leg["end_date"])
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
        .drop_nulls(subset=BULL_FACTORS)
        .collect()
    )
    return df


def to_data_dict(df: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert polars → Dict[ts_code, pd.DataFrame] for AKQuant."""
    out: dict[str, pd.DataFrame] = {}
    for code, sub in df.partition_by("ts_code", as_dict=True).items():
        s = sub.sort("trade_date")
        out[code] = pd.DataFrame({
            "open": s["open"].to_numpy(),
            "high": s["high"].to_numpy(),
            "low": s["low"].to_numpy(),
            "close": s["close"].to_numpy(),
            "volume": np.where(s["vol"].to_numpy() == 0, 1e9, s["vol"].to_numpy()),
        }, index=pd.DatetimeIndex(s["trade_date"].to_numpy())).rename_axis("date")
    # Attach factors as bar.extra via extras dict? AKQuant reads bar.extra
    # for the current bar; we embed per-symbol-per-date factor values via
    # on_bar. Strategy reads bar.extra.get(f). We need to expose factors
    # alongside the bars.
    # Workaround: add factors as columns on the DataFrame and let AKQuant
    # surface them as bar.extra. AKQuant 0.3.64 only emits OHLCV; the
    # strategy will not see extras. Therefore we need a different design:
    # use the panel data directly with a custom feed (out of scope here).
    # For Stage 3 we use a proxy: compute composite at portfolio-build time
    # from the symbol's most recent factor row (re-using the previous day).
    raise NotImplementedError(
        "AKQuant 0.3.64 doesn't expose DataFrame extra cols as bar.extra; "
        "bull_composite achieved it via custom feed. Stage 3 needs the same "
        "feed extension — deferred."
    )


def _closed_only_pct(metrics_df) -> tuple[float, float]:
    total_pnl = float(metrics_df.loc["total_pnl", "value"])
    upnl = float(metrics_df.loc["unrealized_pnl", "value"])
    initial = float(metrics_df.loc["initial_market_value", "value"])
    return (total_pnl - upnl) / initial * 100, upnl


def main() -> int:
    print("== v1 Stage 3: per-regime multi-symbol composite ==")
    print(f"factors: {BULL_FACTORS}")
    print(f"top_k={TOP_K}, rebal={REBAL_DAYS}, commission={COMMISSION*1e4:.0f}bps")
    print(f"output: {OUT_DIR}")

    results = []
    for leg in BULL_LEGS:
        idx = leg["idx"]
        s = leg["start_date"]
        e = leg["end_date"]
        try:
            df = build_leg_panel(leg)
        except Exception as exc:
            print(f"  leg#{idx:>2} {s}~{e}: panel build error: {exc}")
            results.append({"leg_idx": idx, "regime": "bull", "error": str(exc)})
            continue

        if df.shape[0] < 200:
            print(f"  leg#{idx:>2} {s}~{e}: too few rows ({df.shape[0]}), skip")
            results.append({"leg_idx": idx, "regime": "bull", "error": "too_few_rows"})
            continue

        stocks = df["ts_code"].n_unique()
        print(f"  leg#{idx:>2} {s}~{e}: rows={df.shape[0]}, stocks={stocks}")

    # NOTE: This is a smoke stage to verify pipeline correctness.
    # The actual per-leg run is blocked on the DataFrame extras issue
    # (see to_data_dict NotImplementedError). The full Stage 3 backtest
    # requires the AKQuant custom-feed extension that bull_composite used.
    # For now, write the plan + smoke output, defer the heavy run.

    summary = {
        "method": "Stage 3 per-regime multi-symbol composite (PLANNED, not run)",
        "reason": "AKQuant 0.3.64 doesn't surface DataFrame extra cols as bar.extra; "
                  "bull_composite used a custom feed. Stage 3 needs that extension.",
        "panels_per_leg": [
            {"leg_idx": l["idx"], "regime": "bull", "start": l["start_date"],
             "end": l["end_date"], "days": l["days"]}
            for l in BULL_LEGS
        ],
        "blocked_on": "DataFrame extras → bar.extra pipeline",
    }
    (OUT_DIR / "stage3_plan.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote plan to {OUT_DIR}/stage3_plan.json")
    print("Stage 3 BLOCKED on DataFrame extras → bar.extra pipeline.")
    print("Need to either (a) extend AKQuant custom feed, or (b) use simpler proxy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())