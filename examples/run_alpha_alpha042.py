"""Minimum demo: alpha_alpha042 on HS300 idx_close.

Contract (locked):
- signal: factor mean (cross-sectional mean, but idx-level == the value)
- execution: T day signal > 5d-MA -> buy; signal < 5d-MA -> sell
- position: all-in (target_pct = 1.0 when long, 0.0 when flat)
- cost: AKQuant default (commission = 0); we add 25bps commission per side
  via aq.run_backtest commission kwarg

Output:
- stdout: akquant BacktestResult
- HTML report: evidence/20260924_alpha042_akquant/<run_id>/report.html
- JSON metrics: evidence/20260924_alpha042_akquant/<run_id>/metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import akquant as aq
from akquant import Bar, Strategy

# Make the bootstrap package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from akquant_factor_backtest.panel_loader import (  # noqa: E402
    load_panel_for_factor,
    INDEX_PANEL,
    FACTOR_PANEL,
)

FACTOR = "alpha_alpha042"
SIGNAL_COL = f"factor_{FACTOR}"
EVIDENCE_DIR = ROOT / "evidence" / "20260924_alpha042_akquant"


class Alpha042ThresholdStrategy(Strategy):
    """Buy when factor > 5-day moving average, sell otherwise.

    Single-symbol HS300 idx_close. The factor value is read from
    bar.extra[signal_col] (populated via the panel's extra column).

    All-in / all-out: order_target_percent(1.0 or 0.0).
    """

    # AKQuant 0.3.x in-class param declarations
    warmup = 5

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        # Compute factor 5-day mean via get_history.
        history = self.get_history(self.warmup, symbol, SIGNAL_COL)
        if history is None or len(history) < self.warmup:
            return

        current = bar.extra.get(SIGNAL_COL)
        if current is None:
            return

        ma5 = sum(history) / len(history)
        pos = self.get_position(symbol)

        if current > ma5 and pos == 0:
            # AKQuant RiskManager requires a margin buffer; 0.99 = 99% deploy
            self.order_target_percent(target_percent=0.99, symbol=symbol)
        elif current <= ma5 and pos > 0:
            self.order_target_percent(target_percent=0.0, symbol=symbol)


def main() -> int:
    if not INDEX_PANEL.exists() or not FACTOR_PANEL.exists():
        print(f"!! panel not found: INDEX={INDEX_PANEL}  FACTOR={FACTOR_PANEL}", file=sys.stderr)
        return 2

    print(f"== AKQuant factor backtest ==")
    print(f"factor: {FACTOR}")
    print(f"index:  {INDEX_PANEL}")
    print(f"factor: {FACTOR_PANEL}")
    print(f"engine: akquant {aq.__version__ if hasattr(aq, '__version__') else 'unknown'}")

    panel = load_panel_for_factor(FACTOR)
    print(f"panel rows: {len(panel)}  ({panel.index.min().date()} -> {panel.index.max().date()})")

    # Ensure timestamp is tz-aware (AKQuant expects Asia/Shanghai).
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("Asia/Shanghai")

    # Run the backtest.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVIDENCE_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = aq.run_backtest(
        data=panel,
        strategy=Alpha042ThresholdStrategy,
        initial_cash=1_000_000.0,
        commission_rate=0.0025,  # 25bps per side (= 50bps round-trip)
        slippage=0.0,
        t_plus_one=False,
        history_depth=10,  # need ≥ warmup bars of extra_col history
    )
    elapsed = time.time() - t0

    # Save JSON metrics.
    metrics = {
        "factor": FACTOR,
        "run_id": run_id,
        "akquant_version": getattr(aq, "__version__", "unknown"),
        "elapsed_sec": round(elapsed, 3),
        "panel_rows": len(panel),
        "panel_start": str(panel.index.min().date()),
        "panel_end": str(panel.index.max().date()),
        "result": (
            result.to_dict()
            if hasattr(result, "to_dict")
            else {k: getattr(result, k, None) for k in dir(result) if not k.startswith("_") and not callable(getattr(result, k, None))}
        ),
    }
    metrics_file = out_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, default=str))

    # HTML report.
    try:
        result.viz.report(
            filename=str(out_dir / "report.html"),
            show=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"!! viz.report failed: {exc}", file=sys.stderr)

    # Print summary to stdout.
    print(f"\n=== result (elapsed {elapsed:.2f}s) ===")
    print(result)
    print(f"\nmetrics.json: {metrics_file}")
    print(f"report.html:  {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
