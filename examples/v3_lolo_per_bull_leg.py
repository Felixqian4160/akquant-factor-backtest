"""v3 LOLO per-bull-leg validation.

Reuses the v3 sweep contract (single-symbol HS300 trend-following on
factor > MA(5d) -> long 99%, else flat, with end-of-period force-close).
But instead of running over the full 2004-2026 panel, this slices the
panel into the 10 bull legs (>=100d up legs from hs300_index_pivots.json)
and runs one backtest per leg, reporting the per-leg closed-only return.

Why:
- v3 full-panel closed_only for gtja_gtja_132 (top 1) was +2.39%
  (rebal=10). This averaged across regimes including bear / sideways.
- bull legs are the regime where trend-following SHOULD work.
- If per-bull-leg returns are positive and consistent, we have
  regime-specific evidence for a bull-leg specialist.
- If returns are inconsistent (some negative), the v3 single-symbol
  alpha was just averaging artifacts.

Single-variable: change ONLY the panel slicing (full -> per-leg).
Factor: gtja_gtja_132 (v3 closed-only top 1)
Rebal: 10 (v3 best)
Target_pct: 0.99, commission: 25bps/side, slippage: 0
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import akquant as aq
from akquant import Bar, Strategy

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

from akquant_factor_backtest.panel_loader import (  # noqa: E402
    load_panel_for_factor,
)

OUT_DIR = ROOT / "evidence" / f"v3_lolo_per_bull_leg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FACTOR = "gtja_gtja_132"
REBAL_DAYS = 10
MA_LOOKBACK = 5
TARGET_PCT = 0.99

BULL_LEGS_PATH = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_index_leg_splits_100d_20260920/"
    "bull_legs.json"
)


class _LegStrategy(Strategy):
    """Single-symbol trend-following on factor > MA(5d)."""

    warmup = MA_LOOKBACK

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bars_seen = 0
        self._rebal_countdown = REBAL_DAYS  # trade on first eligible bar

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        self._bars_seen += 1
        if self._bars_seen < self.warmup:
            return
        if self._rebal_countdown < REBAL_DAYS:
            self._rebal_countdown += 1
            return
        self._rebal_countdown = 0

        signal_col = f"factor_{FACTOR}"
        history = self.get_history(self.warmup, symbol, signal_col)
        if history is None or len(history) < self.warmup:
            return
        current = bar.extra.get(signal_col)
        if current is None:
            return
        ma = sum(history) / len(history)
        pos = self.get_position(symbol)
        if current > ma and pos == 0:
            self.order_target_percent(target_percent=TARGET_PCT, symbol=symbol)
        elif current <= ma and pos > 0:
            self.order_target_percent(target_percent=0.0, symbol=symbol)


def _closed_only_pct(metrics_df) -> tuple[float, float]:
    total_pnl = float(metrics_df.loc["total_pnl", "value"])
    upnl = float(metrics_df.loc["unrealized_pnl", "value"])
    initial = float(metrics_df.loc["initial_market_value", "value"])
    return (total_pnl - upnl) / initial * 100, upnl


def main() -> int:
    legs = json.loads(BULL_LEGS_PATH.read_text())
    print(f"== v3 LOLO per-bull-leg validation ==")
    print(f"factor: {FACTOR}  rebal: {REBAL_DAYS}d  target_pct: {TARGET_PCT}")
    print(f"output: {OUT_DIR}")
    print(f"bull legs: {len(legs)}")

    # Load full panel once, slice per leg
    panel = load_panel_for_factor(FACTOR)
    signal_col = f"factor_{FACTOR}"
    panel_dates = panel.index  # DatetimeIndex

    results = []
    for leg in legs:
        idx = leg["idx"]
        start = leg["start_date"]
        end = leg["end_date"]
        # Filter panel to leg date range (inclusive)
        leg_panel = panel[(panel_dates >= start) & (panel_dates <= end)].copy()
        if len(leg_panel) < MA_LOOKBACK + 2:
            print(f"  leg#{idx:>2} {start}~{end}: too short ({len(leg_panel)} bars), skip")
            results.append({"leg_idx": idx, "start": start, "end": end, "bars": len(leg_panel), "error": "too_short"})
            continue

        try:
            t0 = time.time()
            result = aq.run_backtest(
                data=leg_panel,
                strategy=_LegStrategy,
                initial_cash=1_000_000.0,
                commission_rate=0.0025,
                slippage=0.0,
                t_plus_one=False,
                history_depth=MA_LOOKBACK,
            )
            elapsed = time.time() - t0
            m = result.metrics_df
            closed_ret, upnl = _closed_only_pct(m)
            total_ret = float(m.loc["total_return_pct", "value"])
            sharpe = float(m.loc["sharpe_ratio", "value"])
            mdd = float(m.loc["max_drawdown_pct", "value"])
            wr = float(m.loc["win_rate", "value"])
            trades = int(float(m.loc["closed_trade_count", "value"]))
            rec = {
                "leg_idx": idx, "start": start, "end": end, "days": leg["days"],
                "bars": len(leg_panel),
                "closed_only_return_pct": round(closed_ret, 4),
                "total_return_pct": round(total_ret, 4),
                "unrealized_pnl": round(upnl, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(mdd, 2),
                "win_rate": round(wr, 2),
                "closed_trade_count": trades,
                "elapsed_s": round(elapsed, 1),
            }
            results.append(rec)
            status = "✅" if closed_ret > 0 else "❌"
            print(f"  {status} leg#{idx:>2} {start}~{end} bars={len(leg_panel):>4} "
                  f"closed={closed_ret:+6.2f}% total={total_ret:+7.2f}% "
                  f"sharpe={sharpe:+.3f} wr={wr:5.1f}% mdd={mdd:5.1f}% "
                  f"trades={trades:>3} ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"  leg#{idx:>2} {start}~{end}: ERROR {type(exc).__name__}: {exc}")
            results.append({"leg_idx": idx, "start": start, "end": end, "error": str(exc)})

    # Save results
    out_path = OUT_DIR / "per_leg_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    valid_results = [r for r in results if "closed_only_return_pct" in r]
    n_pos = sum(1 for r in valid_results if r["closed_only_return_pct"] > 0)
    print()
    print(f"== summary ({len(valid_results)} valid legs) ==")
    print(f"  positive closed_only: {n_pos}/{len(valid_results)}")
    if valid_results:
        closed_rets = [r["closed_only_return_pct"] for r in valid_results]
        import statistics
        print(f"  mean closed_only: {statistics.mean(closed_rets):+.2f}%")
        print(f"  median closed_only: {statistics.median(closed_rets):+.2f}%")
        print(f"  min: {min(closed_rets):+.2f}%  max: {max(closed_rets):+.2f}%")

    # Write ranking CSV
    csv_path = OUT_DIR / "per_leg_summary.csv"
    with csv_path.open("w") as fh:
        fh.write("leg_idx,start,end,days,bars,closed_only_return_pct,total_return_pct,unrealized_pnl,sharpe_ratio,max_drawdown_pct,win_rate,closed_trade_count,elapsed_s\n")
        for r in valid_results:
            fh.write(",".join(str(r[k]) for k in [
                "leg_idx", "start", "end", "days", "bars",
                "closed_only_return_pct", "total_return_pct", "unrealized_pnl",
                "sharpe_ratio", "max_drawdown_pct", "win_rate",
                "closed_trade_count", "elapsed_s",
            ]) + "\n")
    print(f"\nwrote {out_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())