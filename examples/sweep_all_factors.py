"""Sweep all factors in the v10.2 panel through AKQuant, rank by metrics.

Contract (locked):
- Symbol: HS300 idx_close (single-symbol index-level backtest)
- Strategy: factor value > 5d MA -> 99% long; otherwise flat
- Commission: 25bps per side (50bps round-trip)
- Sweep universe: alpha_alpha001..101 + gtja_gtja_001..191 +
  mw_*_xs_rank + mw_ret_*/rs_*/align_*/break_*/up_days_* etc.

Output:
- stdout: top-N table
- evidence/<run_id>/sweep_results.json  (all 320 factors)
- evidence/<run_id>/top20.csv
- evidence/<run_id>/ranking.md (sorted by Sharpe)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import akquant as aq
from akquant import Bar, Strategy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from akquant_factor_backtest.panel_loader import (  # noqa: E402
    load_panel_for_factor,
    FACTOR_PANEL,
)

EVIDENCE_DIR = ROOT / "evidence" / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 5
TARGET_PCT = 0.99


class _SweepStrategy(Strategy):
    """Strategy body reused for every factor in the sweep."""

    warmup = WARMUP

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        # Signal column = the extra column we pre-set on the panel
        signal_col = getattr(self, "_signal_col", None)
        if signal_col is None:
            return
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


def collect_factor_columns() -> list[str]:
    """Return the full list of factor columns to sweep."""
    import polars as pl

    schema = pl.scan_parquet(str(FACTOR_PANEL)).collect_schema()
    cols = list(schema.keys())

    keep = []
    for c in cols:
        if c.startswith("alpha_alpha"):
            keep.append(c)
        elif c.startswith("gtja_gtja"):
            keep.append(c)
        elif c.startswith("mw_"):
            # skip the raw mw_ret / mw_above_ma family if it's an xs_rank
            # duplicate — keep both raw and xs_rank to compare
            keep.append(c)

    # de-dup, stable order
    seen = set()
    out = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def run_one(factor: str) -> dict | None:
    """Run one factor through AKQuant. Return metrics dict or None on error."""
    try:
        panel = load_panel_for_factor(factor)
    except Exception as exc:  # noqa: BLE001
        return {"factor": factor, "error": f"panel: {exc}"}

    if len(panel) < 50:
        return {"factor": factor, "error": "panel too short"}

    signal_col = f"factor_{factor}"

    # Attach the signal column to the strategy via a closure-free class attr.
    class Strategy(_SweepStrategy):
        pass

    Strategy._signal_col = signal_col

    try:
        result = aq.run_backtest(
            data=panel,
            strategy=Strategy,
            initial_cash=1_000_000.0,
            commission_rate=0.0025,
            slippage=0.0,
            t_plus_one=False,
            history_depth=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"factor": factor, "error": f"backtest: {exc}"}

    m = result.metrics_df
    # metrics_df is indexed by name; convert to a flat dict
    metrics = {}
    for name in m.index:
        try:
            metrics[name] = float(m.loc[name, "value"])
        except (TypeError, ValueError):
            metrics[name] = str(m.loc[name, "value"])
    return {"factor": factor, "panel_rows": len(panel), **metrics}


def main() -> int:
    factors = collect_factor_columns()
    print(f"== AKQuant factor sweep ==")
    print(f"engine: akquant {getattr(aq, '__version__', 'unknown')}")
    print(f"factor_panel: {FACTOR_PANEL}")
    print(f"factors to sweep: {len(factors)}")
    print(f"output: {EVIDENCE_DIR}")
    print()

    t0 = time.time()
    results = []
    ok, err = 0, 0
    for i, f in enumerate(factors, 1):
        t_one = time.time()
        r = run_one(f)
        elapsed_one = time.time() - t_one
        if r is None or "error" in r:
            err += 1
            print(f"  [{i:>4}/{len(factors)}] {f:<55} ERROR ({elapsed_one:.1f}s)")
        else:
            ok += 1
            sharpe = r.get("sharpe_ratio", float("nan"))
            print(
                f"  [{i:>4}/{len(factors)}] {f:<55} "
                f"sharpe={sharpe:+.3f}  trades={int(r.get('closed_trade_count', 0)):>4} "
                f"({elapsed_one:.1f}s)"
            )
        results.append(r)

    elapsed = time.time() - t0
    print(f"\n== sweep done ({elapsed:.1f}s, ok={ok}, err={err}) ==")

    # Write full JSON results.
    json_path = EVIDENCE_DIR / "sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"sweep_results.json: {json_path}")

    # Write top-N table ranked by Sharpe.
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r.get("sharpe_ratio", float("-inf")), reverse=True)
    top_n = 20
    top = valid[:top_n]

    csv_path = EVIDENCE_DIR / "top20.csv"
    with csv_path.open("w") as fh:
        header_cols = [
            "factor",
            "sharpe_ratio",
            "sortino_ratio",
            "profit_factor",
            "win_rate",
            "total_return_pct",
            "annualized_return",
            "max_drawdown_pct",
            "volatility",
            "closed_trade_count",
            "panel_rows",
        ]
        fh.write(",".join(header_cols) + "\n")
        for r in top:
            row = [str(r.get(c, "")) for c in header_cols]
            fh.write(",".join(row) + "\n")
    print(f"top20.csv: {csv_path}")

    # Markdown ranking report.
    md_path = EVIDENCE_DIR / "ranking.md"
    with md_path.open("w") as fh:
        fh.write(f"# AKQuant factor sweep ranking\n\n")
        fh.write(f"- engine: akquant {getattr(aq, '__version__', 'unknown')}\n")
        fh.write(f"- factors swept: {len(factors)} (ok={ok}, err={err})\n")
        fh.write(f"- strategy: factor > 5d MA -> 99% long; else flat\n")
        fh.write(f"- commission: 25bps per side; slippage: 0\n")
        fh.write(f"- elapsed: {elapsed:.1f}s\n\n")
        fh.write(f"## Top {top_n} by Sharpe\n\n")
        fh.write("| rank | factor | sharpe | sortino | PF | win% | return% | MDD% | trades |\n")
        fh.write("|---:|:---|---:|---:|---:|---:|---:|---:|---:|\n")
        for rank, r in enumerate(top, 1):
            sharpe = r.get("sharpe_ratio", 0)
            sortino = r.get("sortino_ratio", 0)
            pf = r.get("profit_factor", 0)
            wr = r.get("win_rate", 0)
            ret = r.get("total_return_pct", 0)
            mdd = r.get("max_drawdown_pct", 0)
            trades = int(r.get("closed_trade_count", 0))
            fh.write(
                f"| {rank} | `{r['factor']}` | {sharpe:+.3f} | {sortino:+.3f} | "
                f"{pf:.3f} | {wr:.2f} | {ret:+.2f} | {mdd:.2f} | {trades} |\n"
            )
        fh.write("\n## Notes\n\n")
        fh.write("- Sharpe ranking is over all 320 factors\n")
        fh.write("- top20.csv holds the same data for downstream consumption\n")
        fh.write("- Full per-factor results in sweep_results.json\n")
    print(f"ranking.md: {md_path}")

    # Print top-10 to stdout.
    print(f"\n=== Top 10 by Sharpe ===")
    print(f"{'rank':>4}  {'factor':<55}  sharpe  return%   MDD%")
    for rank, r in enumerate(top[:10], 1):
        sharpe = r.get("sharpe_ratio", 0)
        ret = r.get("total_return_pct", 0)
        mdd = r.get("max_drawdown_pct", 0)
        print(f"{rank:>4}  {r['factor']:<55}  {sharpe:+.3f}  {ret:+.2f}  {mdd:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
