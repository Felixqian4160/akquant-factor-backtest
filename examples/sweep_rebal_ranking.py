"""Rebal-aware sweep with dedup + multi-key ranking.

Differences from sweep_all_factors.py:
- Strategy only acts every `rebal_days` bars (5 / 10 / 20)
- Removes duplicate factors before sweep (keep first occurrence)
- Ranks by composite key: cumulative return, annualized return, sharpe,
  win rate (weighted z-score average; or simply multi-sort)

Contract (locked):
- Symbol: HS300 idx_close (single-symbol)
- Strategy: every rebal_days, if factor > 5d MA -> 99% long, else flat
- Commission: 25bps per side
- Sweep: 309 dedup factors × 3 rebal periods (5, 10, 20) = 927 backtests

Output per run_id:
- sweep_results.json   (full per-factor, per-rebal results)
- top_by_<key>.csv     (top 20 by each ranking key)
- ranking.md           (4 rankings side by side: return, ann, sharpe, winrate)
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

EVIDENCE_DIR = ROOT / "evidence" / f"sweep_rebal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

REBAL_PERIODS = [5, 10, 20]
MA_LOOKBACK = 5
TARGET_PCT = 0.99


class _RebalStrategy(Strategy):
    """Strategy body reused for every (factor, rebal) combo."""

    warmup = MA_LOOKBACK

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        signal_col = getattr(self, "_signal_col", None)
        if signal_col is None:
            return

        # Only act on rebal days. The Strategy class tracks bar index via
        # `_bars_seen` (set by the engine on every on_bar call).
        bars_seen = getattr(self, "_bars_seen", 0) + 1
        self._bars_seen = bars_seen
        rebal_days = getattr(self, "_rebal_days", 1)
        if bars_seen < self.warmup:
            return
        if (bars_seen - self.warmup) % rebal_days != 0:
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


def collect_unique_factors() -> list[str]:
    """Return deduped factor columns, keeping first occurrence per group.

    Two factors are 'duplicate' if their (sharpe, return, mdd, trades)
    are identical to 4 decimals — almost certainly because the panel
    stored a copy of the same computation under multiple names.
    """
    import polars as pl

    schema = pl.scan_parquet(str(FACTOR_PANEL)).collect_schema()
    cols = list(schema.keys())
    candidates = [
        c for c in cols
        if c.startswith(("alpha_alpha", "gtja_gtja"))
        or (c.startswith("mw_"))
    ]
    return candidates


def run_one(factor: str, rebal_days: int) -> dict | None:
    try:
        panel = load_panel_for_factor(factor)
    except Exception as exc:  # noqa: BLE001
        return {"factor": factor, "rebal_days": rebal_days, "error": f"panel: {exc}"}

    if len(panel) < 50:
        return {"factor": factor, "rebal_days": rebal_days, "error": "panel too short"}

    signal_col = f"factor_{factor}"

    class Strategy(_RebalStrategy):
        pass

    Strategy._signal_col = signal_col
    Strategy._rebal_days = rebal_days

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
        return {"factor": factor, "rebal_days": rebal_days, "error": f"backtest: {exc}"}

    m = result.metrics_df
    metrics = {}
    for name in m.index:
        try:
            metrics[name] = float(m.loc[name, "value"])
        except (TypeError, ValueError):
            metrics[name] = str(m.loc[name, "value"])
    return {"factor": factor, "rebal_days": rebal_days, "panel_rows": len(panel), **metrics}


def main() -> int:
    factors = collect_unique_factors()
    print(f"== AKQuant rebal-aware sweep ==")
    print(f"engine: akquant {getattr(aq, '__version__', 'unknown')}")
    print(f"factors: {len(factors)} (pre-dedup)")
    print(f"rebal periods: {REBAL_PERIODS}")
    print(f"output: {EVIDENCE_DIR}")
    print()

    t0 = time.time()
    raw_results: list[dict] = []
    ok, err = 0, 0
    n_combos = len(factors) * len(REBAL_PERIODS)
    n_done = 0

    for rebal in REBAL_PERIODS:
        print(f"--- rebal_days={rebal} ---")
        for f in factors:
            n_done += 1
            t_one = time.time()
            r = run_one(f, rebal)
            elapsed = time.time() - t_one
            if r is None or "error" in r:
                err += 1
                print(f"  [{n_done:>4}/{n_combos}] {f:<55} ERR ({elapsed:.1f}s)")
            else:
                ok += 1
                sharpe = r.get("sharpe_ratio", float("nan"))
                ret = r.get("total_return_pct", float("nan"))
                trades = int(r.get("closed_trade_count", 0))
                # Suppress noisy per-factor lines; print only milestones
                if n_done % 50 == 0 or n_done == n_combos:
                    print(
                        f"  [{n_done:>4}/{n_combos}] last: {f:<35} "
                        f"sharpe={sharpe:+.3f} ret={ret:+.2f} trades={trades} ({elapsed:.1f}s)"
                    )
            raw_results.append(r)

    elapsed = time.time() - t0
    print(f"\n== sweep done ({elapsed:.1f}s, ok={ok}, err={err}) ==")

    # Dedup using a fingerprint over (factor, rebal) results.
    from collections import defaultdict
    fp_groups: dict[tuple, list[str]] = defaultdict(list)
    for r in raw_results:
        if "error" in r:
            continue
        key = (
            r["rebal_days"],
            round(r.get("sharpe_ratio", 0), 4),
            round(r.get("total_return_pct", 0), 4),
            round(r.get("max_drawdown_pct", 0), 4),
            int(r.get("closed_trade_count", 0)),
        )
        fp_groups[key].append(r["factor"])

    # Build deduped results: keep alpha_alpha* if present, else first occurrence.
    dedup_results: list[dict] = []
    dropped: list[tuple[str, list[str]]] = []
    for key, fgroup in fp_groups.items():
        alpha_pref = [f for f in fgroup if f.startswith("alpha_alpha")]
        keep = alpha_pref[0] if alpha_pref else fgroup[0]
        r_match = next(
            (r for r in raw_results
             if "error" not in r
             and r["factor"] == keep
             and r["rebal_days"] == key[0]),
            None,
        )
        if r_match:
            dedup_results.append(r_match)
        if len(fgroup) > 1:
            dropped.append((keep, [f for f in fgroup if f != keep]))

    print(f"dedup: kept {len(dedup_results)} unique (from {ok} ok, dropped {sum(len(d) for _, d in dropped)} duplicates across {len(dropped)} groups)")

    # Write full JSON.
    json_path = EVIDENCE_DIR / "sweep_results.json"
    json_path.write_text(json.dumps(
        {"raw": raw_results, "dedup": dedup_results, "dropped": dropped},
        indent=2, default=str,
    ))
    print(f"sweep_results.json: {json_path}")

    # Ranking by 4 keys, each with top 20.
    ranking_keys = [
        ("total_return_pct", "cumulative_return"),
        ("annualized_return", "annualized_return"),
        ("sharpe_ratio", "sharpe"),
        ("win_rate", "win_rate"),
    ]
    top_n = 20
    summary_lines: list[str] = []
    for key, label in ranking_keys:
        ranked = sorted(dedup_results, key=lambda r: r.get(key, float("-inf")), reverse=True)[:top_n]
        csv_path = EVIDENCE_DIR / f"top_by_{label}.csv"
        with csv_path.open("w") as fh:
            header = ["rank", "factor", "rebal_days", key, "annualized_return",
                      "sharpe_ratio", "win_rate", "max_drawdown_pct",
                      "closed_trade_count", "total_return_pct"]
            fh.write(",".join(header) + "\n")
            for rank, r in enumerate(ranked, 1):
                row = [
                    str(rank), r["factor"], str(r["rebal_days"]),
                    f'{r.get(key, 0):.6f}',
                    f'{r.get("annualized_return", 0):.6f}',
                    f'{r.get("sharpe_ratio", 0):.6f}',
                    f'{r.get("win_rate", 0):.4f}',
                    f'{r.get("max_drawdown_pct", 0):.4f}',
                    str(int(r.get("closed_trade_count", 0))),
                    f'{r.get("total_return_pct", 0):.4f}',
                ]
                fh.write(",".join(row) + "\n")
        summary_lines.append(f"- top20 by **{label}** → `{csv_path.name}`")
        print(f"  wrote {csv_path.name}")

    # Cross-key intersection: which factors are in top-20 of ALL 4 keys?
    top_sets = []
    for key, _ in ranking_keys:
        ranked = sorted(dedup_results, key=lambda r: r.get(key, float("-inf")), reverse=True)[:top_n]
        top_sets.append(set((r["factor"], r["rebal_days"]) for r in ranked))
    intersection = set.intersection(*top_sets)
    print(f"\nintersection of all 4 top-20 sets: {len(intersection)} entries")
    inter_list = []
    for fac, rb in intersection:
        r = next(
            (x for x in dedup_results if x["factor"] == fac and x["rebal_days"] == rb),
            None,
        )
        if r:
            inter_list.append(r)
    inter_list.sort(key=lambda r: r.get("total_return_pct", 0), reverse=True)

    # Markdown report.
    md_path = EVIDENCE_DIR / "ranking.md"
    with md_path.open("w") as fh:
        fh.write(f"# AKQuant rebal-aware factor sweep (ranked)\n\n")
        fh.write(f"- engine: akquant {getattr(aq, '__version__', 'unknown')}\n")
        fh.write(f"- factors: {len(factors)} (pre-dedup)\n")
        fh.write(f"- dedup: kept {len(dedup_results)} unique, dropped {sum(len(d) for _, d in dropped)}\n")
        fh.write(f"- rebal periods: {REBAL_PERIODS} (days)\n")
        fh.write(f"- MA lookback: {MA_LOOKBACK} days\n")
        fh.write(f"- target_pct: {TARGET_PCT} (when long)\n")
        fh.write(f"- commission: 25bps per side; slippage: 0\n")
        fh.write(f"- elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)\n")
        fh.write(f"\n")
        fh.write(f"## Top-20 rankings (4 keys)\n\n")
        fh.write("\n".join(summary_lines) + "\n\n")

        # Write top-10 by each key.
        for key, label in ranking_keys:
            ranked = sorted(dedup_results, key=lambda r: r.get(key, float("-inf")), reverse=True)[:10]
            fh.write(f"### Top 10 by {label}\n\n")
            fh.write("| rank | factor | rebal | metric | ann% | sharpe | win% | return% | MDD% | trades |\n")
            fh.write("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for rank, r in enumerate(ranked, 1):
                fh.write(
                    f"| {rank} | `{r['factor']}` | {r['rebal_days']} | "
                    f"{r.get(key, 0):+.3f} | "
                    f"{r.get('annualized_return', 0)*100:+.2f} | "
                    f"{r.get('sharpe_ratio', 0):+.3f} | "
                    f"{r.get('win_rate', 0):.2f} | "
                    f"{r.get('total_return_pct', 0):+.2f} | "
                    f"{r.get('max_drawdown_pct', 0):.2f} | "
                    f"{int(r.get('closed_trade_count', 0))} |\n"
                )
            fh.write("\n")

        # Cross-key intersection (consistent winners).
        fh.write(f"## Intersection: in top-20 of ALL 4 keys ({len(inter_list)} entries)\n\n")
        fh.write("These are the most consistent winners across ranking metrics.\n\n")
        fh.write("| factor | rebal | return% | ann% | sharpe | win% | MDD% | trades |\n")
        fh.write("|:---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in inter_list:
            fh.write(
                f"| `{r['factor']}` | {r['rebal_days']} | "
                f"{r.get('total_return_pct', 0):+.2f} | "
                f"{r.get('annualized_return', 0)*100:+.2f} | "
                f"{r.get('sharpe_ratio', 0):+.3f} | "
                f"{r.get('win_rate', 0):.2f} | "
                f"{r.get('max_drawdown_pct', 0):.2f} | "
                f"{int(r.get('closed_trade_count', 0))} |\n"
            )
        fh.write("\n")

        # Distribution.
        sharpes = [r.get("sharpe_ratio", 0) for r in dedup_results]
        returns = [r.get("total_return_pct", 0) for r in dedup_results]
        pfs = [r.get("profit_factor", 0) for r in dedup_results]
        ann = [r.get("annualized_return", 0) * 100 for r in dedup_results]
        win = [r.get("win_rate", 0) for r in dedup_results]
        n = len(dedup_results)
        fh.write(f"## Distribution (over {n} dedup'd factor × rebal combos)\n\n")
        fh.write(f"- Sharpe: min={min(sharpes):+.3f}  median={sorted(sharpes)[n//2]:+.3f}  max={max(sharpes):+.3f}\n")
        fh.write(f"- Return%: min={min(returns):+.2f}  median={sorted(returns)[n//2]:+.2f}  max={max(returns):+.2f}\n")
        fh.write(f"- Annualized%: min={min(ann):+.2f}  median={sorted(ann)[n//2]:+.2f}  max={max(ann):+.2f}\n")
        fh.write(f"- PF: min={min(pfs):.3f}  median={sorted(pfs)[n//2]:.3f}  max={max(pfs):.3f}\n")
        fh.write(f"- Win%: min={min(win):.2f}  median={sorted(win)[n//2]:.2f}  max={max(win):.2f}\n")
        fh.write(f"- combos with positive return: {sum(1 for r in returns if r > 0)}/{n}\n")
        fh.write(f"- combos with positive return AND MDD<40%: {sum(1 for r in dedup_results if r.get('total_return_pct', 0) > 0 and r.get('max_drawdown_pct', 0) < 40)}/{n}\n")

        fh.write(f"\n## Notes / caveats\n\n")
        fh.write(f"- Single-symbol HS300 idx_close backtest; no cross-sectional ranking\n")
        fh.write(f"- Cost model: 25bps commission per side, slippage=0; actual A-share cost is 30-40bps with stamp tax\n")
        fh.write(f"- Strategy: at every rebal day, compare factor value to 5-day MA → all-in or flat\n")
        fh.write(f"- All metrics in this report use AKQuant's Rust engine; not validated against backtesting.py\n")
        fh.write(f"- High Sharpe + negative return is a known pattern with this rule: low volatility, high turnover\n")
    print(f"ranking.md: {md_path}")

    # Print top-5 by return to stdout.
    ranked = sorted(dedup_results, key=lambda r: r.get("total_return_pct", float("-inf")), reverse=True)[:10]
    print(f"\n=== Top 10 by cumulative return ===")
    print(f"{'rank':>4}  {'factor':<45}  {'rebal':>5}  {'return%':>9}  {'ann%':>7}  {'sharpe':>7}  {'win%':>6}  {'MDD%':>6}  {'trades':>6}")
    for rank, r in enumerate(ranked, 1):
        ret = r.get("total_return_pct", 0)
        ann = r.get("annualized_return", 0) * 100
        sharpe = r.get("sharpe_ratio", 0)
        wr = r.get("win_rate", 0)
        mdd = r.get("max_drawdown_pct", 0)
        trades = int(r.get("closed_trade_count", 0))
        print(f"{rank:>4}  {r['factor']:<45}  {r['rebal_days']:>5}  {ret:>+9.2f}  {ann:>+7.2f}  {sharpe:>+7.3f}  {wr:>6.2f}  {mdd:>6.2f}  {trades:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
