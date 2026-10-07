"""ALL_FUSION_DYNAMIC v3 — 简化版: 复用 V37 V14_0 picks + V32 dynamic target.

复用 V37 已有 picks (cache 修复后 V14_0):
- 168 rebal dates
- V27 original (2/10/4) router
- V19 historical factor-return ranking (bear)
- V14 IC voting (bull)
- MAX_STOCKS = 10

加上 V32 dynamic target (vol × dd, base=0.005, slope=7.0):
- vol_factor = clip(0.005/vol, 0.2, 1.0)
- dd_factor = clip(1 + dd × 7, 0.25, 1.0)
- target = clip(0.99 × vol × dd, 0.1, 0.99)

跑 AKQuant, 对比 V37 V14_0 static 0.99 target.

然后 5-seed sweep: 复用 V37 已有 V14_0/4/8/12/16 picks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
V37_PICKS_BASE = Path("evidence/v37_v14_causal")
OUT_BASE = Path("evidence/all_fusion_v3_dynamic_target_20261006")
OUT_BASE.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2025-12-31")
DATA_START = pd.Timestamp("2010-01-01")

VOL_BASE = 0.005
DD_SLOPE = 7.0


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dyn_target(equity_history: list[float]) -> float:
    if len(equity_history) < 5:
        return 0.99
    arr = np.array(equity_history[-21:], dtype=np.float64)
    if arr.mean() <= 0:
        return 0.5
    ret = arr[1:] / arr[:-1] - 1.0
    vol = float(ret.std()) if len(ret) > 1 else 0.01
    vol_factor = float(np.clip(VOL_BASE / max(vol, 0.005), 0.2, 1.0))
    peak = arr.max()
    dd = arr[-1] / peak - 1.0 if peak > 0 else 0.0
    dd_factor = float(np.clip(1.0 + dd * DD_SLOPE, 0.25, 1.0))
    target = 0.99 * vol_factor * dd_factor
    return float(np.clip(target, 0.1, 0.99))


def build_data(universe: list[str]):
    cols = ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
    source = pl.read_parquet(PANEL, columns=cols).filter(
        (pl.col("trade_date") >= DATA_START.to_pydatetime()) &
        (pl.col("trade_date") <= END.to_pydatetime())
    )
    all_dates = source.get_column("trade_date").unique().sort().to_list()
    date_grid = pl.DataFrame({"trade_date": all_dates}).with_columns(
        pl.col("trade_date").cast(pl.Datetime("ms"))
    )
    data = {}
    for symbol in universe:
        bars = source.filter(pl.col("ts_code") == symbol).sort("trade_date")
        if bars.is_empty():
            continue
        pdf = (date_grid.join(bars, on="trade_date", how="left")
               .sort("trade_date")
               .with_columns([
                   pl.col("ts_code").fill_null(symbol),
                   pl.col("close").forward_fill(),
                   pl.col("open").fill_null(pl.col("close").forward_fill()),
                   pl.col("high").fill_null(pl.col("close").forward_fill()),
                   pl.col("low").fill_null(pl.col("close").forward_fill()),
                   pl.col("vol").fill_null(0.0),
               ])
               .to_pandas()
               .set_index("trade_date")
               .rename_axis("date"))
        pdf["symbol"] = symbol
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1.0e9
        data[symbol] = pdf
    return data


def run_seed(tag: str) -> dict:
    """Run AKQuant for V37 {tag} picks with V32 dynamic target."""
    picks_path = V37_PICKS_BASE / tag / "picks.json"
    if not picks_path.exists():
        return {"error": f"no picks for {tag}"}
    log(f"  Loading {tag} picks...")
    picks_obj = json.loads(picks_path.read_text())
    universe = sorted({s for p in picks_obj.values() for s in p})
    log(f"  universe: {len(universe)}, picks dates: {len(picks_obj)}")

    data = build_data(universe)

    _DAILY_PICKS = picks_obj
    _REBAL_SET = set(picks_obj.keys())
    _NAV_HISTORY = [INITIAL_CASH]
    _TARGETS_LOG = []

    class DynTargetStrategy(aq.Strategy):
        warmup = 5
        def on_bar(self, bar): pass
        def on_after_trading(self, trading_date, timestamp):
            try:
                eq = float(self.equity)
                if eq > 0:
                    if not _NAV_HISTORY or _NAV_HISTORY[-1] != eq:
                        _NAV_HISTORY.append(eq)
                        if len(_NAV_HISTORY) > 30:
                            _NAV_HISTORY.pop(0)
            except Exception:
                pass
        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET: return
            p = _DAILY_PICKS.get(d, {})
            if not p: return
            n = len(p)
            target = dyn_target(_NAV_HISTORY)
            self.rebalance_weights(
                target_weights={s: target/n for s in p},
                liquidate_unmentioned=True,
            )
            _TARGETS_LOG.append((d, target, _NAV_HISTORY[-1] if _NAV_HISTORY else 0))

    log(f"  Running AKQuant...")
    t0 = time.time()
    result = aq.run_backtest(
        data=data, strategy=DynTargetStrategy,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        t_plus_one=False,
        fill_policy=aq.NextOpen(),
        lot_size=LOT_SIZE,
    )
    log(f"  AKQuant done in {time.time()-t0:.1f}s")

    metrics = result.metrics_df
    metrics_dict = {}
    for index in metrics.index:
        value = metrics.loc[index, "value"]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        try:
            metrics_dict[index] = float(value)
        except (TypeError, ValueError):
            metrics_dict[index] = str(value)

    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "tag": tag,
            "contract": {
                "picks_source": f"v37_v14_causal/{tag}/picks.json",
                "router": "V27 original (2/10/4) — inherited from v37",
                "dynamic_target": f"vol_base={VOL_BASE}, dd_slope={DD_SLOPE}",
                "t_plus_one": False, "round_trip_cost": 0.005, "lot_size": 100,
            },
            "metrics": metrics_dict,
            "targets_log_count": len(_TARGETS_LOG),
            "universe_size": len(universe),
        }, f, indent=2, ensure_ascii=False)
    return metrics_dict


def main():
    log("=" * 80)
    log("ALL_FUSION_DYNAMIC v3 — 复用 V37 picks + V32 dynamic target (vol×dd)")
    log("=" * 80)
    summary = {}
    for tag in ["V14_0", "V14_4", "V14_8", "V14_12", "V14_16"]:
        log(f"\n=== {tag} ===")
        try:
            m = run_seed(tag)
            if 'error' in m:
                summary[tag] = m
                continue
            ret = m.get("total_return_pct", 0)
            sharpe = m.get("sharpe_ratio", 0)
            mdd = m.get("max_drawdown_pct", 0)
            pf = m.get("profit_factor", 0)
            win = m.get("win_rate", 0)
            n_trades = int(m.get("closed_trade_count", 0))
            summary[tag] = {"ret": ret, "sharpe": sharpe, "mdd": mdd,
                            "pf": pf, "win": win, "n_trades": n_trades}
            marker = "✅" if ret > 0 else "❌"
            log(f"  {marker} {tag}: +{ret:.2f}% / sharpe {sharpe:.3f} / MDD -{mdd:.2f}% / PF {pf:.3f} / {n_trades} trades")
        except Exception as e:
            log(f"  {tag} FAILED: {e}")
            summary[tag] = {"error": str(e)}

    log("\n=== 5-seed summary ===")
    log(f"{'tag':>8} {'ret':>8} {'sharpe':>7} {'MDD':>7} {'PF':>5} {'Win':>5} {'Trades':>7}")
    for tag, m in summary.items():
        if 'error' in m:
            log(f"  {tag:>6} ERROR")
            continue
        marker = "✅" if m['ret'] > 0 else "❌"
        log(f"  {marker} {tag:>6} {m['ret']:>+7.2f}% {m['sharpe']:>7.3f} {m['mdd']:>6.2f}% {m['pf']:>5.3f} {m['win']:>4.1f}% {m['n_trades']:>7}")

    rets = [m['ret'] for m in summary.values() if 'ret' in m]
    if rets:
        log(f"\n  mean: {sum(rets)/len(rets):+.2f}%")
        log(f"  median: {sorted(rets)[len(rets)//2]:+.2f}%")
        log(f"  spread: {max(rets)-min(rets):.2f}%")
        log(f"  n_pos: {sum(1 for r in rets if r > 0)}/{len(rets)}")
    log("\n对比 V37 V14 static 0.99 (从 SEED_SWEEP_REPORT):")
    log("  V14_0: +401.32% / V14_4: +269.39% / V14_8: +65.82%")
    log("  V14_12: +6.37% / V14_16: +168.95% / mean +182.37%")

    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()