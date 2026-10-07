"""Baseline comparison: bull composite vs buy-hold vs random vs reverse.

Same 2513 bull-leg dates, same 351 stocks. Three baselines:
  1. buy_hold: equal-weight top-20 static basket (held entire period)
  2. random_top20: every rebal, pick 20 random stocks equal-weight
  3. reverse_top20: every rebal, pick 20 LOWEST-composite stocks

If our bull composite really has alpha, it should beat all three.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import akquant as aq
from akquant import Bar, Strategy

PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)
PIVOTS = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "hs300_index_pivots_clean_20260919/"
    "hs300_index_pivots.json"
)
OUT = Path("/media/felix/f/quant/akquant-factor-backtest/evidence/bull_composite_backtest")
OUT.mkdir(parents=True, exist_ok=True)

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]
TOP_K = 20
REBAL_DAYS = 20
COST_BPS_PER_SIDE = 40
INITIAL_CASH = 1_000_000.0


def bull_date_set() -> set[date]:
    data = json.load(open(PIVOTS))
    out: set[date] = set()
    for leg in data["legs"]:
        if leg["kind"] != "up" or leg["duration_days"] < 100:
            continue
        st = date.fromisoformat(leg["start"])
        en = date.fromisoformat(leg["end"])
        d = st
        while d <= en:
            out.add(d)
            d += timedelta(days=1)
    return out


class BullCompositeTopK(Strategy):
    """Top-K by composite — our proposed alpha strategy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._last_rebal_date = None

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
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

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date
        if d not in self._factor_store:
            return
        if self._last_rebal_date is not None:
            if (d - self._last_rebal_date).days < REBAL_DAYS:
                return
        self._last_rebal_date = d
        raw = self._factor_store[d]
        if len(raw) < TOP_K:
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
                scores=scores, top_n=TOP_K, weight_mode="equal",
                long_only=True, liquidate_unmentioned=True,
            )
        except Exception:
            pass


class ReverseTopK(Strategy):
    """Bottom-K (worst composite) — should LOSE money if our top-K has alpha."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._last_rebal_date = None

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
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

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date
        if d not in self._factor_store:
            return
        if self._last_rebal_date is not None:
            if (d - self._last_rebal_date).days < REBAL_DAYS:
                return
        self._last_rebal_date = d
        raw = self._factor_store[d]
        if len(raw) < TOP_K:
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
        # Negate to pick the WORST composite.
        scores = {syms[i]: -float(composite[i]) for i in range(len(syms))}
        try:
            self.rebalance_to_topn(
                scores=scores, top_n=TOP_K, weight_mode="equal",
                long_only=True, liquidate_unmentioned=True,
            )
        except Exception:
            pass


class BuyHoldTopK(Strategy):
    """Static top-20 by composite on day 1, hold forever."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._initial_done = False

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
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

    def on_cross_section(self, trading_date, timestamp) -> None:
        if self._initial_done:
            return
        d = trading_date
        if d not in self._factor_store:
            return
        raw = self._factor_store[d]
        if len(raw) < TOP_K:
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
                scores=scores, top_n=TOP_K, weight_mode="equal",
                long_only=True, liquidate_unmentioned=False,
            )
            self._initial_done = True
        except Exception:
            pass


class BuyHoldRandomTopK(Strategy):
    """Static random-20, hold forever."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initial_done = False
        rng = random.Random(42)

    def on_cross_section(self, trading_date, timestamp) -> None:
        if self._initial_done:
            return
        try:
            snap_map = self.get_instruments()
        except Exception:
            return
        syms = [s for s in snap_map.keys() if s != "BENCHMARK"]
        if len(syms) < TOP_K:
            return
        rng = random.Random(42)
        chosen = rng.sample(syms, TOP_K)
        scores = {s: float(i) for i, s in enumerate(chosen)}
        try:
            self.rebalance_to_topn(
                scores=scores, top_n=TOP_K, weight_mode="equal",
                long_only=True, liquidate_unmentioned=False,
            )
            self._initial_done = True
        except Exception:
            pass


def build_data():
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .collect()
    )
    bd = bull_date_set()
    bd_min, bd_max = min(bd), max(bd)
    df = df.filter((pl.col("trade_date") >= bd_min) & (pl.col("trade_date") <= bd_max))
    df = df.drop_nulls(subset=BULL_FACTORS)
    stocks = sorted(df["ts_code"].unique().to_list())
    data_dict = {}
    for code in stocks:
        pdf = df.filter(pl.col("ts_code") == code).sort("trade_date").to_pandas()
        pdf["date"] = pd.to_datetime(pdf["trade_date"]).dt.tz_localize("Asia/Shanghai")
        pdf = pdf.set_index("date").rename_axis("date").drop(columns=["trade_date"])
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        data_dict[code] = pdf
    return data_dict


def run_backtest(strategy_cls, data_dict, label):
    print(f"\n=== {label} ===")
    result = aq.run_backtest(
        data=data_dict,
        strategy=strategy_cls,
        initial_cash=INITIAL_CASH,
        commission_rate=COST_BPS_PER_SIDE / 10_000,
        slippage=0.0,
        t_plus_one=False,
        fill_policy=aq.NextOpen(),
        lot_size=100,
    )
    m = result.metrics_df
    total_pnl = float(m.loc["total_pnl", "value"])
    unrealized = float(m.loc["unrealized_pnl", "value"])
    closed_pnl = total_pnl - unrealized
    closed_ret = closed_pnl / INITIAL_CASH * 100
    metrics = {
        "label": label,
        "total_pnl": total_pnl,
        "unrealized": unrealized,
        "closed_only_ret_pct": closed_ret,
        "sharpe": float(m.loc["sharpe_ratio", "value"]),
        "sortino": float(m.loc["sortino_ratio", "value"]),
        "profit_factor": float(m.loc["profit_factor", "value"]),
        "win_rate": float(m.loc["win_rate", "value"]),
        "max_drawdown_pct": float(m.loc["max_drawdown_pct", "value"]),
        "trades": int(m.loc["closed_trade_count", "value"]),
        "total_commission": float(m.loc["total_commission", "value"]),
    }
    print(f"  closed-only return: {closed_ret:+.2f}%  "
          f"sharpe={metrics['sharpe']:.3f}  PF={metrics['profit_factor']:.3f}  "
          f"win={metrics['win_rate']:.2%}  MDD={metrics['max_drawdown_pct']:.1f}%")
    print(f"  total_commission: {metrics['total_commission']:,.0f}")
    return metrics


def main() -> int:
    print("Building bull-leg data (2513 days × 351 stocks)…")
    data_dict = build_data()
    print(f"  stocks: {len(data_dict)}")

    results = []
    results.append(run_backtest(BullCompositeTopK, data_dict, "Bull composite top-20 (PROPOSED)"))
    results.append(run_backtest(ReverseTopK, data_dict, "Reverse top-20 (worst composite)"))
    results.append(run_backtest(BuyHoldTopK, data_dict, "Buy-hold static top-20"))
    results.append(run_backtest(BuyHoldRandomTopK, data_dict, "Buy-hold random-20 (seed=42)"))

    print("\n=== Head-to-head ===")
    print(f"{'strategy':<35}  {'ret%':>9}  {'sharpe':>7}  {'PF':>6}  {'win%':>6}  {'MDD%':>6}  {'trades':>6}")
    for r in results:
        print(
            f"  {r['label']:<33}  {r['closed_only_ret_pct']:+9.2f}  "
            f"{r['sharpe']:>+7.3f}  {r['profit_factor']:>6.3f}  "
            f"{r['win_rate']*100:>6.2f}  {r['max_drawdown_pct']:>6.1f}  "
            f"{r['trades']:>6}"
        )

    out_path = OUT / "baseline_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nartefact: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())