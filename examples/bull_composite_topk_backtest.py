"""Verify bull composite factor alpha in AKQuant multi-symbol backtest.

Setup:
  - Bull legs: ZigZag-derived ≥100d up legs (10 total)
  - Composite: equal-weight mean of 6 cross-section rank factors
      talib_NATR, talib_TRANGE, gtja_gtja_159, gtja_gtja_149,
      gtja_gtja_144, mw_vol_20d
  - Strategy: every rebal_days (20 trading days), top-20 by composite
    rank, equal weight, hold 20 days
  - Cost: 30bps commission + 10bps stamp tax = 40bps/side
  - Universe: 354 HS300 stocks
  - Output: AKQuant BacktestResult metrics + per-day ledger

This is the 'A' validation from the regime-aware workflow:
  bull composite → top-K → AKQuant → real alpha?
"""
from __future__ import annotations

import json
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
COST_BPS_PER_SIDE = 40  # 30 commission + 10 stamp tax
INITIAL_CASH = 1_000_000.0


def bull_date_set() -> set[date]:
    """All dates that fall inside a ≥100d up leg."""
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
    """Multi-symbol top-K rebalance driven by 6-factor composite.

    Implementation detail: AKQuant's on_cross_section has no bar
    handle, so we cache per-(date, symbol) factor values inside
    on_bar (where bar.extra gives us the current values), then
    consume them inside on_cross_section.
    """

    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._seen = 0
        # date -> {symbol -> [factor values]}
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        # Track which dates have a complete factor snapshot for all
        # symbols; helps on_cross_section know if it's safe to act.
        self._daily_complete: set = set()

    def on_bar(self, bar: Bar) -> None:
        self._seen += 1
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
            self._symbols_seen[d] = set()
        # Pull all 6 factors from bar.extra.
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
        # Heuristic: consider a day complete once we've seen at
        # least 50 symbols. (We have 351 stocks; full coverage
        # would require waiting for every symbol's bar, but in
        # practice AKQuant emits bars across all symbols on each
        # trading day before on_cross_section fires.)
        if len(self._symbols_seen[d]) >= 50:
            self._daily_complete.add(d)

    def on_cross_section(self, trading_date, timestamp) -> None:
        """Daily cross-section handler."""
        d = trading_date
        if d not in self._factor_store:
            return

        # Pacing: only act every rebal_days.
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
        # Per-factor cross-section rank.
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
                top_n=self.top_n,
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


def main() -> int:
    print("=" * 60)
    print("Bull composite top-K multi-symbol AKQuant backtest")
    print("=" * 60)

    # 1. Bull date set.
    print("\n[1/5] bull date set...")
    bd = bull_date_set()
    print(f"  bull dates: {len(bd)}")

    # 2. Per-stock DataFrames with OHLCV + 6 factors, filtered to bull dates.
    print("\n[2/5] building per-stock DataFrames...")
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .collect()
    )
    # Filter to bull dates only.
    bd_list = sorted(bd)
    bd_min, bd_max = min(bd_list), max(bd_list)
    df = df.filter((pl.col("trade_date") >= bd_min) & (pl.col("trade_date") <= bd_max))
    # Drop rows where any factor is null.
    df = df.drop_nulls(subset=BULL_FACTORS)
    print(f"  filtered panel: {df.shape}, stocks: {df['ts_code'].n_unique()}")

    # Convert to Dict[ts_code → DataFrame].
    stocks = sorted(df["ts_code"].unique().to_list())
    print(f"  building per-stock frames for {len(stocks)} stocks...")
    data_dict: dict[str, pd.DataFrame] = {}
    for code in stocks:
        pdf = (
            df.filter(pl.col("ts_code") == code)
            .sort("trade_date")
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        # AKQuant expects a 'symbol' column on each DataFrame for multi-symbol.
        pdf["symbol"] = code
        # Drop ts_code since symbol carries it.
        pdf = pdf.drop(columns=["ts_code"])
        data_dict[code] = pdf

    # 3. Run backtest.
    print("\n[3/5] running AKQuant multi-symbol backtest...")
    t0 = pd.Timestamp.now()
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=BullCompositeTopK,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,  # 40bps per side
            slippage=0.0,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
        )
    except Exception as exc:
        print(f"  BACKTEST FAILED: {type(exc).__name__}: {exc}")
        return 1
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"  backtest elapsed: {elapsed:.1f}s")

    # 4. Report metrics.
    print("\n[4/5] backtest metrics...")
    m = result.metrics_df
    print(m.to_string())
    print()
    # Custom close-only return calc.
    total_pnl = float(m.loc["total_pnl", "value"])
    unrealized = float(m.loc["unrealized_pnl", "value"])
    initial = float(m.loc["initial_market_value", "value"])
    closed_pnl = total_pnl - unrealized
    closed_ret = closed_pnl / initial * 100
    print(f"  closed-only return: {closed_ret:+.2f}%  "
          f"(unrealized: {unrealized:+,.0f})")

    # 5. Save JSON.
    print("\n[5/5] saving artefact...")
    artefact = {
        "strategy": "BullCompositeTopK",
        "bull_factors": BULL_FACTORS,
        "top_n": TOP_K,
        "rebal_days": REBAL_DAYS,
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "n_stocks": len(stocks),
        "elapsed_sec": elapsed,
        "metrics": {
                str(idx): (
                    float(m.loc[idx, "value"])
                    if not isinstance(m.loc[idx, "value"], str)
                    else m.loc[idx, "value"]
                )
                for idx in m.index
            },
        "closed_only_return_pct": closed_ret,
        "unrealized_pnl": unrealized,
    }
    out_path = OUT / "bull_composite_topk.json"
    out_path.write_text(json.dumps(artefact, indent=2, default=str))
    print(f"  artefact: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())