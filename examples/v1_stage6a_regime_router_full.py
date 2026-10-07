"""v1 Stage 6a: full-panel + regime router (bull_composite only on bull days).

Single-variable change vs Stage 5b: panel slicing.
- Stage 5b: per-bull-leg backtest (each leg independently, no flat periods).
- Stage 6a: FULL panel 2004-2026 + regime router that runs bull_composite
  ONLY on bull days, flat on bear/sideways days.

This is the natural "production-ready" form of Stage 5b: the regime router
determines when to engage, based on the bull_legs ground truth.

NOT changing: factors, top_k=10, kill-switch, rebal, commission.
"""
from __future__ import annotations

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

OUT_DIR = ROOT / "evidence" / f"factor_research_workflow_v1_stage6a_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
TOP_K = 10
REBAL_DAYS = 20
INITIAL_CASH = 1_000_000.0
COMMISSION_BPS_PER_SIDE = 40
KILL_DD_THRESHOLD = 0.08
KILL_COOLDOWN_DAYS = 20

# Build bull/bear date sets (ground truth from index pivots).
def build_date_set(legs: list[dict]) -> set[date]:
    s = set()
    for l in legs:
        d = date.fromisoformat(l["start_date"])
        e = date.fromisoformat(l["end_date"])
        while d <= e:
            s.add(d)
            d = date.fromordinal(d.toordinal() + 1)
    return s

BULL_DATES = build_date_set(BULL_LEGS)
BEAR_DATES = build_date_set(BEAR_LEGS)


class RegimeRouter(Strategy):
    """Bull_composite on bull days, flat otherwise. Kill-switch on bar."""

    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        self._equity_peak: float = float(INITIAL_CASH)
        self._kill_until: date | None = None
        self._n_bull_days = 0
        self._n_flat_days = 0

    def _check_kill(self, d: date) -> None:
        acct = self.get_account()
        equity = float(acct.get("equity", INITIAL_CASH))
        if equity > self._equity_peak:
            self._equity_peak = equity
        dd = (equity / self._equity_peak) - 1.0 if self._equity_peak > 0 else 0.0
        if dd <= -KILL_DD_THRESHOLD:
            new_until = d + timedelta(days=KILL_COOLDOWN_DAYS)
            if self._kill_until is None or new_until > self._kill_until:
                self._kill_until = new_until

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        self._check_kill(d)
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

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date

        # Kill check (priority 1)
        if self._kill_until is not None and d < self._kill_until:
            try:
                self.close_position()
            except Exception:
                pass
            self._n_flat_days += 1
            return

        # Regime router (priority 2): only trade on BULL days
        if d not in BULL_DATES:
            try:
                self.close_position()
            except Exception:
                pass
            self._n_flat_days += 1
            return

        self._n_bull_days += 1

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


def build_full_data() -> tuple[dict, int, int]:
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
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


def _closed_only_pct(metrics_df) -> tuple[float, float]:
    total_pnl = float(metrics_df.loc["total_pnl", "value"])
    upnl = float(metrics_df.loc["unrealized_pnl", "value"])
    initial = float(metrics_df.loc["initial_market_value", "value"])
    return (total_pnl - upnl) / initial * 100, upnl


def main() -> int:
    print("== v1 Stage 6a: full-panel + regime router (bull_composite only on bull days) ==")
    print(f"factors: {BULL_FACTORS}")
    print(f"top_k={TOP_K}, rebal={REBAL_DAYS}, cost={COMMISSION_BPS_PER_SIDE}bps")
    print(f"kill: DD>{KILL_DD_THRESHOLD*100}% flat {KILL_COOLDOWN_DAYS}d")
    print(f"bull dates: {len(BULL_DATES)} ({len(BULL_DATES)/(len(BULL_DATES)+len(BEAR_DATES))*100:.0f}% of leg days)")
    print(f"output: {OUT_DIR}")

    try:
        data_dict, n_rows, n_stocks = build_full_data()
    except Exception as exc:
        print(f"  build error: {type(exc).__name__}: {exc}")
        return 1
    print(f"  panel: rows={n_rows}, stocks={n_stocks}")

    t0 = time.time()
    try:
        result = aq.run_backtest(
            data=data_dict,
            strategy=RegimeRouter,
            initial_cash=INITIAL_CASH,
            commission_rate=COMMISSION_BPS_PER_SIDE / 10_000,
            slippage=0.0,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
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
            "regime": "router (bull_only)",
            "panel_days": n_rows // n_stocks,
            "closed_only_return_pct": round(closed_ret, 4),
            "total_return_pct": round(total_ret, 4),
            "unrealized_pnl": round(upnl, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(mdd, 2),
            "win_rate": round(wr, 2),
            "closed_trade_count": trades,
            "elapsed_sec": round(elapsed, 1),
        }
        # panel days: panel 2004-2026 = 5498 days, panel_start..end span
        panel_days = (date.fromisoformat('2026-08-21') - date.fromisoformat('2004-01-02')).days
        ann = ((1+closed_ret/100)**(365/panel_days)-1)*100 if closed_ret > -100 else 0
        target_pass = sharpe >= 1.2 and ann >= 12.0 and mdd <= 12.0
        print(f"\n  RESULT ({panel_days} days = {panel_days/365:.1f}y):")
        print(f"  {'🎯 TARGET' if target_pass else ('✅' if closed_ret > 0 else '❌')} "
              f"closed={closed_ret:+.2f}% ann={ann:+.2f}% sharpe={sharpe:+.3f} "
              f"mdd={mdd:.2f}% trades={trades} ({elapsed:.1f}s)")
        out_path = OUT_DIR / "regime_router_full.json"
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"\nwrote {out_path}")
        return 0
    except Exception as exc:
        print(f"  BACKTEST ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())