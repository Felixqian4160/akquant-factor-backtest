"""v1 Stage 4: bull_composite + kill-switch (portfolio MDD > 8% -> flat 20d).

Single-variable change vs Stage 3: ADD kill-switch only.
- Stage 3 factors / top_k / rebal / commission / strategy class unchanged.
- ADD: portfolio MDD tracking in on_cross_section.
  If portfolio equity drops > 8% from peak → set flat_kill_until = today + 20d.
  During kill window: don't trade (rebalance skipped).

The kill-switch is implemented using strategy.get_account()['total_equity'].
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

OUT_DIR = ROOT / "evidence" / f"factor_research_workflow_v1_stage4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]
TOP_K = 20
REBAL_DAYS = 20
INITIAL_CASH = 1_000_000.0
COMMISSION_BPS_PER_SIDE = 40
KILL_DD_THRESHOLD = 0.08  # 8% drawdown -> flat
KILL_COOLDOWN_DAYS = 20


class BullCompositeTopKKillSwitch(Strategy):
    """Stage 4: same as Stage 3 BullCompositeTopK but with portfolio kill-switch."""

    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        self._daily_complete: set = set()
        # kill-switch state
        self._equity_peak: float = float(INITIAL_CASH)
        self._kill_until: date | None = None
        self._kill_triggered_count: int = 0

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

        # 1. Update equity peak & kill-switch state.
        acct = self.get_account()
        equity = float(acct.get("equity", INITIAL_CASH))
        if equity > self._equity_peak:
            self._equity_peak = equity
        dd = (equity / self._equity_peak) - 1.0 if self._equity_peak > 0 else 0.0
        if dd <= -KILL_DD_THRESHOLD:
            self._kill_until = d + timedelta(days=KILL_COOLDOWN_DAYS)
            self._kill_triggered_count += 1

        # 2. Skip rebalance if in kill window.
        if self._kill_until is not None and d < self._kill_until:
            # Flatten any open positions.
            try:
                self.close_position()
            except Exception:
                pass
            return

        if d not in self._factor_store:
            return

        # 3. Rebalance pacing.
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


def build_data_dict(leg: dict) -> tuple[dict, int, int]:
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
    print("== v1 Stage 4: bull_composite + kill-switch (DD>8%->flat 20d) ==")
    print(f"factors: {BULL_FACTORS}")
    print(f"top_k={TOP_K}, rebal={REBAL_DAYS}, cost={COMMISSION_BPS_PER_SIDE}bps")
    print(f"kill: DD>{KILL_DD_THRESHOLD*100}% flat {KILL_COOLDOWN_DAYS}d")
    print(f"output: {OUT_DIR}")

    results = []
    for leg in BULL_LEGS:
        idx = leg["idx"]
        s = leg["start_date"]
        e = leg["end_date"]
        try:
            data_dict, n_rows, n_stocks = build_data_dict(leg)
        except Exception as exc:
            print(f"  leg#{idx:>2} {s}~{e}: build error: {type(exc).__name__}: {exc}")
            results.append({"leg_idx": idx, "regime": "bull", "error": str(exc)})
            continue

        if n_stocks < TOP_K:
            print(f"  leg#{idx:>2} {s}~{e}: too few stocks ({n_stocks})")
            results.append({"leg_idx": idx, "regime": "bull", "error": "too_few_stocks"})
            continue

        t0 = time.time()
        try:
            result = aq.run_backtest(
                data=data_dict,
                strategy=BullCompositeTopKKillSwitch,
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
                "leg_idx": idx, "regime": "bull",
                "start": s, "end": e, "days": leg["days"],
                "n_rows": n_rows, "n_stocks": n_stocks,
                "closed_only_return_pct": round(closed_ret, 4),
                "total_return_pct": round(total_ret, 4),
                "unrealized_pnl": round(upnl, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(mdd, 2),
                "win_rate": round(wr, 2),
                "closed_trade_count": trades,
                "elapsed_sec": round(elapsed, 1),
            }
            results.append(rec)
            ann = ((1+closed_ret/100)**(365/leg["days"])-1)*100 if closed_ret > -100 else 0
            target_pass = (
                sharpe >= 1.2 and ann >= 12.0 and mdd <= 12.0
            )
            marker = '🎯 TARGET' if target_pass else ('✅' if closed_ret > 0 else '❌')
            print(f"  {marker} leg#{idx:>2} {s}~{e} closed={closed_ret:+7.2f}% ann={ann:+6.1f}% "
                  f"sharpe={sharpe:+.3f} mdd={mdd:5.1f}% trades={trades:>4} ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"  leg#{idx:>2} {s}~{e}: BACKTEST ERROR: {type(exc).__name__}: {exc}")
            results.append({"leg_idx": idx, "regime": "bull", "error": str(exc)})

    out_path = OUT_DIR / "per_leg_bull_killswitch.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    valid = [r for r in results if "closed_only_return_pct" in r]
    if valid:
        n_pos = sum(1 for r in valid if r["closed_only_return_pct"] > 0)
        n_target = sum(
            1 for r in valid
            if r["sharpe_ratio"] >= 1.2
            and ((1+r["closed_only_return_pct"]/100)**(365/r["days"])-1)*100 >= 12
            and r["max_drawdown_pct"] <= 12
        )
        closed_rets = [r["closed_only_return_pct"] for r in valid]
        sharpes = [r["sharpe_ratio"] for r in valid]
        mdds = [r["max_drawdown_pct"] for r in valid]
        import statistics
        print()
        print(f"== Stage 4 summary ({len(valid)}/{len(BULL_LEGS)} valid) ==")
        print(f"  positive closed_only: {n_pos}/{len(valid)} ({n_pos/len(valid)*100:.0f}%)")
        print(f"  MEET ALL TARGETS (ann>=12% / Sharpe>=1.2 / MDD<=12%): {n_target}/{len(valid)}")
        print(f"  mean closed_only: {statistics.mean(closed_rets):+.2f}%")
        print(f"  median closed_only: {statistics.median(closed_rets):+.2f}%")
        print(f"  mean sharpe: {statistics.mean(sharpes):+.3f}")
        print(f"  mean MDD: {statistics.mean(mdds):.2f}%")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())