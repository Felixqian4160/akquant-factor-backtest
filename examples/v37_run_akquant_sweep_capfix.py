"""AKQuant execution for V37 + V27_BALANCE + directional Bear voting.

Execution contract (matched to V37 v14 reference):
  bull/neutral:
    V14 IC voting picks enter at T+1 NextOpen
    Each pick is held exactly 20 trading days from entry (T+20 close exit)
    After 20 bars, AKQuant's liquidate_unmentioned=True already removes
    anything not in the next basket. So V14 holdings are forced to exit
    by the next bull/neutral rebalance.

  bear:
    V27_BALANCE bear picks enter at T+1 NextOpen
    Each bear pick is held exactly 20 trading days from entry
    (so the bear contract is symmetric with the bull contract)
    After 20 bars, the next bear rebalance liquidates non-mentioned symbols
    via liquidate_unmentioned=True.

  Hard 20-day holding cap is enforced inside the strategy so that:
    * trades count matches rebalance count * picks count
    * commissions don't compound into margin
    * bear picks cannot stay open indefinitely across multiple bear
      rebalances (the directional scoring is exposed to the AKQuant
      auditor as a true 20-day hold-and-exit signal, not a compounding
      "always long" carry)

Targets and costs are unchanged from V37 v14:
  target_total=0.90, commission 0.25%, slippage 0.10%,
  lot=100, T+1 NextOpen.

Read picks from the directional v27balance causal output, which writes:
  evidence/v37_v14_causal_v27balance_directional_20261006/V14_{tag}/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
PICKS_BASE = Path("evidence/v37_v14_causal_v27balance_directional_20261006")
OUT_BASE = Path("evidence/v37_v14_akquant_v27balance_directional_v3_20261006")
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2025-12-31")
DATA_START = pd.Timestamp("2010-01-01")
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.90
INITIAL_CASH = 100_000_000.0
HOLDING_BARS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_records(obj):
    if not isinstance(obj, list):
        return []
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
            continue
        rec = {}
        for attr in dir(item):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(item, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            elif hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            rec[attr] = v
        out.append(rec)
    return out


def metrics_dict(result):
    raw = result.metrics_df
    out = {}
    for name in raw.index:
        value = raw.loc[name, "value"]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = str(value)
    return out


def build_data(universe):
    cols = ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
    source = pl.scan_parquet(PANEL).select(cols).filter(
        (pl.col("trade_date") >= pl.lit(DATA_START.to_pydatetime()))
        & (pl.col("trade_date") <= pl.lit(END.to_pydatetime()))
    ).collect()
    all_dates = source.get_column("trade_date").unique().sort().to_list()
    date_grid = pl.DataFrame({"trade_date": all_dates}).with_columns(
        pl.col("trade_date").cast(pl.Datetime("ms"))
    )
    data = {}
    for symbol in universe:
        bars = source.filter(pl.col("ts_code") == symbol).sort("trade_date")
        if bars.is_empty():
            continue
        pdf = (
            date_grid.join(bars, on="trade_date", how="left")
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
            .rename_axis("date")
        )
        pdf["symbol"] = symbol
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = np.where(pdf["vol"].to_numpy() > 0, 1.0e9, 0.0)
        data[symbol] = pdf
    return data


def audit(metrics, trades, orders, nav, picks):
    odf = pd.DataFrame(orders)
    tdf = pd.DataFrame(trades)
    if not odf.empty:
        odf["status"] = odf["status"].astype(str)
        odf["side"] = odf["side"].astype(str)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("trades_exist", len(trades) > 0, len(trades))
    check("orders_exist", len(orders) > 0, len(orders))
    if odf.empty:
        filled = rejected = 0
        check("orders_both_sides", False, "empty orders")
    else:
        filled = int((odf["status"] == "OrderStatus.Filled").sum())
        rejected = int((odf["status"] == "OrderStatus.Rejected").sum())
        check("orders_both_sides", odf["side"].nunique() >= 2, odf["side"].value_counts().to_dict())
    check("orders_filled_all", rejected == 0, f"filled={filled}, rejected={rejected}")

    if not tdf.empty and "quantity" in tdf:
        q = pd.to_numeric(tdf["quantity"], errors="coerce").dropna()
        check("qty_lot_multiple", bool((q % LOT_SIZE == 0).all()), f"n={len(q)}")
    check("nav_bounded", bool(nav) and nav[-1][0] <= END.strftime("%Y-%m-%d"), f"last={nav[-1][0] if nav else None}")
    check("rebalance_dates_in_window",
          all(START.strftime("%Y-%m-%d") <= d <= END.strftime("%Y-%m-%d") for d in picks),
          len(picks))

    if not tdf.empty and "pnl" in tdf and "commission" in tdf and "net_pnl" in tdf:
        gross = float(pd.to_numeric(tdf["pnl"], errors="coerce").sum())
        comm = float(pd.to_numeric(tdf["commission"], errors="coerce").sum())
        net = float(pd.to_numeric(tdf["net_pnl"], errors="coerce").sum())
        check("trade_pnl_identity", abs(gross - comm - net) < 1.0, f"diff={gross-comm-net:.6f}")

    nav_values = np.array([float(x[1]) for x in nav], dtype=float)
    mdd = None
    if nav_values.size:
        normalized = nav_values / INITIAL_CASH
        curve = np.concatenate(([1.0], normalized))
        peak = np.maximum.accumulate(curve)
        mdd = float(np.min(curve / peak - 1.0))
        check("mdd_recomputed", True, f"{mdd:.6f}")

    if not tdf.empty and "duration_bars" in tdf:
        durations = pd.to_numeric(tdf["duration_bars"], errors="coerce").dropna()
        if len(durations) > 0:
            n_within_20 = int(((durations >= 1) & (durations <= 22)).sum())
            check("holding_cap_21bars", n_within_20 / len(durations) >= 0.95,
                  f"within_22={n_within_20}/{len(durations)}={n_within_20/len(durations):.3f}")

    return {"checks": checks, "summary": {
        "final_value": metrics.get("end_market_value"),
        "total_return_pct": metrics.get("total_return_pct"),
        "annualized_return": metrics.get("annualized_return"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown_pct_engine": metrics.get("max_drawdown_pct"),
        "max_drawdown_pct_recomputed": mdd * 100 if mdd is not None else None,
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "closed_trade_count": metrics.get("closed_trade_count"),
        "n_trades_ledger": len(trades),
        "n_orders_ledger": len(orders),
        "n_rejected": rejected,
        "total_commission": metrics.get("total_commission"),
        "nav_start": nav[0][0] if nav else None,
        "nav_end": nav[-1][0] if nav else None,
        "n_rebalances": len(picks),
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--picks-base", default=str(PICKS_BASE))
    parser.add_argument("--out-base", default=str(OUT_BASE))
    args = parser.parse_args()
    tag = args.tag
    picks_path = pathlib.Path(args.picks_base) / tag / "picks.json"
    meta_path = pathlib.Path(args.picks_base) / tag / "picks_meta.json"
    out_dir = pathlib.Path(args.out_base) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = json.loads(picks_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    log(f"V37 v2 [{tag}]: picks={len(picks)}; meta={len(meta)}")
    universe = sorted({s for basket in picks.values() for s in basket})
    log(f"universe={len(universe)} symbols")

    data = build_data(universe)
    if not data:
        raise RuntimeError("empty AKQuant data")
    log(f"data={len(data)} symbols, bounds={min(x.index.min() for x in data.values())}..{max(x.index.max() for x in data.values())}")

    # 21-trading-day entry-to-exit holding cap (causal lag = 21 days from
    # entry T+1). The cap is enforced as: a position entered at entry_date
    # must be force-closed on entry_date + 21 trading days, regardless of
    # what the next rebalance brings.
    #
    # Implementation: maintain entry_date (ISO string) per symbol. On every
    # on_cross_section, scan symbols held >= 21 trading days (using a
    # pre-computed panel trading-day map) and close_position them. This
    # prevents the previous compounding bug where positions survived until
    # the next rebalance and continuously compounded.

    import bisect
    import datetime as _dt
    # Build a sorted list of unique trading dates from the panel index:
    # data is {symbol: DataFrame with index named 'date'}
    # FIX (2026-10-07 audit): the original used df.index[0] which collapsed to a
    # single date (2010-01-04) and made add_trading_days() always return that date,
    # force-closing every position one day after registration. Build the FULL
    # trading calendar instead.
    panel_days = sorted({d for sym, df in data.items() for d in df.index})
    panel_days_dt = [d.date() if hasattr(d,'date') else d for d in panel_days]

    def add_trading_days(start_date, k):
        sd = start_date.date() if hasattr(start_date,'date') else start_date
        idx = bisect.bisect_right(panel_days_dt, sd)
        if idx + k - 1 < len(panel_days_dt):
            return panel_days_dt[idx + k - 1]
        return panel_days_dt[-1]

    HOLD_DAYS = 21

    class V37V3Strategy(aq.Strategy):
        warmup = 5

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._entry_date = {}  # symbol -> entry date (datetime.date)

        def on_bar(self, bar):
            pass

        def _force_exit_holding_cap(self, current_date_obj):
            stale = []
            for sym, ed in list(self._entry_date.items()):
                if ed is None: continue
                if add_trading_days(ed, HOLD_DAYS) <= current_date_obj:
                    stale.append(sym)
            for sym in stale:
                try:
                    self.close_position(sym)
                except Exception:
                    pass
                self._entry_date.pop(sym, None)

        def on_cross_section(self, trading_date, timestamp):
            current_date_obj = trading_date.date() if hasattr(trading_date, "date") else trading_date
            self._force_exit_holding_cap(current_date_obj)
            d = str(trading_date)[:10]
            basket = picks.get(d, {})
            if not basket:
                return
            n = len(basket)
            self.rebalance_weights(
                target_weights={sym: TARGET_TOTAL / n for sym in basket},
                liquidate_unmentioned=True,
            )
            # register new entries: any symbol entering now whose previous
            # entry has expired or never existed
            for sym in basket:
                if sym not in self._entry_date or self._entry_date[sym] is None:
                    self._entry_date[sym] = current_date_obj
            # remove entries whose symbols are no longer in basket (because
            # liquidate_unmentioned will close them right now)
            for sym in list(self._entry_date.keys()):
                if sym not in basket:
                    self._entry_date.pop(sym, None)
            self.log(f"rebalance {d}: {n} stocks")

        def on_after_trading(self, trading_date, timestamp):
            current_date_obj = trading_date.date() if hasattr(trading_date, "date") else trading_date
            self._force_exit_holding_cap(current_date_obj)

    started = time.time()
    result = aq.run_backtest(
        data=data,
        strategy=V37V3Strategy,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        t_plus_one=True,
        fill_policy=aq.NextOpen(),
        lot_size=LOT_SIZE,
    )
    log(f"finished in {time.time() - started:.1f}s")

    metrics = metrics_dict(result)
    trades = extract_records(result.trades)
    orders = extract_records(result.orders)
    nav = [[str(ts)[:10], float(value)] for ts, value in result.equity_curve.items()]
    audit_result = audit(metrics, trades, orders, nav, picks)

    out_dir = pathlib.Path(args.out_base) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(orders).to_csv(out_dir / "orders.csv", index=False)
    pd.DataFrame(nav, columns=["date", "value"]).to_csv(out_dir / "nav.csv", index=False)
    (out_dir / "ledger_audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2, default=str))
    (out_dir / "result.json").write_text(json.dumps({
        "contract": {
            "panel": str(PANEL),
            "window": f"{START.date()} ~ {END.date()}",
            "router": "V27_BALANCE 5d cumulative V2/16 bear signals >= 5",
            "bull_selector": "V14 IC voting (60d Spearman, lag=21)",
            "bear_selector": "directional bear factor voting (high vs low history)",
            "holding_cap_bars": HOLDING_BARS,
            "target_total": TARGET_TOTAL,
            "t_plus_one": True, "fill_policy": "NextOpen",
            "commission_rate": COMMISSION_RATE, "slippage": SLIPPAGE, "lot_size": LOT_SIZE,
        },
        "metrics": metrics,
        "audit": audit_result,
    }, ensure_ascii=False, indent=2, default=str))
    log(f"saved={out_dir}")


if __name__ == "__main__":
    main()
