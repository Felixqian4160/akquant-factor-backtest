"""V37 AKQuant execution with V27_BALANCE router.

Reads picks from the isolated v27balance causal output and runs the unchanged
AKQuant execution contract: T+1 NextOpen, lot=100, 25 bps commission,
10 bps slippage, target_total=0.90.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
PICKS_BASE = Path("evidence/v37_v14_causal_v27balance_20261006")
OUT_BASE = Path("evidence/v37_v14_akquant_v27balance_20261006")
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2025-12-31")
DATA_START = pd.Timestamp("2010-01-01")
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.90
INITIAL_CASH = 100_000_000.0


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
    source = (
        pl.scan_parquet(PANEL)
        .select(cols)
        .filter(
            (pl.col("trade_date") >= pl.lit(DATA_START.to_pydatetime()))
            & (pl.col("trade_date") <= pl.lit(END.to_pydatetime()))
        )
        .collect()
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
    parser.add_argument("--tag", default="K10")
    args = parser.parse_args()
    tag = args.tag
    picks_path = PICKS_BASE / tag / "picks.json"
    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = json.loads(picks_path.read_text())
    log(f"V37 AKQuant [{tag}]: picks={len(picks)}")
    universe = sorted({s for basket in picks.values() for s in basket})
    log(f"universe={len(universe)} symbols")

    data = build_data(universe)
    if not data:
        raise RuntimeError("empty AKQuant data")
    log(f"data={len(data)} symbols, bounds={min(x.index.min() for x in data.values())}..{max(x.index.max() for x in data.values())}")

    global _DAILY_PICKS, _REBAL_SET
    _DAILY_PICKS = picks
    _REBAL_SET = set(picks)

    class V37Strategy(aq.Strategy):
        warmup = 5

        def on_bar(self, bar):
            pass

        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET:
                return
            basket = _DAILY_PICKS[d]
            n = len(basket)
            self.rebalance_weights(
                target_weights={symbol: TARGET_TOTAL / n for symbol in basket},
                liquidate_unmentioned=True,
            )
            self.log(f"rebalance {d}: {n} stocks")

    started = time.time()
    result = aq.run_backtest(
        data=data,
        strategy=V37Strategy,
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

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(orders).to_csv(out_dir / "orders.csv", index=False)
    pd.DataFrame(nav, columns=["date", "value"]).to_csv(out_dir / "nav.csv", index=False)
    (out_dir / "ledger_audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2, default=str))
    (out_dir / "result.json").write_text(json.dumps({
        "contract": {
            "panel": str(PANEL),
            "window": f"{START.date()} ~ {END.date()}",
            "ic_window": 20,
            "causal_lag": 20,
            "fwd_contract": "T+1 raw open -> T+21 close",
            "rebalance_step": 20,
            "top_k": 10, "k_nom": 10, "min_votes": 3,
            "min_stocks": 5, "max_stocks": 20,
        },
        "execution": {
            "data_start": str(DATA_START.date()), "data_end": str(END.date()),
            "commission_rate": COMMISSION_RATE, "slippage": SLIPPAGE,
            "lot_size": LOT_SIZE, "target_total": TARGET_TOTAL,
            "t_plus_one": True, "fill_policy": "NextOpen",
        },
        "metrics": metrics, "audit": audit_result,
    }, ensure_ascii=False, indent=2, default=str))
    log(f"saved={out_dir}")


if __name__ == "__main__":
    main()