"""V34 IC-voting AKQuant execution: one bounded full-window run per tag.

Reads already audited picks from v34_ic_voting_2010_2025_correct/{A,B}.
Does not recompute factors or IC. Run sequentially:
  /usr/bin/python3.12 examples/v34_run_akquant.py --tag A
  /usr/bin/python3.12 examples/v34_run_akquant.py --tag B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
PICKS_ROOT = Path("evidence/v34_ic_voting_2010_2025_correct")
OUT_ROOT = Path("evidence/v34_ic_voting_2010_2025_akquant_v2")
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2025-12-31")
DATA_START = pd.Timestamp("2010-01-01")
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.90  # execution buffer; 0.99 caused margin rejects on long-window rotations
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
                value = getattr(item, attr)
            except Exception:
                continue
            if callable(value):
                continue
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = str(value)
            rec[attr] = value
        out.append(rec)
    return out


def metrics_dict(result) -> dict:
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


def build_data(universe: list[str]) -> dict:
    # AKQuant expects a bar at every market date for every supplied symbol.
    # Forward-fill only the execution price/mark on absent stock rows and
    # set volume to zero on those synthetic gap bars (so orders cannot fill).
    # Signal-time factor values remain untouched and are never carried forward.
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
    date_grid = pl.DataFrame({"trade_date": all_dates}).with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
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


def audit(tag: str, result, metrics: dict, trades: list, orders: list, nav: list, picks: dict) -> dict:
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
        check("orders_both_sides", False, "empty orders")
        filled = rejected = 0
    else:
        filled = int((odf["status"] == "OrderStatus.Filled").sum())
        rejected = int((odf["status"] == "OrderStatus.Rejected").sum())
        check("orders_both_sides", odf["side"].nunique() >= 2, odf["side"].value_counts().to_dict())
    check("orders_filled_all", rejected == 0, f"filled={filled}, rejected={rejected}")

    if not tdf.empty and "quantity" in tdf:
        q = pd.to_numeric(tdf["quantity"], errors="coerce").dropna()
        check("qty_lot_multiple", bool((q % LOT_SIZE == 0).all()), f"n={len(q)}")
    else:
        check("qty_lot_multiple", False, "no trade quantity")

    nav_last = nav[-1][0] if nav else None
    check("nav_bounded", nav_last is not None and nav_last <= END.strftime("%Y-%m-%d"), f"last={nav_last}")
    check("rebalance_dates_in_window", all(START.strftime("%Y-%m-%d") <= d <= END.strftime("%Y-%m-%d") for d in picks), len(picks))

    if not tdf.empty and "pnl" in tdf and "commission" in tdf and "net_pnl" in tdf:
        gross = float(pd.to_numeric(tdf["pnl"], errors="coerce").sum())
        comm = float(pd.to_numeric(tdf["commission"], errors="coerce").sum())
        net = float(pd.to_numeric(tdf["net_pnl"], errors="coerce").sum())
        check("trade_pnl_identity", abs(gross - comm - net) < 1.0, f"diff={gross-comm-net:.6f}")
    else:
        check("trade_pnl_identity", False, "missing PnL fields")

    # Rebuild MDD from the persisted NAV curve with initial NAV prepended.
    nav_values = np.array([float(x[1]) for x in nav], dtype=float)
    if nav_values.size:
        normalized = nav_values / INITIAL_CASH
        curve = np.concatenate(([1.0], normalized))
        peak = np.maximum.accumulate(curve)
        mdd = float(np.min(curve / peak - 1.0))
        check("mdd_recomputed", np.isfinite(mdd), f"{mdd:.6f}")
    else:
        mdd = None
        check("mdd_recomputed", False, "empty NAV")

    result = {
        "tag": tag,
        "checks": checks,
        "summary": {
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
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=["A", "B"], required=True)
    args = parser.parse_args()
    tag = args.tag
    pick_dir = PICKS_ROOT / tag
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = json.loads((pick_dir / "picks.json").read_text())
    contract = json.loads((pick_dir / "summary.json").read_text())
    if contract["factor_count_declared"] != 410 or len(picks) != contract["rebalance_dates_with_picks"]:
        raise RuntimeError("picks contract/artifact mismatch")
    universe = sorted({s for basket in picks.values() for s in basket})
    log(f"V34 AKQuant {tag}: picks={len(picks)}, universe={len(universe)}, panel=V33")
    data = build_data(universe)
    if not data:
        raise RuntimeError("empty AKQuant data")
    log(f"data={len(data)} symbols, bounds={min(x.index.min() for x in data.values())}..{max(x.index.max() for x in data.values())}")

    global _DAILY_PICKS, _REBAL_SET
    _DAILY_PICKS = picks
    _REBAL_SET = set(picks)

    class V34VoteStrategy(aq.Strategy):
        warmup = 5

        def on_bar(self, bar):
            pass

        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET:
                return
            basket = _DAILY_PICKS[d]
            n = len(basket)
            account = self.get_account()
            equity = float(account.get("equity", INITIAL_CASH))
            cash = float(account.get("cash", INITIAL_CASH))
            deploy = min(TARGET_TOTAL * equity, cash * 0.95)
            if deploy <= 0 or not np.isfinite(deploy):
                return
            price_map = {}
            for symbol in basket:
                pos = self.get_position(symbol)
                if pos != 0:
                    price = float(self.get_bar(symbol).get("close", 0.0)) if hasattr(self, "get_bar") else 0.0
                    if price > 0:
                        price_map[symbol] = price
            # close anything not selected to remove stale  positions
            for symbol in list(self.positions.keys()) if isinstance(getattr(self, "positions", {}), dict) else []:
                if symbol not in basket:
                    self.close_position(symbol)
            self.rebalance_positions(
                target_positions={symbol: deploy * TARGET_TOTAL / (n * (price_map.get(symbol, 0) or 1)) for symbol in basket},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
            self.log(f"rebalance {d}: {n} stocks, deploy=~{deploy:.0f}")

    log("Running AKQuant sequentially...")
    started = time.time()
    result = aq.run_backtest(
        data=data,
        strategy=V34VoteStrategy,
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
    audit_result = audit(tag, result, metrics, trades, orders, nav, picks)

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(orders).to_csv(out_dir / "orders.csv", index=False)
    pd.DataFrame(nav, columns=["date", "value"]).to_csv(out_dir / "nav.csv", index=False)
    (out_dir / "ledger_audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2, default=str))
    (out_dir / "result.json").write_text(json.dumps({
        "contract": contract,
        "execution": {
            "panel": str(PANEL), "data_start": str(DATA_START.date()), "data_end": str(END.date()),
            "commission_rate": COMMISSION_RATE, "slippage": SLIPPAGE, "lot_size": LOT_SIZE,
            "target_total": TARGET_TOTAL, "t_plus_one": True, "fill_policy": "NextOpen",
        },
        "metrics": metrics,
        "audit": audit_result,
    }, ensure_ascii=False, indent=2, default=str))
    log(f"saved={out_dir}")
    log("DONE")


if __name__ == "__main__":
    main()
