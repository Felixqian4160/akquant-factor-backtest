"""V33 Audit Module 7 (v2): Trade execution verification — CST timezone fix + order-level reconciliation.

Key context: AKQuant timestamps are nanoseconds; a "trading day" is stored as
16:00 UTC of the PREVIOUS calendar day (= 00:00 CST of the trade day).
So date conversion must use UTC+8 (CST).
"""
import csv
import json
import pathlib
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
BASE = ROOT / "evidence/sweep/v33_factorrank_20261007/akquant/V14_0"
OUT = ROOT / "evidence" / "audit_v33_20261007"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

CST = timezone(timedelta(hours=8))

def ts_to_date_cst(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, tz=CST).date()

results = {}

with (BASE / "trades.csv").open() as f:
    trades = list(csv.DictReader(f))
with (BASE / "orders.csv").open() as f:
    orders = list(csv.DictReader(f))
with (BASE / "nav.csv").open() as f:
    nav = list(csv.DictReader(f))

# price map
px = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"]).to_pandas()
px["trade_date"] = pd.to_datetime(px["trade_date"]).dt.date
px_map = {(r.ts_code, r.trade_date): (r.open, r.close) for r in px.itertuples()}

# ── 1. Entry price vs panel open (CST date) ──
log("=== 1. Entry price vs panel open (CST) ===")
checked = mismatch = 0
max_rel = 0.0
bad = []
for t in trades:
    sym = t["symbol"]
    entry_d = ts_to_date_cst(t["entry_time"])
    ep = float(t["entry_price"])
    key = (sym, entry_d)
    if key not in px_map:
        continue
    po, pc = px_map[key]
    if po is None or (isinstance(po, float) and np.isnan(po)):
        continue
    checked += 1
    rel = abs(ep - po) / po
    max_rel = max(max_rel, rel)
    if rel > 0.02:
        mismatch += 1
        if len(bad) < 8:
            bad.append({"sym": sym, "date": str(entry_d), "trade": ep, "panel_open": po, "rel": round(rel, 4)})
log(f"  checked={checked}, mismatch(>2%)={mismatch}, max_rel={max_rel:.4f}")
for x in bad:
    log(f"    {x}")

# Check the ones that mismatch: maybe price at T or T+2?
if bad:
    log("  Deep check first mismatched trade:")
    t0 = next(t for t in trades if t["symbol"] == bad[0]["sym"] and abs(float(t["entry_price"]) - bad[0]["trade"]) < 1e-9)
    ed = ts_to_date_cst(t0["entry_time"])
    log(f"    {bad[0]['sym']} entry_date(CST)={ed}")
    # check panel prices around
    for delta in [-3, -2, -1, 0, 1, 2]:
        dd = ed + timedelta(days=delta)
        k = (bad[0]["sym"], dd)
        if k in px_map:
            log(f"      {dd}: open={px_map[k][0]:.3f} close={px_map[k][1]:.3f}")

results["entry_price"] = {"checked": checked, "mismatch": mismatch, "max_rel": float(max_rel), "examples": bad}

# ── 2. Exit price vs panel (CST) ──
log("=== 2. Exit price vs panel (force-closed 21-bar) ===")
checked2 = mismatch2 = 0
bad2 = []
for t in trades:
    if int(t["duration_bars"]) != 21:
        continue
    sym = t["symbol"]
    exit_d = ts_to_date_cst(t["exit_time"])
    xp = float(t["exit_price"])
    key = (sym, exit_d)
    if key not in px_map:
        continue
    po, pc = px_map[key]
    if po is None or (isinstance(po, float) and np.isnan(po)):
        continue
    checked2 += 1
    rel = abs(xp - po) / po
    if rel > 0.02:
        mismatch2 += 1
        if len(bad2) < 8:
            bad2.append({"sym": sym, "date": str(exit_d), "trade": xp, "panel_open": po, "rel": round(rel, 4)})
log(f"  checked={checked2}, mismatch(>2%)={mismatch2}")
for x in bad2:
    log(f"    {x}")
results["exit_price"] = {"checked": checked2, "mismatch": mismatch2, "examples": bad2}

# ── 3. Order-level cash reconciliation ──
log("=== 3. Order-level cash reconciliation ===")
filled = [o for o in orders if o.get("status") == "Filled"]
log(f"  filled orders: {len(filled)}")
cash = 0.0
comm = 0.0
for o in filled:
    qty = float(o["filled_quantity"])
    price = float(o["average_filled_price"])
    c = float(o["commission"]) if o["commission"] else 0.0
    side = o["side"]
    if side == "OrderSide.Buy":
        cash -= qty * price
    else:
        cash += qty * price
    comm += c
log(f"  net cash flow from fills: {cash:,.2f}")
log(f"  total order commission: {comm:,.2f}")
nav_change = float(nav[-1]["value"]) - float(nav[0]["value"])
log(f"  NAV change: {nav_change:,.2f}")
log(f"  cash + NAV_start - NAV_end: {cash + float(nav[0]['value']) - float(nav[-1]['value']):,.2f}")
log(f"  (this residual = mark-to-market of open positions at end + other adjustments)")

# Check residual components
# remaining position value at end
pos = defaultdict(float)
for o in filled:
    q = float(o["filled_quantity"])
    if o["side"] == "OrderSide.Buy":
        pos[o["symbol"]] += q
    else:
        pos[o["symbol"]] -= q
open_pos = {s: q for s, q in pos.items() if abs(q) > 1e-9}
log(f"  open positions from orders: {len(open_pos)} symbols")
for s, q in list(open_pos.items())[:10]:
    log(f"    {s}: {q:,.0f}")

results["order_reconciliation"] = {
    "filled_orders": len(filled),
    "net_cash_flow": cash,
    "order_commission": comm,
    "nav_change": nav_change,
    "residual": cash + float(nav[0]["value"]) - float(nav[-1]["value"]),
    "open_positions": len(open_pos),
}

with (OUT / "audit_07_execution_v2.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Saved {OUT/'audit_07_execution_v2.json'}")
log("DONE module 7 v2")
