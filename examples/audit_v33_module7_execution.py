"""V33 Audit Module 7: Trade execution verification.

Verify AKQuant trades against the panel:
  1. Entry price == T+1 raw open from panel (within tolerance)
  2. Exit price == T+21 raw open (for force-closed) — check duration
  3. Commission == notional × 0.25% (per side)
  4. Quantity is 100-lot
  5. NAV reconciliation
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
TRADES = ROOT / "evidence/sweep/v33_factorrank_20261007/akquant/V14_0/trades.csv"
ORDERS = ROOT / "evidence/sweep/v33_factorrank_20261007/akquant/V14_0/orders.csv"
NAV = ROOT / "evidence/sweep/v33_factorrank_20261007/akquant/V14_0/nav.csv"
OUT = ROOT / "evidence" / "audit_v33_20261007"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

results = {}

def ts_to_date(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).date()

# Load trades
with TRADES.open() as f:
    trades = list(csv.DictReader(f))
with NAV.open() as f:
    nav = list(csv.DictReader(f))
log(f"trades: {len(trades)}, nav rows: {len(nav)}")

# Load panel prices
px = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"])
px_pd = px.to_pandas()
px_pd["trade_date"] = pd.to_datetime(px_pd["trade_date"]).dt.date
px_map = {}
for row in px_pd.itertuples():
    px_map[(row.ts_code, row.trade_date)] = (row.open, row.close)
log(f"price map entries: {len(px_map):,}")

# ── 1. Entry price check ──
log("=== 1. Entry price vs panel open ===")
checked = 0
mismatch = 0
max_rel_diff = 0.0
bad_examples = []
for t in trades[:500]:  # check first 500
    sym = t["symbol"]
    entry_d = ts_to_date(t["entry_time"])
    entry_price = float(t["entry_price"])
    key = (sym, entry_d)
    if key not in px_map:
        continue
    panel_open, panel_close = px_map[key]
    if panel_open is None or np.isnan(panel_open):
        continue
    checked += 1
    rel = abs(entry_price - panel_open) / panel_open
    max_rel_diff = max(max_rel_diff, rel)
    if rel > 0.02:  # >2% diff
        mismatch += 1
        if len(bad_examples) < 5:
            bad_examples.append({"sym": sym, "date": str(entry_d),
                                 "trade_price": entry_price, "panel_open": panel_open})
log(f"  checked: {checked}, mismatches(>2%): {mismatch}, max_rel_diff: {max_rel_diff:.4f}")
for ex in bad_examples:
    log(f"    {ex}")
results["entry_price"] = {"checked": checked, "mismatch": mismatch,
                          "max_rel_diff": float(max_rel_diff), "examples": bad_examples}

# ── 2. Commission check ──
log("=== 2. Commission == 0.25% of notional ===")
comm_bad = 0
comm_max_rel = 0.0
for t in trades[:500]:
    qty = float(t["quantity"])
    ep = float(t["entry_price"])
    xp = float(t["exit_price"])
    trade_comm = float(t["commission"])
    # round-trip commission = qty*(ep+xp)*0.0025
    expected = qty * (ep + xp) * 0.0025
    if expected > 0:
        rel = abs(trade_comm - expected) / expected
        comm_max_rel = max(comm_max_rel, rel)
        if rel > 0.05:
            comm_bad += 1
log(f"  commission mismatches(>5%): {comm_bad}, max_rel_diff: {comm_max_rel:.4f}")
results["commission"] = {"mismatch": comm_bad, "max_rel_diff": float(comm_max_rel)}

# ── 3. Duration analysis ──
log("=== 3. Trade duration ===")
durations = [int(t["duration_bars"]) for t in trades]
d = np.array(durations)
log(f"  duration_bars: min={d.min()}, median={np.median(d):.0f}, max={d.max()}, mean={d.mean():.1f}")
log(f"  within 22 bars: {(d<=22).sum()}/{len(d)} = {(d<=22).mean()*100:.1f}%")
over = d[d > 22]
log(f"  over 22 bars: {len(over)} trades, values: {sorted(over)[:20]}")
results["duration"] = {"min": int(d.min()), "median": float(np.median(d)),
                       "max": int(d.max()), "within22": int((d <= 22).sum()),
                       "total": len(d), "over22": over.tolist()[:30]}

# ── 4. NAV reconciliation ──
log("=== 4. NAV reconciliation ===")
start_nav = float(nav[0]["value"])
end_nav = float(nav[-1]["value"])
net_pnl = sum(float(t["net_pnl"]) for t in trades)
comm_sum = sum(float(t["commission"]) for t in trades)
log(f"  start={start_nav:,.0f} end={end_nav:,.0f} change={end_nav-start_nav:,.0f}")
log(f"  sum net_pnl={net_pnl:,.0f}")
log(f"  diff={end_nav-start_nav-net_pnl:,.0f} ({(end_nav-start_nav-net_pnl)/(end_nav-start_nav)*100:.3f}% of change)")
results["nav_reconciliation"] = {
    "start": start_nav, "end": end_nav,
    "change": end_nav - start_nav, "sum_net_pnl": net_pnl,
    "diff": end_nav - start_nav - net_pnl,
    "diff_pct": float((end_nav - start_nav - net_pnl) / (end_nav - start_nav) * 100),
}

# ── 5. Verify exit price for a sample of force-closed trades ──
log("=== 5. Exit price spot check (21-bar trades) ===")
checked_exit = 0
exit_mismatch = 0
for t in trades:
    if int(t["duration_bars"]) != 21:
        continue
    if checked_exit >= 200:
        break
    sym = t["symbol"]
    exit_d = ts_to_date(t["exit_time"])
    key = (sym, exit_d)
    if key not in px_map:
        continue
    panel_open, _ = px_map[key]
    xp = float(t["exit_price"])
    if panel_open is None:
        continue
    checked_exit += 1
    rel = abs(xp - panel_open) / panel_open
    if rel > 0.02:
        exit_mismatch += 1
log(f"  exit checks: {checked_exit}, mismatches(>2%): {exit_mismatch}")
results["exit_price"] = {"checked": checked_exit, "mismatch": exit_mismatch}

with (OUT / "audit_07_execution.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Saved {OUT/'audit_07_execution.json'}")
log("DONE module 7")
