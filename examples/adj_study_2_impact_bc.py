"""Adjustment Impact Study — Layer B (fwd returns) + Layer C (execution PnL)

Layer B: _fwd_net raw vs adjusted over the full panel (2010-2025)
Layer C: actual trade ledger — per-trade raw return vs adjusted return
"""
import csv, json, pathlib, time
from datetime import datetime, timezone, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
ADJ = ROOT / "evidence" / "audit_v33_20261007" / "adj_study" / "adj_prices_354.parquet"
OUT = ROOT / "evidence" / "audit_v33_20261007" / "adj_study"
LEDGER = ROOT / "evidence/sweep/v33_factorrank_lb20_bothfix_20261007/akquant/V14_0"
CST = timezone(timedelta(hours=8))

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════
# Layer B: _fwd_net impact
# ══════════════════════════════════════════════════
log("=== Layer B: fwd_net raw vs adjusted ===")
df = pl.read_parquet(ADJ)
df = df.sort(["ts_code", "trade_date"])

df = df.with_columns([
    pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
    pl.col("close").shift(-21).over("ts_code").alias("_close_t21"),
    pl.col("adj_open").shift(-1).over("ts_code").alias("_aopen_t1"),
    pl.col("adj_close").shift(-21).over("ts_code").alias("_aclose_t21"),
]).with_columns([
    (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - 0.005).alias("fwd_raw"),
    (pl.col("_aclose_t21") / pl.col("_aopen_t1") - 1.0 - 0.005).alias("fwd_adj"),
])

# restrict to strategy window
w = df.filter(
    (pl.col("trade_date") >= pl.datetime(2010, 1, 1)) &
    (pl.col("trade_date") <= pl.datetime(2025, 12, 31))
).filter(pl.col("fwd_raw").is_not_null() & pl.col("fwd_adj").is_not_null())

w = w.with_columns((pl.col("fwd_adj") - pl.col("fwd_raw")).alias("fwd_diff"))
diffs = w["fwd_diff"].to_numpy()
log(f"rows in window: {len(diffs):,}")

stats = {
    "n_rows": int(len(diffs)),
    "mean_diff": float(np.mean(diffs)),
    "median_diff": float(np.median(diffs)),
    "p05": float(np.percentile(diffs, 5)),
    "p95": float(np.percentile(diffs, 95)),
    "pct_abs_gt_0.2pct": float((np.abs(diffs) > 0.002).mean() * 100),
    "pct_abs_gt_1pct": float((np.abs(diffs) > 0.01).mean() * 100),
    "pct_positive": float((diffs > 0).mean() * 100),
}
log(f"fwd_adj - fwd_raw: mean={stats['mean_diff']*100:.3f}%, median={stats['median_diff']*100:.3f}%")
log(f"  |diff|>0.2%: {stats['pct_abs_gt_0.2pct']:.1f}%  |diff|>1%: {stats['pct_abs_gt_1pct']:.1f}%  positive: {stats['pct_positive']:.1f}%")

# by year
w2 = w.with_columns(pl.col("trade_date").dt.year().alias("y"))
by_year = w2.group_by("y").agg([
    pl.col("fwd_diff").mean().alias("mean_diff"),
    (pl.col("fwd_diff").abs() > 0.002).mean().alias("pct_gt02"),
]).sort("y")
log("by year:")
for row in by_year.iter_rows(named=True):
    log(f"  {row['y']}: mean_diff={row['mean_diff']*100:+.3f}%, pct(|d|>0.2%)={row['pct_gt02']*100:.1f}%")

# ══════════════════════════════════════════════════
# Layer C: execution impact on actual trades
# ══════════════════════════════════════════════════
log("\n=== Layer C: actual trade PnL raw vs adjusted ===")
with (LEDGER / "trades.csv").open() as f:
    trades = list(csv.DictReader(f))
log(f"trades: {len(trades)}")

def ts_to_date(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, tz=CST).date()

# price maps (date-level)
px = df.select(["ts_code", "trade_date", "open", "adj_open"]).to_pandas()
px["trade_date"] = pd.to_datetime(px["trade_date"]).dt.date
raw_map = {}
adj_map = {}
for r in px.itertuples():
    raw_map[(r.ts_code, r.trade_date)] = r.open
    adj_map[(r.ts_code, r.trade_date)] = r.adj_open

rows = []
affected = 0
total_impact = 0.0
total_notional = 0.0
per_year = Counter()
per_year_impact = {}
for t in trades:
    sym = t["symbol"]
    ed = ts_to_date(t["entry_time"])
    xd = ts_to_date(t["exit_time"])
    qty = float(t["quantity"])
    ro, rx = raw_map.get((sym, ed)), raw_map.get((sym, xd))
    ao, ax = adj_map.get((sym, ed)), adj_map.get((sym, xd))
    if None in (ro, rx, ao, ax) or min(ro, rx, ao, ax) <= 0:
        continue
    raw_ret = rx / ro - 1.0
    adj_ret = ax / ao - 1.0
    diff = adj_ret - raw_ret
    notional = qty * ro
    impact = diff * notional
    total_impact += impact
    total_notional += notional
    yr = ed.year
    per_year[yr] += 1
    per_year_impact[yr] = per_year_impact.get(yr, 0.0) + impact
    if abs(diff) > 0.002:
        affected += 1
    rows.append({"symbol": sym, "entry": str(ed), "exit": str(xd), "qty": qty,
                 "raw_ret": raw_ret, "adj_ret": adj_ret, "diff": diff, "impact": impact})

log(f"checked trades: {len(rows)}")
log(f"affected (|diff|>0.2%): {affected} ({affected/len(rows)*100:.1f}%)")
log(f"total PnL impact (adj - raw): +¥{total_impact:,.0f}")
log(f"total notional traded: ¥{total_notional:,.0f}")
log(f"impact as fraction of notional: {total_impact/total_notional*100:.3f}%")
log("by year (impact in ¥):")
for y in sorted(per_year_impact):
    log(f"  {y}: {per_year[y]} trades, impact=+¥{per_year_impact[y]:,.0f}")

results = {
    "layer_b": stats,
    "layer_b_by_year": {str(r["y"]): {"mean_diff": r["mean_diff"], "pct_gt02": r["pct_gt02"]}
                        for r in by_year.iter_rows(named=True)},
    "layer_c": {
        "n_trades": len(rows),
        "affected_trades": affected,
        "affected_pct": round(affected / len(rows) * 100, 2),
        "total_pnl_impact": total_impact,
        "total_notional": total_notional,
        "impact_pct_of_notional": total_impact / total_notional * 100,
        "by_year_impact": {str(y): per_year_impact[y] for y in sorted(per_year_impact)},
    },
}
(OUT / "impact_B_C.json").write_text(json.dumps(results, indent=2, default=str))

# save per-trade detail
pd.DataFrame(rows).to_csv(OUT / "trade_adjustment_impact.csv", index=False)
log("saved impact_B_C.json + trade_adjustment_impact.csv")
log("DONE")
