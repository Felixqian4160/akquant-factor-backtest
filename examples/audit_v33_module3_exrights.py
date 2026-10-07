"""V33 Comprehensive Audit — Module 3: Ex-rights contamination impact quantification

Finding from Module 2:
  Panel uses RAW prices (adj_factor=1.0, adj_close==close).
  737 daily moves beyond ±21% (impossible under A-share daily limits) → ex-rights dates.

Impact:
  Any _fwd_net (21-day window) that crosses an ex-rights date shows artificial loss/gain.
  Question: how much does this contaminate (a) factor-return scores, (b) strategy trades?

Checks:
  1. Identify all extreme-move dates per stock (|ret| > 21%)
  2. Count how many _fwd_net windows cross extreme dates
  3. Quantify: for the ACTUAL trades of v33_factorrank (lb=20), did entry→exit span any extreme date?
  4. Estimate correction: replace extreme returns with median return → how do results change?
"""
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
OUT = ROOT / "evidence" / "audit_v33_20261007"
TRADES = ROOT / "evidence/sweep/v33_factorrank_20261007/akquant/V14_0/trades.csv"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

results = {}

# ── 1. Identify extreme-move (ex-rights candidate) dates ──
log("=== 1. Extreme move dates (|ret1| > 21%) ===")
px = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"]).sort(["ts_code", "trade_date"])
px = px.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret1")
)
extreme = px.filter((pl.col("_ret1").abs() > 0.21)).select(["trade_date", "ts_code", "_ret1"])
log(f"  extreme moves: {extreme.height}")
log(f"  extreme up (+>21%): {extreme.filter(pl.col('_ret1') > 0).height}")
log(f"  extreme down (-<-21%): {extreme.filter(pl.col('_ret1') < 0).height}")

# Count extreme dates by year
extreme_pd = extreme.to_pandas()
extreme_pd["year"] = pd.to_datetime(extreme_pd["trade_date"]).dt.year
by_year = extreme_pd.groupby("year").size()
log("  extreme moves by year:")
for y, n in by_year.items():
    log(f"    {y}: {n}")
results["extreme_by_year"] = {int(y): int(n) for y, n in by_year.items()}

# ── 2. Count _fwd_net windows crossing extreme dates ──
log("=== 2. _fwd_net window contamination ===")
# Use positional indices: for each stock, an fwd window at index t covers rows [t+1, t+21].
# An extreme move at row j contaminates all t in [j-21, j-1].
px2 = px.with_columns(pl.int_range(0, pl.len()).over("ts_code").alias("_idx"))
extreme_map = extreme.join(
    px2.select(["trade_date", "ts_code", "_idx"]), on=["trade_date", "ts_code"], how="left"
)
log(f"  extreme events with row index: {extreme_map.height}")

# per stock: list of extreme indices
from collections import defaultdict
ext_idx = defaultdict(list)
for row in extreme_map.iter_rows(named=True):
    if row["_idx"] is not None:
        ext_idx[row["ts_code"]].append(int(row["_idx"]))

# For each stock, total rows
row_counts = px2.group_by("ts_code").agg(pl.len().alias("n")).to_pandas()
total_rows = row_counts["n"].sum()

# Count contaminated (t) positions: t such that some extreme j exists with t < j <= t+21
contaminated_total = 0
for s, idxs in ext_idx.items():
    n = row_counts[row_counts["ts_code"] == s]["n"].iloc[0]
    mask = np.zeros(n, dtype=bool)
    for j in idxs:
        lo = max(0, j - 21)
        hi = min(n, j)
        mask[lo:hi] = True
    contaminated_total += int(mask.sum())

contam_pct = contaminated_total / total_rows * 100
log(f"  contaminated (t) positions: {contaminated_total:,} / {total_rows:,} = {contam_pct:.3f}%")
results["fwd_contamination"] = {
    "contaminated_positions": contaminated_total,
    "total_positions": int(total_rows),
    "pct": round(contam_pct, 4),
}

# ── 3. Strategy trades: how many crossed an extreme date? ──
log("=== 3. Strategy trades spanning extreme dates ===")
with TRADES.open() as f:
    import csv
    trades = list(csv.DictReader(f))
log(f"  trades: {len(trades)}")

# Build per-stock extreme date sets
ext_dates_by_stock = defaultdict(set)
for row in extreme_map.iter_rows(named=True):
    d = str(row["trade_date"])[:10]
    ext_dates_by_stock[row["ts_code"]].add(d)

# For each trade: check if any extreme date falls in (entry_time, exit_time] window for that stock
from datetime import datetime as _dt, timezone, timedelta
CST = timezone(timedelta(hours=8))

def ts_to_date(ns):
    return _dt.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).date()

contaminated_trades = 0
worst_examples = []
for t in trades:
    sym = t["symbol"]
    entry_d = ts_to_date(t["entry_time"])
    exit_d = ts_to_date(t["exit_time"])
    exts = ext_dates_by_stock.get(sym, set())
    hit = None
    for ed in exts:
        y, m, dd = map(int, ed.split("-"))
        from datetime import date as _date
        edd = _date(y, m, dd)
        if entry_d < edd <= exit_d:
            hit = ed
            break
    if hit:
        contaminated_trades += 1
        if len(worst_examples) < 10:
            worst_examples.append({
                "symbol": sym, "entry": str(entry_d), "exit": str(exit_d),
                "extreme_date": hit, "net_pnl": float(t["net_pnl"]),
            })

tcnt_pct = contaminated_trades / len(trades) * 100
log(f"  contaminated trades: {contaminated_trades} / {len(trades)} = {tcnt_pct:.2f}%")
log("  examples:")
for ex in worst_examples[:5]:
    log(f"    {ex['symbol']} {ex['entry']} → {ex['exit']} (spans {ex['extreme_date']}) pnl={ex['net_pnl']:,.0f}")
results["trade_contamination"] = {
    "contaminated_trades": contaminated_trades,
    "total_trades": len(trades),
    "pct": round(tcnt_pct, 2),
    "examples": worst_examples,
}

# ── 4. Estimate financial impact ──
log("=== 4. Estimated financial impact ===")
# For contaminated trades, the extreme move is mostly a mechanism for price adjustment.
# The trade PnL in raw price space includes the jump. Estimate the mean extreme magnitude for contaminated trades.
mean_extreme = extreme_pd["_ret1"].abs().mean()
# For the contaminated trades, typical span includes 1 extreme event; estimate impact:
# if half are negative (splits/dividends, mostly −30%..−78%), the PnL is artificially reduced by the jump %
contam_trade_pnl = sum(ex["net_pnl"] for ex in worst_examples)  # only examples
log(f"  mean extreme move magnitude: {mean_extreme:.3f}")
log(f"  note: ex-rights drops are artificial (shareholder not harmed), raw-price PnL understates returns")
results["impact_estimate"] = {
    "mean_extreme_abs": float(mean_extreme),
    "note": "Full PnL correction requires adjusted prices; estimate via event count only.",
}

# ── 5. Summary verdict ──
log("=== 5. Verdict ===")
verdict = {
    "panel_uses_raw_prices": True,
    "extreme_moves_total": extreme.height,
    "fwd_contaminated_pct": round(contam_pct, 4),
    "trades_contaminated_pct": round(tcnt_pct, 2),
    "severity": "MEDIUM" if tcnt_pct < 5 else "HIGH",
    "recommendation": "Verify with adjusted prices; quantify directional bias (likely understates returns, "
                      "since most ex-rights moves are downward price adjustments)."
}
log(f"  severity: {verdict['severity']}")

with (OUT / "audit_03_exrights.json").open("w") as f:
    json.dump({**results, "verdict": verdict}, f, indent=2, default=str)
log(f"\nSaved {OUT/'audit_03_exrights.json'}")
log("DONE module 3")
