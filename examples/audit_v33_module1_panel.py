"""V33 Comprehensive Audit — Module 1: Panel Data Integrity

Audits the v33 panel used by the V33 Factor-Return Voting strategy.
Outputs to evidence/audit_v33_20261007/.

Checks:
  1. Structure — rows, cols, dates, stocks
  2. Duplicate (trade_date, ts_code) pairs
  3. OHLC sanity — high >= max(open,close), low <= min(open,close), etc.
  4. Negative / zero prices
  5. adj_factor behavior per stock
  6. Null rates for voting factors
  7. Date continuity
"""
import json
import pathlib
import time
from datetime import datetime

import numpy as np
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
OUT = ROOT / "evidence" / "audit_v33_20261007"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


results = {}

# ── 1. Structure ──
log("=== 1. Structure ===")
schema = pl.read_parquet_schema(PANEL)
n_cols = len(schema)
log(f"Cols: {n_cols}")

# Full row count via metadata-friendly read
df_keys = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"])
n_rows = df_keys.height
n_stocks = df_keys["ts_code"].n_unique()
n_dates = df_keys["trade_date"].n_unique()
log(f"Rows: {n_rows:,} | Stocks: {n_stocks} | Dates: {n_dates}")
results["structure"] = {
    "n_cols": n_cols, "n_rows": n_rows,
    "n_stocks": n_stocks, "n_dates": n_dates,
    "first_date": str(df_keys["trade_date"].min()),
    "last_date": str(df_keys["trade_date"].max()),
}

# ── 2. Duplicates ──
log("=== 2. Duplicate check ===")
dup_count = n_rows - df_keys.unique().height
log(f"Duplicate (trade_date, ts_code) pairs: {dup_count}")
results["duplicates"] = int(dup_count)

del df_keys  # free memory

# ── 3. OHLC sanity ──
log("=== 3. OHLC sanity ===")
t0 = time.time()
ohlc = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "high", "low", "close"])
log(f"  loaded OHLC in {time.time()-t0:.1f}s")

checks = {}
# 3a. high >= max(open, close)
bad_high = ohlc.filter(
    (pl.col("open").is_not_null()) & (pl.col("high").is_not_null()) & (pl.col("close").is_not_null())
    & (pl.col("high") < pl.max_horizontal("open", "close"))
).height
checks["high_lt_max_oc"] = bad_high

# 3b. low <= min(open, close)
bad_low = ohlc.filter(
    (pl.col("open").is_not_null()) & (pl.col("low").is_not_null()) & (pl.col("close").is_not_null())
    & (pl.col("low") > pl.min_horizontal("open", "close"))
).height
checks["low_gt_min_oc"] = bad_low

# 3c. high >= low
bad_hl = ohlc.filter(
    (pl.col("high").is_not_null()) & (pl.col("low").is_not_null())
    & (pl.col("high") < pl.col("low"))
).height
checks["high_lt_low"] = bad_hl

# 3d. Negative prices
neg = ohlc.filter(
    (pl.col("open") < 0) | (pl.col("high") < 0) | (pl.col("low") < 0) | (pl.col("close") < 0)
).height
checks["negative_prices"] = neg

# 3e. Zero or null prices
zero_close = ohlc.filter(pl.col("close") == 0).height
null_close = ohlc.filter(pl.col("close").is_null()).height
checks["zero_close"] = zero_close
checks["null_close"] = null_close

for k, v in checks.items():
    flag = "PASS" if v == 0 else "FAIL"
    log(f"  [{flag}] {k}: {v}")
results["ohlc_sanity"] = checks

# ── 4. adj_factor behavior ──
log("=== 4. adj_factor ===")
adjf = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_factor"])
adjf_unique_per_stock = adjf.group_by("ts_code").agg(pl.col("adj_factor").n_unique().alias("n"))
stats = adjf_unique_per_stock["n"].to_numpy()
log(f"  adj_factor unique values per stock: min={stats.min()}, median={np.median(stats):.0f}, max={stats.max()}")
# adj_factor should be monotonic non-decreasing per stock (typical)
adjf_sorted = adjf.sort(["ts_code", "trade_date"])
viol = adjf_sorted.with_columns(
    (pl.col("adj_factor").diff().over("ts_code") < -1e-9).alias("_dec")
).filter(pl.col("_dec")).height
log(f"  adj_factor decreases (within stock): {viol}")
results["adj_factor"] = {
    "min_unique": int(stats.min()), "median_unique": float(np.median(stats)),
    "max_unique": int(stats.max()), "decreases": int(viol),
}
del adjf, adjf_sorted, adjf_unique_per_stock

# ── 5. Null rates for voting factors ──
log("=== 5. Voting factor null rates ===")
base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
             "vol", "amount", "pct_chg", "adj_factor", "adj_close",
             "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
             "turnover_rate", "circ_cap", "cap", "symbol", "volume",
             "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
zigzag_labels = {"v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
                 "v10_1_b1_start", "v10_1_b1_interval",
                 "v10_1_down_start", "v10_1_down_interval",
                 "v10_1_peak_zone", "v10_1_valley_zone",
                 "v10_1_zig_peak", "v10_1_zig_valley"}
all_cols = list(schema.keys())
voting_factors = [c for c in all_cols if c not in (base_cols | zigzag_labels)]
log(f"  voting factors: {len(voting_factors)}")

# Compute null rates for all voting factors in one pass (memory-aware, batch of 50)
null_rates = {}
t0 = time.time()
for i in range(0, len(voting_factors), 50):
    batch = voting_factors[i:i+50]
    d = pl.read_parquet(PANEL, columns=batch)
    for c in batch:
        null_rates[c] = d[c].null_count() / len(d)
    del d
log(f"  null rates computed in {time.time()-t0:.1f}s")

all_null = [c for c, r in null_rates.items() if r > 0.99]
high_null = [c for c, r in null_rates.items() if 0.5 < r <= 0.99]
mid_null = [c for c, r in null_rates.items() if 0.1 < r <= 0.5]
low_null = [c for c, r in null_rates.items() if r <= 0.1]
log(f"  all-null (>99%): {len(all_null)}")
log(f"  high-null (50-99%): {len(high_null)}")
log(f"  mid-null (10-50%): {len(mid_null)}")
log(f"  low-null (<=10%): {len(low_null)}")
if all_null:
    for c in all_null[:30]:
        log(f"    ALL-NULL: {c}")
results["null_rates"] = {
    "n_voting_factors": len(voting_factors),
    "all_null_count": len(all_null), "all_null_list": all_null,
    "high_null_count": len(high_null),
    "mid_null_count": len(mid_null), "low_null_count": len(low_null),
}
with (OUT / "null_rates.json").open("w") as f:
    json.dump({c: round(r, 4) for c, r in sorted(null_rates.items(), key=lambda x: -x[1])}, f, indent=1)

# ── 6. Date continuity ──
log("=== 6. Date continuity ===")
dates = pl.read_parquet(PANEL, columns=["trade_date"]).unique().sort("trade_date")
dlist = dates["trade_date"].to_list()
gaps = []
for i in range(1, len(dlist)):
    delta = (dlist[i] - dlist[i-1]).days
    if delta > 20:  # > 20 days = suspicious gap (should be 1-7 days typical)
        gaps.append((str(dlist[i-1])[:10], str(dlist[i])[:10], delta))
log(f"  gaps > 20 calendar days: {len(gaps)}")
for g in gaps[:10]:
    log(f"    {g}")
results["date_gaps"] = [{"from": g[0], "to": g[1], "days": g[2]} for g in gaps[:50]]

# ── 7. Stock count per date stability ──
log("=== 7. Stocks per date ===")
spd = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"]).group_by("trade_date").agg(pl.len().alias("n")).sort("trade_date")
narr = spd["n"].to_numpy()
log(f"  stocks/date: min={narr.min()}, max={narr.max()}, mean={narr.mean():.1f}, median={np.median(narr):.0f}")
# check recent years
recent = spd.filter(pl.col("trade_date") >= pl.datetime(2010, 1, 1))
rarr = recent["n"].to_numpy()
log(f"  2010+: min={rarr.min()}, max={rarr.max()}, mean={rarr.mean():.1f}")
results["stocks_per_date"] = {
    "min": int(narr.min()), "max": int(narr.max()),
    "mean": float(narr.mean()), "median": float(np.median(narr)),
    "min_2010_plus": int(rarr.min()), "max_2010_plus": int(rarr.max()),
}

with (OUT / "audit_01_panel_integrity.json").open("w") as f:
    json.dump(results, f, indent=2)
log(f"\nSaved {OUT/'audit_01_panel_integrity.json'}")
log("DONE module 1")
