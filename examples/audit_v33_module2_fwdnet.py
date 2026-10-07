"""V33 Comprehensive Audit — Module 2 (fixed): Price & fwd_net calculation correctness

Key checks:
  1. adj_factor / adj_close / close relationship  → panel uses RAW prices (adj_factor=1)
  2. Split/dividend jump detection in raw prices  → impact on fwd returns
  3. _fwd_net formula replication
  4. T+1 timing correctness
  5. Terminal edge behavior
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

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

results = {}

# ── 1. adj_factor confirm + significance ──
log("=== 1. adj_factor analysis ===")
df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "close", "adj_factor", "adj_close"])
log(f"  adj_factor n_unique: {df['adj_factor'].n_unique()} (all=1.0 means RAW prices)")
log(f"  adj_close == close: {(df['adj_close'] - df['close']).abs().max()}")
results["adj_factor"] = {
    "n_unique": int(df["adj_factor"].n_unique()),
    "adj_close_eq_close": bool((df["adj_close"] - df["close"]).abs().max() == 0),
}
del df

# ── 2. Split/jump detection: look for daily returns < -30% (possible split or crash) ──
log("=== 2. Extreme daily move detection (split/crash candidates) ===")
px = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"]).sort(["ts_code", "trade_date"])
px = px.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret1")
)
# returns < -30%: could be splits, or real crashes (A-share daily limit is ±10% for normal, ±20% for STAR/ChiNext)
big_drop = px.filter(pl.col("_ret1") < -0.30)
big_rise = px.filter(pl.col("_ret1") > 0.30)
log(f"  daily returns < -30%: {big_drop.height}")
log(f"  daily returns > +30%: {big_rise.height}")
# Top 10 biggest
log("  Top 10 biggest drops:")
for row in big_drop.sort("_ret1").head(10).iter_rows(named=True):
    log(f"    {str(row['trade_date'])[:10]} {row['ts_code']}: close={row['close']:.3f} ret={row['_ret1']:.3f}")
results["extreme_moves"] = {
    "n_big_drop": big_drop.height,
    "n_big_rise": big_rise.height,
}
# Note: A-shares have ±10% daily limit (main board), ±20% (ChiNext/STAR). Moves beyond ±21% are impossible without ex-rights adjustment
impossible_moves = px.filter((pl.col("_ret1") < -0.21) | (pl.col("_ret1") > 0.21))
log(f"  moves beyond ±21% (impossible without ex-rights or data error): {impossible_moves.height}")
results["extreme_moves"]["beyond_21pct"] = impossible_moves.height
del px, big_drop, big_rise

# ── 3. _fwd_net replication ──
log("=== 3. _fwd_net formula replication ===")
prices = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"]).sort(["ts_code", "trade_date"])
prices = prices.with_columns([
    pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
    pl.col("close").shift(-21).over("ts_code").alias("_close_t21"),
])
prices = prices.with_columns(
    (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - 0.005).alias("_fwd_net_calc")
)

# Independent pandas replication on full panel
pdf = prices.select(["trade_date", "ts_code", "open", "close"]).to_pandas()
pdf = pdf.sort_values(["ts_code", "trade_date"])
pdf["_open_t1_indep"] = pdf.groupby("ts_code")["open"].shift(-1)
pdf["_close_t21_indep"] = pdf.groupby("ts_code")["close"].shift(-21)
pdf["_fwd_indep"] = pdf["_close_t21_indep"] / pdf["_open_t1_indep"] - 1 - 0.005
merged = pdf.merge(
    prices.select(["trade_date", "ts_code", "_fwd_net_calc"]).to_pandas(),
    on=["trade_date", "ts_code"], how="inner")
diff = (merged["_fwd_indep"] - merged["_fwd_net_calc"]).abs()
log(f"  polars vs pandas max diff: {diff.max():.2e}, mismatches(>1e-9): {(diff>1e-9).sum()}")
results["fwd_net_replication"] = {
    "max_diff": float(diff.max()) if not np.isnan(diff.max()) else None,
    "n_large_mismatch": int((diff > 1e-9).sum()),
}

# ── 4. Timing check on a long-history stock ──
log("=== 4. T+1 entry / T+21 exit timing ===")
s = "600519.SH"  # Moutai, full history
sub = prices.filter(pl.col("ts_code") == s).sort("trade_date")
log(f"  {s} rows: {sub.height}")
i = 3000  # safe index
row_t = sub.slice(i, 1).to_dicts()[0]
row_t1 = sub.slice(i + 1, 1).to_dicts()[0]
row_t21 = sub.slice(i + 21, 1).to_dicts()[0]
expected_fwd = row_t21["close"] / row_t1["open"] - 1 - 0.005
got = row_t["_fwd_net_calc"]
log(f"  t={str(row_t['trade_date'])[:10]}: fwd={got:.6f}")
log(f"  t+1={str(row_t1['trade_date'])[:10]} open={row_t1['open']:.3f}")
log(f"  t+21={str(row_t21['trade_date'])[:10]} close={row_t21['close']:.3f}")
log(f"  expected={expected_fwd:.6f}  match={abs(expected_fwd-got)<1e-9}")
results["timing"] = {"expected": float(expected_fwd), "got": float(got),
                     "match": bool(abs(expected_fwd - got) < 1e-9)}

# ── 5. Terminal rows ──
log("=== 5. Terminal rows ===")
last_rows = prices.group_by("ts_code").tail(21)
null_in_last21 = last_rows.filter(pl.col("_fwd_net_calc").is_null()).height / 21
log(f"  last-21 fwd_net null rate: {null_in_last21:.4f} (expected 1.0)")
overall_null = prices.filter(pl.col("_fwd_net_calc").is_null()).height
log(f"  overall null: {overall_null:,} (expected ≈ {354*21})")
results["terminal"] = {
    "last21_null_rate": float(null_in_last21),
    "overall_null": int(overall_null),
    "expected_null": 354 * 21,
}

# ── 6. Raw price plausibility ──
log("=== 6. Raw price plausibility ===")
mt = prices.filter((pl.col("ts_code") == "600519.SH") & (pl.col("trade_date") == pl.datetime(2021, 2, 18)))
if mt.height:
    row = mt.to_dicts()[0]
    log(f"  600519.SH 2021-02-18 (all-time high zone): close={row['close']:.2f} (expected raw ≈ 2600)")
check2 = prices.filter((pl.col("ts_code") == "000001.SZ") & (pl.col("trade_date") == pl.datetime(2022, 6, 1)))
if check2.height:
    row = check2.to_dicts()[0]
    log(f"  000001.SZ 2022-06-01: close={row['close']:.2f} (expected raw ≈ 13-14)")

with (OUT / "audit_02_fwd_net.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved {OUT/'audit_02_fwd_net.json'}")
log("DONE module 2")
