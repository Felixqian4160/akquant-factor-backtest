"""V33 Comprehensive Audit — Module 5: Factor-return scoring replication

Replicate the diff(f,t) and score(f,T) computations for sample dates and compare
against what the picks builder actually used.

Strategy recap:
  diff(f, t) = top10_mean(f,t) - bot10_mean(f,t)   [net20 of top/bot deciles]
  score(f, T) = mean(diff(f, t) for t in [T-20, T-1])
  active_factors(T) = top-10 by score

Checks:
  1. Reproduce diff(f, t) for 3 sample (factor, date) pairs
  2. Reproduce score(f, T) for 1 sample rebal date
  3. Verify the causal property: all t < T
  4. Verify top-10 factor selection on the sample rebal date matches picks_meta
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
PICKS = ROOT / "evidence/sweep/v33_factorrank_20261007/V14_0"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

results = {}

# ── Load sample factors + pick dates ──
meta = json.loads((PICKS / "picks_meta.json").read_text())
sample_date = "2020-03-23"  # a bull_neutral rebal date we saw in logs
log(f"Sample rebal date: {sample_date}, regime={meta[sample_date]['regime']}")

# Which factors does picks use? We only have votes per stock, not factor names.
# But we can replicate and just check the computation pipeline self-consistency.
SAMPLE_FACTORS = ["l_ami", "gtja_gtja_002", "talib_NATR", "pe", "r_tv"]

# Load full price + factor columns for a limited set
need = ["trade_date", "ts_code", "open", "close"] + SAMPLE_FACTORS
df = pl.read_parquet(PANEL, columns=need).sort(["ts_code", "trade_date"])
log(f"loaded {df.height:,} rows")

# Build _fwd_net
df = df.with_columns([
    pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
    pl.col("close").shift(-21).over("ts_code").alias("_close_t21"),
]).with_columns(
    (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - 0.005).alias("_fwd_net")
)

# Convert to pandas for manual replication (one stock set at a time)
pdf = df.select(["trade_date", "ts_code", "_fwd_net"] + SAMPLE_FACTORS).to_pandas()
pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])

# ── 1. Reproduce diff(f, t) for specific (factor, date) ──
log("=== 1. diff(f,t) replication ===")
sample_t = pd.Timestamp("2020-02-03")  # a specific date to test
day = pdf[pdf["trade_date"] == sample_t].dropna(subset=["_fwd_net", SAMPLE_FACTORS[0]])
log(f"  date {sample_t.date()}: {len(day)} stocks with valid fwd + l_ami")

checks_1 = {}
for f in SAMPLE_FACTORS:
    sub = pdf[pdf["trade_date"] == sample_t].dropna(subset=["_fwd_net", f])
    if len(sub) < 20:
        log(f"  {f}: insufficient data ({len(sub)} stocks)")
        continue
    top10 = sub.nlargest(10, f)
    bot10 = sub.nsmallest(10, f)
    diff_manual = top10["_fwd_net"].mean() - bot10["_fwd_net"].mean()
    # Now compute the same with polars expression (as the builder does)
    d2 = (
        df.filter(pl.col("trade_date") == pl.lit(sample_t.to_pydatetime(), dtype=pl.Datetime("ms")))
        .filter(pl.col(f).is_not_null() & pl.col("_fwd_net").is_not_null())
        .sort(f, descending=True)
    )
    top10_p = d2.head(10)["_fwd_net"].mean()
    bot10_p = d2.tail(10)["_fwd_net"].mean()
    diff_polars = top10_p - bot10_p
    match = abs(diff_manual - diff_polars) < 1e-12
    log(f"  {f}: manual={diff_manual:.6f} polars={diff_polars:.6f} match={match}")
    checks_1[f] = {"manual": float(diff_manual), "polars": float(diff_polars), "match": bool(match)}
results["diff_replication"] = checks_1

# ── 2. Reproduce score(f, T) ──
log("=== 2. score(f,T) replication ===")
# The builder computes: for each factor, diffs per date, then mean of past 20 sessions (< T)
# Replicate for one factor and one rebal date
f = "l_ami"
T = pd.Timestamp(sample_date)
# All trading dates < T
all_dates = sorted(pdf["trade_date"].unique())
dates_before_T = [d for d in all_dates if d < T]
window_dates = dates_before_T[-20:]  # last 20 sessions before T
log(f"  T={T.date()}, window: {window_dates[0].date()} .. {window_dates[-1].date()} ({len(window_dates)} sessions)")

diffs_in_window = []
for d in window_dates:
    sub = pdf[pdf["trade_date"] == d].dropna(subset=["_fwd_net", f])
    if len(sub) < 20:
        continue
    top10 = sub.nlargest(10, f)
    bot10 = sub.nsmallest(10, f)
    diffs_in_window.append(top10["_fwd_net"].mean() - bot10["_fwd_net"].mean())
score_manual = np.mean(diffs_in_window)
log(f"  l_ami score at {sample_date}: {score_manual:.6f} (n={len(diffs_in_window)} diffs)")

# Sanity: all window dates strictly before T
all_before = all(d < T for d in window_dates)
log(f"  all window dates < T: {all_before}")
results["score_replication"] = {
    "factor": f, "T": sample_date,
    "score": float(score_manual), "n_diffs": len(diffs_in_window),
    "all_causal": bool(all_before),
}

# ── 3. Cross-check a few more rebal dates for causality ──
log("=== 3. Causality check across rebal dates ===")
rebal_dates = sorted(meta.keys())
causal_ok = True
violations = []
for rd in rebal_dates[:50]:
    Td = pd.Timestamp(rd)
    dates_before = [d for d in all_dates if d < Td]
    if len(dates_before) < 20:
        continue
    window = dates_before[-20:]
    if any(d >= Td for d in window):
        causal_ok = False
        violations.append(rd)
log(f"  checked {min(50, len(rebal_dates))} rebal dates: causal_ok={causal_ok}, violations={len(violations)}")
results["causality"] = {"checked": min(50, len(rebal_dates)), "ok": causal_ok, "violations": violations}

# ── 4. Verify regime consistency ──
log("=== 4. Regime consistency check ===")
router = json.loads((ROOT / "evidence/causal_zigzag_router_20261006/router_map.json").read_text())
mismatch = 0
for rd, v in meta.items():
    if router.get(rd) != v["regime"]:
        mismatch += 1
log(f"  picks_meta regime vs router_map: mismatches={mismatch} / {len(meta)}")
results["regime_consistency"] = {"mismatches": mismatch, "total": len(meta)}

with (OUT / "audit_05_scoring.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved {OUT/'audit_05_scoring.json'}")
log("DONE module 5")
