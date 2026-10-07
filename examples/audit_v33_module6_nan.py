"""V33 Audit Module 6: NaN-handling bug impact quantification.

Bug: compute_factor_returns (scoring) uses `is_not_null()` which does NOT filter
NaN float values. Polars `rank(descending=True)` puts NaN at rank 1 (top).
=> For factors containing NaN, the "top-10" stocks in the diff computation are
   NaN-valued (arbitrary) stocks, biasing diff(f,t).

This script:
  1. Recomputes per-(factor,date) diffs with FIXED logic (exclude NaN from ranking)
     for all 428 factors.
  2. Recomputes diffs with LEGACY (buggy) logic for the 33 NaN-containing factors.
  3. Builds scores for both variants, ranks factors per rebal date, extracts top-10.
  4. Compares: how many dates have different top-10 sets; how many slots change.

Output: evidence/audit_v33_20261007/audit_06_nan_impact.json
"""
import json
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
ROUTER = ROOT / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = ROOT / "evidence" / "audit_v33_20261007"

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
LOOKBACK = 20
TOP_K = 10

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Setup: factor list & panel load (same as builder) ──
v33_schema = pl.read_parquet_schema(PANEL)
v33_cols = list(v33_schema.keys())
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
factors = [c for c in v33_cols if c not in (base_cols | zigzag_labels)]
log(f"voting factors: {len(factors)}")

log("loading panel...")
t0 = time.time()
panel_full = (
    pl.scan_parquet(PANEL)
    .select(["trade_date", "ts_code", "open", "close"] + factors)
    .sort(["ts_code", "trade_date"])
    .with_columns([
        pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
        pl.col("close").shift(-21).over("ts_code").alias("_close_t21"),
    ])
    .with_columns(
        (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - 0.005).alias("_fwd_net")
    )
    .select(["trade_date", "ts_code", "_fwd_net"] + factors)
    .collect()
)
log(f"panel loaded in {time.time()-t0:.0f}s, shape {panel_full.shape}")

all_dates_df = pl.scan_parquet(PANEL).select(["trade_date"]).unique().sort("trade_date").collect()
all_dates = [str(x)[:10] for x in all_dates_df["trade_date"].to_list()]
start_dt = pd.Timestamp(START).date()
end_dt = pd.Timestamp(END).date()
rebal_dates = [d for d in all_dates[0::REBAL_STEP] if start_dt <= pd.Timestamp(d).date() <= end_dt]
log(f"rebal dates: {len(rebal_dates)}")


def compute_scores(panel, factors_list, use_finite_filter, all_dates, lookback):
    """Compute score[f][date] = mean of past-lookback diffs (t < date).

    use_finite_filter=True  → exclude NaN from ranking (FIXED)
    use_finite_filter=False → keep NaN; NaN ranks top (LEGACY bug)
    """
    n_dates = len(all_dates)
    scores = {}  # {date_iso: {factor: score}}
    for d_iso in all_dates[lookback:]:
        scores[d_iso] = {}
    t0 = time.time()
    for fi, f in enumerate(factors_list, start=1):
        df = panel.select(["trade_date", "_fwd_net", f])
        if use_finite_filter:
            df = df.filter(pl.col(f).is_finite() & pl.col("_fwd_net").is_finite())
        else:
            df = df.filter(pl.col(f).is_not_null() & pl.col("_fwd_net").is_not_null())
        df = df.with_columns(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk")
        )
        per_date = df.group_by("trade_date").agg([
            (pl.col("_fwd_net").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top_mean"),
            (pl.col("_fwd_net").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot_mean"),
        ]).sort("trade_date").select(["trade_date", (pl.col("_top_mean") - pl.col("_bot_mean")).alias("_diff")])

        date_pos, diffs = [], []
        for row in per_date.iter_rows(named=True):
            d_str = str(row["trade_date"])[:10]
            pos = all_dates.index(d_str) if d_str in all_dates else -1
            if pos >= 0 and np.isfinite(row["_diff"]):
                date_pos.append(pos)
                diffs.append(row["_diff"])
        if not date_pos:
            continue
        date_pos_arr = np.asarray(date_pos, dtype=np.int64)
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        for i in range(lookback, n_dates):
            d_iso = all_dates[i]
            mask = date_pos_arr < i
            if mask.sum() < 10:
                continue
            window = diffs_arr[mask][-lookback:]
            window = window[np.isfinite(window)]
            if len(window) >= 10:
                scores[d_iso][f] = float(np.mean(window))
        if fi % 50 == 0:
            log(f"    {fi}/{len(factors_list)} {f} elapsed {time.time()-t0:.0f}s")
    log(f"  scores computed in {time.time()-t0:.0f}s")
    return scores


# Identify NaN-containing factors first
log("identifying NaN factors...")
nan_factors = []
for i in range(0, len(factors), 50):
    batch = factors[i:i+50]
    d = pl.read_parquet(PANEL, columns=batch)
    for c in batch:
        if d[c].dtype == pl.Float64 and d[c].is_nan().sum() > 0:
            nan_factors.append(c)
    del d
clean_factors = [f for f in factors if f not in nan_factors]
log(f"NaN factors: {len(nan_factors)}, clean: {len(clean_factors)}")

# ── Pass 1: fixed scores for ALL factors ──
log("=== Pass 1: FIXED scores (all 428) ===")
scores_fixed = compute_scores(panel_full, factors, True, all_dates, LOOKBACK)

# ── Pass 2: legacy scores for NaN factors only ──
log("=== Pass 2: LEGACY scores (33 NaN factors) ===")
scores_legacy_nan = compute_scores(panel_full, nan_factors, False, all_dates, LOOKBACK)

# ── Build top-10 per rebal date for both variants ──
log("=== Compare top-10 selection ===")
diff_dates = []
slot_changes = 0
for rd in rebal_dates:
    fixed_map = scores_fixed.get(rd, {})
    # legacy: clean factors same as fixed + legacy scores for NaN factors
    legacy_map = {f: s for f, s in fixed_map.items() if f in clean_factors}
    legacy_map.update(scores_legacy_nan.get(rd, {}))
    if not fixed_map and not legacy_map:
        continue
    top_fixed = [f for f, _ in sorted(fixed_map.items(), key=lambda kv: -kv[1])[:TOP_K]]
    top_legacy = [f for f, _ in sorted(legacy_map.items(), key=lambda kv: -kv[1])[:TOP_K]]
    if set(top_fixed) != set(top_legacy):
        diff_dates.append({
            "date": rd,
            "added": sorted(set(top_legacy) - set(top_fixed)),
            "removed": sorted(set(top_fixed) - set(top_legacy)),
        })
        slot_changes += len(set(top_legacy) - set(top_fixed))

log(f"dates with different top-10: {len(diff_dates)} / {len(rebal_dates)} "
    f"({len(diff_dates)/len(rebal_dates)*100:.1f}%)")
log(f"total slot changes: {slot_changes} (of {len(rebal_dates)*TOP_K} = "
    f"{slot_changes/(len(rebal_dates)*TOP_K)*100:.2f}%)")
log("First 10 diff dates:")
for x in diff_dates[:10]:
    log(f"  {x['date']}: -{x['removed']} +{x['added']}")

results = {
    "nan_factor_count": len(nan_factors),
    "nan_factors": nan_factors,
    "rebal_dates": len(rebal_dates),
    "dates_with_diff": len(diff_dates),
    "pct_dates_diff": round(len(diff_dates)/len(rebal_dates)*100, 1),
    "total_slot_changes": slot_changes,
    "pct_slot_changes": round(slot_changes/(len(rebal_dates)*TOP_K)*100, 2),
    "diff_dates": diff_dates,
}
with (OUT / "audit_06_nan_impact.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Saved {OUT/'audit_06_nan_impact.json'}")
log("DONE module 6")
