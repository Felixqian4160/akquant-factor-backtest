"""V33 Audit — Module 8: SURGICAL divergence analysis.

Question: why do the legacy and nanfix builders produce different picks on
2010-01-06, when module 6 claimed top-10 factor lists differ on only 28 dates?

Plan:
  1. Compute scores for ALL 428 factors under BOTH filters (full paths).
  2. For 2010-01-06: print both top-10 lists with scores.
  3. Reconstruct votes for both lists; compare with actual picks.json.
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
OUT = ROOT / "evidence" / "audit_v33_20261007"

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
LOOKBACK = 20
TOP_K = 10
K_NOM = 10

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

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

# dtype check
dtypes = {}
batch = pl.read_parquet(PANEL, columns=factors[:100])
for c in factors[:100]:
    dtypes[str(batch[c].dtype)] = dtypes.get(str(batch[c].dtype), 0) + 1
log(f"dtype sample (first 100): {dtypes}")

log("loading panel...")
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
log(f"panel shape: {panel_full.shape}")

all_dates_df = pl.scan_parquet(PANEL).select(["trade_date"]).unique().sort("trade_date").collect()
all_dates = [str(x)[:10] for x in all_dates_df["trade_date"].to_list()]
start_dt = pd.Timestamp(START).date()
end_dt = pd.Timestamp(END).date()
rebal_dates = [d for d in all_dates[0::REBAL_STEP] if start_dt <= pd.Timestamp(d).date() <= end_dt]
log(f"rebal dates: {len(rebal_dates)}")


def compute_scores(panel, factors_list, use_finite, all_dates, lookback):
    n_dates = len(all_dates)
    out = {d: {} for d in all_dates[lookback:]}
    t0 = time.time()
    for fi, f in enumerate(factors_list, start=1):
        df = panel.select(["trade_date", "_fwd_net", f])
        if use_finite:
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
        dpa = np.asarray(date_pos, dtype=np.int64)
        da = np.asarray(diffs, dtype=np.float64)
        for i in range(lookback, n_dates):
            d_iso = all_dates[i]
            m = dpa < i
            if m.sum() < 10:
                continue
            w = da[m][-lookback:]
            w = w[np.isfinite(w)]
            if len(w) >= 10:
                out[d_iso][f] = float(np.mean(w))
        if fi % 100 == 0:
            log(f"    {fi}/{len(factors_list)} elapsed {time.time()-t0:.0f}s")
    return out


log("computing FIXED scores (all 428)...")
t0 = time.time()
scores_fixed = compute_scores(panel_full, factors, True, all_dates, LOOKBACK)
log(f"fixed done {time.time()-t0:.0f}s")

log("computing LEGACY scores (all 428)...")
t0 = time.time()
scores_legacy = compute_scores(panel_full, factors, False, all_dates, LOOKBACK)
log(f"legacy done {time.time()-t0:.0f}s")

# ── Top-10 comparison for 2010-01-06 ──
d = "2010-01-06"
fixed_map = scores_fixed[d]
legacy_map = scores_legacy[d]
top_fixed = sorted(fixed_map.items(), key=lambda kv: -kv[1])[:15]
top_legacy = sorted(legacy_map.items(), key=lambda kv: -kv[1])[:15]

log(f"\n=== {d} top-15 FIXED ===")
for f, s in top_fixed:
    log(f"  {f}: {s:.6f}")
log(f"=== {d} top-15 LEGACY ===")
for f, s in top_legacy:
    log(f"  {f}: {s:.6f}")

top10_fixed = [f for f, _ in top_fixed[:10]]
top10_legacy = [f for f, _ in top_legacy[:10]]
log(f"\ntop10 fixed: {top10_fixed}")
log(f"top10 legacy: {top10_legacy}")
log(f"same set: {set(top10_fixed) == set(top10_legacy)}")

# ── Reconstruct votes for both lists ──
log("\n=== Vote reconstruction ===")
day = panel_full.filter(pl.col("trade_date") == pl.lit(pd.Timestamp(d).to_pydatetime(), dtype=pl.Datetime("ms")))
codes = day.get_column("ts_code").to_list()
log(f"day rows: {day.height}")

for label, active in [("FIXED", top10_fixed), ("LEGACY", top10_legacy)]:
    values = day.select(active).to_numpy()
    votes = np.zeros(day.height, dtype=np.int32)
    skip_count = 0
    for j, factor in enumerate(active):
        column = values[:, j]
        valid = np.flatnonzero(np.isfinite(column))
        if len(valid) < K_NOM:
            skip_count += 1
            continue
        chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
        votes[chosen] += 1
    order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
    sel = [(codes[i], int(votes[i])) for i in order[:10]]
    log(f"{label}: skipped_factors={skip_count}, top10 picks: {sel}")

# Actual picks
old_picks = json.loads((ROOT/'evidence/sweep/v33_factorrank_20261007/V14_0/picks.json').read_text())
new_picks = json.loads((ROOT/'evidence/sweep/v33_factorrank_lb20_nanfix_20261007/V14_0/picks.json').read_text())
log(f"\nactual OLD picks: {old_picks[d]}")
log(f"actual NEW picks: {new_picks[d]}")

# Save
res = {
    "date": d,
    "top10_fixed": top10_fixed,
    "top10_legacy": top10_legacy,
    "same_set": bool(set(top10_fixed) == set(top10_legacy)),
    "actual_old": old_picks[d],
    "actual_new": new_picks[d],
}
with (OUT / "audit_08_surgical.json").open("w") as f:
    json.dump(res, f, indent=2)
log("\nDONE module 8")
