"""Probe: compare stage-A matrix values vs builder-style direct computation for v34.

For the 20 sessions before 2010-01-06, compute per-factor diffs two ways:
  (a) stage-A matrix_v34_ADJ
  (b) direct replication of build_picks_v34_factorrank.py (both filters, its windowing)
Find diverging factors/values.
"""
import json
import pathlib
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
ENTRY_TO_EXIT_SHIFT = 21
ROUND_TRIP_COST = 0.005
LOOKBACK = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_factors(panel_path):
    cols = pl.read_parquet_schema(panel_path).keys()
    base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
                 "vol", "amount", "pct_chg", "adj_factor", "adj_close",
                 "adj_open", "adj_high", "adj_low",
                 "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
                 "turnover_rate", "circ_cap", "cap", "symbol", "volume",
                 "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
    zigzag = {"v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
              "v10_1_b1_start", "v10_1_b1_interval",
              "v10_1_down_start", "v10_1_down_interval",
              "v10_1_peak_zone", "v10_1_valley_zone",
              "v10_1_zig_peak", "v10_1_valley_zone"}
    return [c for c in cols if c not in (base_cols | zigzag)]


factors = get_factors(V34)
log(f"factors: {len(factors)}")

# ── (b) builder-style: exact replication ──
panel_full = (
    pl.scan_parquet(V34)
    .select(["trade_date", "ts_code", "adj_open", "adj_close"] + factors)
    .sort(["ts_code", "trade_date"])
    .with_columns([
        pl.col("adj_open").shift(-1).over("ts_code").alias("_open_t1"),
        pl.col("adj_close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
    ])
    .with_columns((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_net"))
    .select(["trade_date", "ts_code", "_fwd_net"] + factors)
    .collect()
)
log(f"panel_full: {panel_full.shape}")

all_dates = [str(x)[:10] for x in panel_full["trade_date"].unique().sort().to_list()]
assert len(all_dates) == len(set(all_dates))
i0 = all_dates.index("2010-01-06")
probe_dates = all_dates[i0 - LOOKBACK:i0]
log(f"probe dates: {probe_dates[0]}..{probe_dates[-1]} ({len(probe_dates)})")

dtset = set(probe_dates)
sub = panel_full.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(list(dtset)))

# compute diffs builder-style on the subset
bdiff = {}
for f in factors:
    d = sub.select(["trade_date", "_fwd_net", f])
    d = d.filter(pl.col(f).is_finite() & pl.col("_fwd_net").is_finite())
    d = d.with_columns(pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk"))
    agg = d.group_by("trade_date").agg([
        (pl.col("_fwd_net").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top"),
        (pl.col("_fwd_net").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot"),
    ])
    agg = agg.with_columns((pl.col("_top") - pl.col("_bot")).alias(f)).select(["trade_date", f])
    bdiff[f] = agg

# ── (a) matrix-style ──
mat = pl.read_parquet(MAT)
mat_sub = mat.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(list(dtset)))
log(f"matrix sub: {mat_sub.shape}")

# compare per factor at each date
from collections import defaultdict
diffs_by_factor = defaultdict(list)
base = pl.DataFrame({"trade_date": mat_sub["trade_date"]})
for f in factors:
    m = base.join(bdiff[f], on="trade_date", how="left")[f].to_numpy()
    a = mat_sub[f].to_numpy()
    both = np.isfinite(m) & np.isfinite(a)
    if both.sum() == 0:
        diffs_by_factor[f] = None
        continue
    dd = np.abs(m[both] - a[both])
    diffs_by_factor[f] = float(dd.max()) if len(dd) else None

# stats
vals = {k: v for k, v in diffs_by_factor.items() if v is not None}
big = sorted(vals.items(), key=lambda kv: -kv[1])[:20]
print("\nTop diverging factors (max|Δdiff| over 20 sessions):")
for k, v in big:
    print(f"  {k}: {v:.6g}")
n_zero = sum(1 for v in vals.values() if v < 1e-12)
n_big = sum(1 for v in vals.values() if v > 1e-9)
print(f"\nfactors identical: {n_zero}/{len(vals)}; divergent(>1e-9): {n_big}")

# missing coverage
n_none = sum(1 for f in factors if diffs_by_factor.get(f) is None)
print(f"factors with no overlap: {n_none}")

# which of the top diverging are missing in one source?
for k, v in big[:6]:
    m = base.join(bdiff[k], on="trade_date", how="left")[k]
    a = mat_sub[k]
    print(f"\n{k}: builder n_finite={m.is_finite().sum()}, matrix n_finite={a.is_finite().sum()}")
    frame = pl.DataFrame({"trade_date": mat_sub["trade_date"], "builder": m, "matrix": a})
    print(frame.tail(4))
