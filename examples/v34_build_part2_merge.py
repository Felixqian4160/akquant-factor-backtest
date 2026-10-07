"""V34 panel build — Part 2: merge adjusted factors into the v33 panel.

Output: data/wavehunter_hs300_v34_adj_20261007.parquet

Changes vs v33:
  - Replace all gtja_/alpha_/talib_ factor columns with adjusted-price versions
  - Replace 12 price-sensitive academic factors (l_ami, r_tv, r_beta, p_m1..p_mdr)
  - Replace 17 github factors
  - Replace adj_factor with REAL factor; adj_close = real; ADD adj_open/adj_high/adj_low
  - Keep: raw OHLCV (execution), fundamentals, labels, idx_*, p_season etc.
"""
from __future__ import annotations
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V33 = AKQ / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
NEWF = AKQ / "evidence" / "v34_adj_20261007" / "new_factors_adj.parquet"
ADJ = AKQ / "evidence" / "audit_v33_20261007" / "adj_study" / "adj_prices_354.parquet"
OUT = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
OUTDIR = AKQ / "evidence" / "v34_adj_20261007"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ACADEMIC_REBUILD = ["l_ami", "r_tv", "r_beta", "p_m1", "p_m3", "p_m6", "p_m11", "p_m24",
                    "p_mchg", "p_52w", "p_mdr"]
GITHUB17 = ["winner_ratio", "efficiency_ratio", "fractal_dimension", "alpha191_040",
            "alpha191_095", "mom12m_jt", "maxret_bcw", "accruals_sloan", "idiovola_clmx",
            "gp_novymarx", "overnight_intraday_spread", "skew21_lottery", "pvcorr_21",
            "kurt21_returns", "coskew60", "hl_52w_disposition", "resmom_6m"]

log("loading v33...")
v33 = pl.read_parquet(V33)
log(f"  v33: {v33.shape}")
n_rows = v33.height

log("loading new factors...")
newf = pl.read_parquet(NEWF)
log(f"  new: {newf.shape}")

# sanity: same key set
assert v33.height == newf.height

# cast dates
v33 = v33.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
newf = newf.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))

# talib coverage check
v33_talib = [c for c in v33.columns if c.startswith("talib_")]
newf_talib = [c for c in newf.columns if c.startswith("talib_")]
missing_talib = [c for c in v33_talib if c not in newf_talib]
log(f"  talib: v33={len(v33_talib)}, new={len(newf_talib)}, missing_in_new={missing_talib}")
if missing_talib:
    raise RuntimeError(f"missing talib cols in new: {missing_talib}")

# build replace set
families_replace = [c for c in v33.columns if c.startswith("gtja_") or c.startswith("alpha_")
                    or c.startswith("talib_")]
replace_names = families_replace + ACADEMIC_REBUILD + GITHUB17
# guard: all must exist in newf
missing_rep = [c for c in replace_names if c not in newf.columns]
log(f"  replace set: {len(replace_names)} cols; missing in newf: {missing_rep}")
if missing_rep:
    raise RuntimeError(f"replace cols missing: {missing_rep}")

# ── merge adj factor for adj OHLC cols ──
log("merging adj_factor...")
adj = pl.read_parquet(ADJ, columns=["ts_code", "trade_date", "adj_factor"])
adj = adj.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
v33 = v33.join(adj.rename({"adj_factor": "_adj_factor_new"}), on=["ts_code", "trade_date"], how="left")
n_null = v33.filter(pl.col("_adj_factor_new").is_null()).height
log(f"  adj_factor null after merge: {n_null}")
v33 = v33.sort(["ts_code", "trade_date"]).with_columns(
    pl.col("_adj_factor_new").forward_fill().backward_fill().over("ts_code")
)

# ── prepare new frame's replace columns (add row index for order restore) ──
v33 = v33.with_row_index("_ri").sort(["ts_code", "trade_date"])
newf_sub = newf.select(["ts_code", "trade_date"] + replace_names)

# drop old columns and join new
log("replacing factor columns...")
v34 = v33.drop(replace_names + ["adj_close", "adj_factor"], strict=False)
v34 = v34.join(newf_sub, on=["ts_code", "trade_date"], how="left")
log(f"  after replace-join: {v34.shape}")
n_null_after = v34.select(replace_names[:5]).null_count().to_dicts()
log(f"  null spot check: {n_null_after}")

# add adj price columns
v34 = v34.with_columns([
    pl.col("_adj_factor_new").alias("adj_factor"),
    (pl.col("close") * pl.col("_adj_factor_new")).alias("adj_close"),
    (pl.col("open") * pl.col("_adj_factor_new")).alias("adj_open"),
    (pl.col("high") * pl.col("_adj_factor_new")).alias("adj_high"),
    (pl.col("low") * pl.col("_adj_factor_new")).alias("adj_low"),
])

# restore original row order
v34 = v34.sort("_ri").drop(["_ri", "_adj_factor_new"])
log(f"  final: {v34.shape}")

# ── column count sanity ──
expected = len(v33.columns) - 1 - 2 + 5  # -_ri -adj_close -adj_factor +5 new
# simpler: 461 = 458 + 3 (adj_open/high/low; adj_factor replaced; adj_close replaced)
log(f"  expected cols ~461, got {v34.shape[1]}")

# ── verification: changed-columns summary ──
log("verification: comparing v34 vs v33 on replaced columns (sample)...")
v33_old = pl.read_parquet(V33, columns=["ts_code", "trade_date"] + replace_names)
v33_old = v33_old.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
cmp_full = (v34.select(["ts_code", "trade_date"] + replace_names)
            .join(v33_old, on=["ts_code", "trade_date"], how="inner", suffix="_old"))
sample_idx = np.sort(np.random.RandomState(42).choice(cmp_full.height, size=min(300_000, cmp_full.height), replace=False))
cmp = cmp_full.gather(sample_idx)
changed = []
for c in replace_names:
    both = cmp.filter(pl.col(c).is_not_null() & pl.col(c + "_old").is_not_null())
    if both.height == 0:
        continue
    d = (both[c] - both[c + "_old"]).abs()
    n_diff = (d > 1e-6).sum()
    if n_diff > both.height * 0.001:  # >0.1% of sample differ
        changed.append({"col": c, "n_diff": int(n_diff), "pct": round(n_diff / both.height * 100, 2),
                        "max_abs_diff": float(d.max())})
log(f"  columns with material change: {len(changed)} / {len(replace_names)}")

# family-level summary
def fam(c):
    if c.startswith("gtja_"): return "gtja"
    if c.startswith("alpha_"): return "alpha"
    if c.startswith("talib_"): return "talib"
    if c in GITHUB17: return "github"
    return "academic"
from collections import Counter, defaultdict
fam_total = Counter(fam(c) for c in replace_names)
fam_changed = Counter(fam(x["col"]) for x in changed)
log("  per-family changed:")
for f in fam_total:
    log(f"    {f}: {fam_changed.get(f,0)}/{fam_total[f]} changed")

# ── write ──
log(f"writing {OUT}...")
v34.write_parquet(OUT)
log(f"  saved {OUT.stat().st_size/1e9:.2f} GB")

summary = {
    "v34_shape": list(v34.shape),
    "v33_shape": [n_rows, 458],
    "replaced_cols": len(replace_names),
    "material_changed_cols": len(changed),
    "changed_by_family": {f: {"changed": fam_changed.get(f, 0), "total": fam_total[f]} for f in fam_total},
    "changed_top30": sorted(changed, key=lambda x: -x["pct"])[:30],
}
(OUTDIR / "part2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log("DONE part 2")
