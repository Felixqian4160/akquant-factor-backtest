"""Part 2 (fixed): input-level rank correlation between stockdb-v4 inputs and v34 inputs.

For each key raw input, per-date cross-sectional rank corr on matched rows.
This upper-bounds the factor agreement for factors driven by that input.
Also: verify v34's p_pr semantics (log raw close?) and alpha100 anomaly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE / "factorlib_build"))

V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"

# ---- 0. p_pr semantics check on v34 ----
chk = pl.read_parquet(V34, columns=["ts_code", "trade_date", "close", "adj_close", "p_pr"]).filter(
    pl.col("ts_code") == "000001.SZ"
).tail(5)
chk = chk.with_columns([
    np.log(chk["close"]).alias("log_raw"),
    np.log(chk["adj_close"]).alias("log_adj"),
])
print("=== v34 p_pr semantics ===")
print(chk.select(["trade_date", "close", "adj_close", "p_pr", "log_raw", "log_adj"]))
d_raw = (chk["p_pr"] - chk["log_raw"]).abs().max()
d_adj = (chk["p_pr"] - chk["log_adj"]).abs().max()
print(f"max |p_pr - log(raw)| = {d_raw:.6f};  max |p_pr - log(adj)| = {d_adj:.6f}")

# ---- 1. build sample dates ----
k4 = pl.read_parquet(FSDB, columns=["trade_date"]).unique()
k4 = k4.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"))
dates = sorted(k4["d"].unique().to_list())
sample_dates = [dates[int(i)] for i in np.linspace(0, len(dates) - 1, 30)]

# ---- 2. v4 inputs ----
f4 = pl.read_parquet(
    FSDB,
    columns=["ts_code", "trade_date", "open_hfq", "high_hfq", "low_hfq", "close_hfq",
             "adj_factor_hfq", "cum_latest", "volume", "amount", "total_mv", "float_mv", "pe_ttm", "pb"],
).with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d")).drop("trade_date")
f4 = f4.with_columns([
    (pl.col("close_hfq") * pl.col("cum_latest")).alias("close_adj"),
    (pl.col("high_hfq") * pl.col("cum_latest")).alias("high_adj"),
    (pl.col("low_hfq") * pl.col("cum_latest")).alias("low_adj"),
    (pl.col("open_hfq") * pl.col("cum_latest")).alias("open_adj"),
    (pl.col("volume") / 100.0).alias("vol_lots"),
    (pl.col("amount") / 1000.0).alias("amount_k"),
    (pl.col("total_mv") / 1e4).alias("cap"),
    (pl.col("float_mv") / 1e4).alias("circ_cap"),
])
f4 = f4.filter(pl.col("d").is_in(sample_dates))

# ---- 3. v34 inputs ----
# NOTE: v34 stores no `vwap` column (it was only used internally during the v34
# build), so the comparison covers the stored inputs only.
f3 = pl.read_parquet(
    V34,
    columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
             "cap", "circ_cap", "pe", "adj_open", "adj_high", "adj_low", "adj_close",
             "adj_factor"],
).with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")
f3 = f3.with_columns([
    pl.col("adj_close").alias("close_adj"),
    pl.col("adj_high").alias("high_adj"),
    pl.col("adj_low").alias("low_adj"),
    pl.col("adj_open").alias("open_adj"),
    pl.col("vol").alias("vol_lots"),
    (pl.col("amount") / 1.0).alias("amount_k"),  # v34 already 千元
]).filter(pl.col("d").is_in(sample_dates))

PAIRS = [
    ("close_adj", "close_adj"), ("high_adj", "high_adj"), ("low_adj", "low_adj"),
    ("open_adj", "open_adj"), ("vol_lots", "vol_lots"), ("amount_k", "amount_k"),
    ("cap", "cap"), ("circ_cap", "circ_cap"), ("pe_ttm", "pe"),
]
j = f4.join(f3, on=["ts_code", "d"], how="inner", suffix="_34")

print()
print("=== input-level per-date rank corr (v4 stockdb vs v34 tushare) ===")
print(f"{'input':<12} {'mean_rho':>9} {'min_rho':>9} {'n_dates':>8}")
for a_col, b_col in PAIRS:
    # f3 columns colliding with f4 got the "_34" suffix; non-colliding keep the name
    b_name = f"{b_col}_34" if f"{b_col}_34" in j.columns else b_col
    rhos = []
    for d in sample_dates:
        sub = j.filter(pl.col("d") == d)
        if sub.height < 50:
            continue
        a = sub[a_col].to_numpy().astype(float)
        b = sub[b_name].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        ra = a[ok].argsort().argsort()
        rb = b[ok].argsort().argsort()
        if ra.std() == 0 or rb.std() == 0:
            continue
        rhos.append(np.corrcoef(ra, rb)[0, 1])
    if rhos:
        print(f"{a_col:<12} {np.mean(rhos):>9.4f} {np.min(rhos):>9.4f} {len(rhos):>8}")
    else:
        print(f"{a_col:<12} {'--':>9} {'--':>9} {0:>8}")

# ---- 4. adjustment factor dispersion per stock (v4 cum_t vs v34 adj) ----
print()
print("=== adjustment scale ratio: v4 (cum_t) / v34 (adj_factor) distribution ===")
adj4 = pl.read_parquet(FSDB, columns=["ts_code", "trade_date", "adj_factor_hfq", "cum_latest"])
adj4 = adj4.with_columns((pl.col("adj_factor_hfq") * pl.col("cum_latest")).alias("cum_t"))
adj4 = adj4.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d")).drop("trade_date")
adj3 = pl.read_parquet(V34, columns=["ts_code", "trade_date", "adj_factor"])
adj3 = adj3.with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")
aj = adj4.join(adj3, on=["ts_code", "d"], how="inner").filter(pl.col("d") == sample_dates[15])
aj = aj.with_columns((pl.col("cum_t") / pl.col("adj_factor")).alias("ratio"))
print(aj.select([
    pl.col("ratio").median().alias("median"),
    pl.col("ratio").quantile(0.1).alias("p10"),
    pl.col("ratio").quantile(0.9).alias("p90"),
    pl.col("ratio").min().alias("min"),
    pl.col("ratio").max().alias("max"),
]))

# ---- 5. alpha100 anomaly ----
print()
print("=== alpha_alpha100 on sample dates (v4) ===")
a100 = (
    pl.read_parquet(
        BASE / "data" / "wavehunter_hs300_fsdb_v4_20261009_235409.parquet",
        columns=["ts_code", "trade_date", "alpha_alpha100"],
    )
    .with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"))
)
for d in sample_dates[:8]:
    sub = a100.filter(pl.col("d") == d)
    nn = sub.height - sub["alpha_alpha100"].null_count()
    nu = sub["alpha_alpha100"].n_unique()
    print(f"  {d}  rows={sub.height} non_null={nn} unique={nu}")
