"""v37 matrix: industry z-scored factor spreads.

对每个因子, 每个日期做行业内 z-score, 然后用 z-scored 值选 top10/bot10
计算 fwd21 spread. 新 M 列名与 v34 相同 (413 因子), 但 spread 基于行业中性值.
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
DEDUP = ROOT / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
INDUSTRY = ROOT / "data" / "stock_basic_full.parquet"
MAT34 = ROOT / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
MAT37 = ROOT / "evidence" / "v34_dedup_20261010" / "matrix_v37_indzscore_ADJ.parquet"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Load industry map
ind_df = pd.read_parquet(INDUSTRY)[["ts_code", "industry"]]
ind_map = dict(zip(ind_df["ts_code"], ind_df["industry"]))
log(f"industry map: {len(ind_map)}")

# Load v34 matrix
mat34 = pl.read_parquet(MAT34)
factors = [c for c in mat34.columns if c != "trade_date"]
log(f"factors: {len(factors)}")

# Load fwd returns (computed in same way as v34)
t0 = time.time()
df = (pl.read_parquet(DEDUP, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
      .sort(["ts_code", "trade_date"])
      .with_columns(fwd=pl.col("adj_close").shift(-21).over("ts_code")
                          / pl.col("adj_open").shift(-1).over("ts_code") - 1.0 - 0.005)
      .select(["trade_date", "ts_code", "fwd"]))
log(f"fwd panel: {time.time()-t0:.0f}s")

# Build wide index
T = len(df["trade_date"].unique())
stocks_arr = np.sort(df["ts_code"].unique().to_numpy())
S = len(stocks_arr)
date_to_idx = {d: i for i, d in enumerate(np.sort(df["trade_date"].unique().to_numpy()))}
stock_to_idx = {s: i for i, s in enumerate(stocks_arr)}
log(f"T={T} S={S}")

ind_arr = np.array([ind_map.get(s, "未知") for s in stocks_arr])
unique_inds = list(set(ind_arr))
log(f"industries: {len(unique_inds)}")

# fwd wide (T x S)
fwd_wide = np.full((T, S), np.nan)
fwd_pdf = df.to_pandas().dropna(subset=["fwd"])
fwd_pdf = fwd_pdf.assign(
    di=fwd_pdf["trade_date"].map(date_to_idx),
    si=fwd_pdf["ts_code"].map(stock_to_idx),
)
fwd_wide[fwd_pdf["di"].values, fwd_pdf["si"].values] = fwd_pdf["fwd"].values
log(f"fwd wide built")

# Main loop
t0 = time.time()
M_new = np.full((T, len(factors)), np.nan)
NIND_MIN = 2

# Read all factor values in one go (memory ~ 413 * 1.3M * 8B = 4.3GB - too big)
# Process in batches of 50
BATCH = 50
for fi_start in range(0, len(factors), BATCH):
    fi_end = min(fi_start + BATCH, len(factors))
    batch_factors = factors[fi_start:fi_end]
    # Read batch
    df_batch = pl.read_parquet(DEDUP, columns=["trade_date", "ts_code"] + batch_factors)
    # For each factor in batch, build wide and compute z-scored spread
    for bi, f in enumerate(batch_factors):
        fi = fi_start + bi
        sub = df_batch.select(["trade_date", "ts_code", f]).to_pandas()
        sub = sub.dropna(subset=[f]).assign(
            di=sub["trade_date"].map(date_to_idx),
            si=sub["ts_code"].map(stock_to_idx),
        )
        wide = np.full((T, S), np.nan)
        wide[sub["di"].values, sub["si"].values] = sub[f].values
        # z-score per (date, industry)
        out = np.zeros_like(wide)
        for ind in unique_inds:
            idx = np.where(ind_arr == ind)[0]
            if len(idx) < NIND_MIN:
                continue
            sub_v = wide[:, idx]
            with np.errstate(all="ignore"):
                mean = np.nanmean(sub_v, axis=1, keepdims=True)
                std = np.nanstd(sub_v, axis=1, keepdims=True)
            std = np.where(np.isnan(std) | (std < 1e-10), 1, std)
            out[:, idx] = (sub_v - mean) / std
        out = np.where(np.isnan(out) | np.isinf(out), 0, out)
        # Decile spread
        order = np.argsort(-out, axis=1)
        top10 = order[:, :10]
        bot10 = order[:, -10:]
        fwd_top = np.take_along_axis(fwd_wide, top10, axis=1)
        fwd_bot = np.take_along_axis(fwd_wide, bot10, axis=1)
        fwd_top_mean = np.nanmean(fwd_top, axis=1)
        fwd_bot_mean = np.nanmean(fwd_bot, axis=1)
        valid = np.isfinite(fwd_top_mean) & np.isfinite(fwd_bot_mean)
        M_new[valid, fi] = fwd_top_mean[valid] - fwd_bot_mean[valid]
    log(f"  {fi_end}/{len(factors)} factors done {time.time()-t0:.0f}s")

# Save
t0 = time.time()
out_pl = mat34.with_columns([pl.Series(f, M_new[:, fi]) for fi, f in enumerate(factors)])
out_pl.write_parquet(MAT37)
log(f"saved {MAT37} {time.time()-t0:.0f}s")
