"""批次1 因子归因审计: 与现有 413 库的截面秩相关 (查"重包装的旧因子")."""
import pathlib

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
DEDUP = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
NEW = AKQ / "evidence" / "v35_new_factors_20261012" / "new_factors_b1.parquet"

new = pl.read_parquet(NEW).to_pandas()
new["trade_date"] = pd.to_datetime(new["trade_date"])
new_factors = [c for c in new.columns if c not in ("trade_date", "ts_code")]

col_info = pl.read_parquet_schema(DEDUP)
SKIP = {
    "trade_date", "ts_code", "symbol", "open", "high", "low", "close", "vol", "amount",
    "pct_chg", "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
    "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60", "turnover_rate",
    "circ_cap", "cap", "volume",
    "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
}
lib_factors = [c for c in col_info.keys() if c not in SKIP and not c.startswith("v10_1_")]
print(f"library factors: {len(lib_factors)}")

dates_all = np.sort(new["trade_date"].unique())
sample_dates = pd.to_datetime(dates_all[::20])
print(f"sample dates: {len(sample_dates)}")

samp = pl.Series("d", sample_dates).cast(pl.Datetime("ms"))
lib_df = (pl.scan_parquet(DEDUP)
          .filter(pl.col("trade_date").is_in(samp))
          .select(["trade_date", "ts_code"] + lib_factors)
          .collect().to_pandas())
lib_df["trade_date"] = pd.to_datetime(lib_df["trade_date"])
print(f"lib rows loaded: {len(lib_df)}")

acc = {name: [] for name in new_factors}
n_used = 0
for d in sample_dates:
    a = new[new["trade_date"] == d]
    b = lib_df[lib_df["trade_date"] == d]
    if len(a) < 30 or len(b) < 30:
        continue
    m = a.merge(b, on="ts_code", suffixes=("", "_lib"))
    if len(m) < 30:
        continue
    R = m[new_factors + lib_factors].rank()
    C = R.corr()  # (10+413)^2 pairwise
    for name in new_factors:
        acc[name].append(C.loc[name, lib_factors])
    n_used += 1

print(f"dates used: {n_used}\n")
print(f"{'factor':16s} {'|corr|max':>9s}  top matches (mean signed corr)")
for name in new_factors:
    if not acc[name]:
        continue
    df = pd.DataFrame(acc[name])
    mc = df.abs().mean(axis=0).sort_values(ascending=False)
    top5 = mc.head(5)
    txt = ", ".join(f"{f}={df[f].mean():+.2f}" for f in top5.index)
    print(f"{name:16s} {top5.iloc[0]:9.3f}  {txt}")
