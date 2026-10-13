"""H3: 量能 surprise alpha (AQR 2024).

公式: vol_z = (vol - rolling_mean(vol, 20)) / rolling_std(vol, 20)
      ret_sign = sign(close.pct_change())
      factor = vol_z * ret_sign

- 经济学: 高量+正收益 = 强势买入 (continuation); 高量+负收益 = 强势卖出
- causal: rolling 用过去 20 日 (含当日)
- 步骤: 实现 → 归因审计 (vs 413 库) → 单因子 OOS sanity
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
DEDUP = PANEL
OUT = AKQ / "evidence" / "h3_vol_surprise_20261013"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 1. 计算因子 ----
t0 = time.time()
df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "vol", "adj_close"])
df = df.sort(["ts_code", "trade_date"])
df = df.with_columns([
    pl.col("vol").rolling_mean(20, min_periods=10).over("ts_code").alias("vol_mean_20"),
    pl.col("vol").rolling_std(20, min_periods=10).over("ts_code").alias("vol_std_20"),
    pl.col("adj_close").pct_change().over("ts_code").alias("ret"),
])
df = df.with_columns([
    ((pl.col("vol") - pl.col("vol_mean_20")) / pl.col("vol_std_20")).alias("vol_z"),
])
df = df.with_columns([
    pl.when(pl.col("ret") > 0).then(1).when(pl.col("ret") < 0).then(-1).otherwise(0).alias("ret_sign"),
])
df = df.with_columns([
    (pl.col("vol_z").fill_null(0.0) * pl.col("ret_sign")).alias("qp_vol_surp"),
    # 变体 2: signed magnitude
    (pl.col("vol_z").fill_null(0.0) * pl.col("ret").fill_null(0.0)).alias("qp_vol_surp_mag"),
    # 变体 3: pure vol z (no sign)
    pl.col("vol_z").alias("qp_vol_z"),
])
out = df.select(["trade_date", "ts_code", "qp_vol_surp", "qp_vol_surp_mag", "qp_vol_z"])
out.write_parquet(OUT / "factor.parquet")
log(f"factor built {time.time()-t0:.0f}s | rows {len(out)} | coverage "
    f"{out['qp_vol_surp'].is_not_null().mean()*100:.1f}%")

# ---- 2. 初筛 (decile-spread) ----
t0 = time.time()
px = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
fwd = (px.sort(["ts_code", "trade_date"])
       .with_columns(fwd=pl.col("adj_close").shift(-21).over("ts_code")
                          / pl.col("adj_open").shift(-1).over("ts_code") - 1.0 - 0.005)
       .select(["trade_date", "ts_code", "fwd"]))
j = out.join(fwd, on=["trade_date", "ts_code"], how="left")
j = j.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
pdf = j.to_pandas()
pdf["dstr"] = pd.to_datetime(pdf["trade_date"]).dt.strftime("%Y-%m-%d")

screen_rows = []
for name in ["qp_vol_surp", "qp_vol_surp_mag", "qp_vol_z"]:
    sp_map = {}
    for d, sub in pdf.groupby("dstr", sort=False):
        v = sub[name].to_numpy()
        r = sub["fwd"].to_numpy()
        m = np.isfinite(v) & np.isfinite(r)
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        vals, rs = v[idx], r[idx]
        order = np.argsort(-vals)
        sp_map[d] = rs[order[:10]].mean() - rs[order[-10:]].mean()
    sps = pd.Series(sp_map).sort_index()
    sub1 = sps.loc[:"2017-12-31"]
    sub2 = sps.loc["2018-01-01":]

    def stat(x):
        x = x.dropna()
        if len(x) < 30:
            return (np.nan, np.nan, np.nan)
        return (x.mean(), x.mean() / x.std() * np.sqrt(len(x)), (x > 0).mean())
    a = stat(sps); b1 = stat(sub1); b2 = stat(sub2)
    screen_rows.append({"factor": name, "mean_bps": a[0]*1e4, "t": a[1], "hit": a[2],
                        "mean_1017_bps": b1[0]*1e4, "t_1017": b1[1],
                        "mean_1825_bps": b2[0]*1e4, "t_1825": b2[1]})
    log(f"SCREEN {name:18s}  {a[0]*1e4:6.1f}bps t={a[1]:5.2f} hit={a[2]:.0%}  "
        f"| 10-17 {b1[0]*1e4:6.1f} t={b1[1]:5.2f}  | 18-25 {b2[0]*1e4:6.1f} t={b2[1]:5.2f}")
pd.DataFrame(screen_rows).to_csv(OUT / "screen.csv", index=False)
log(f"screen done {time.time()-t0:.0f}s")

# ---- 3. 归因审计 (vs 413 库) ----
t0 = time.time()
col_info = pl.read_parquet_schema(DEDUP)
SKIP = {"trade_date", "ts_code", "symbol", "open", "high", "low", "close", "vol", "amount",
        "pct_chg", "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
        "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60", "turnover_rate",
        "circ_cap", "cap", "volume", "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
lib_factors = [c for c in col_info.keys() if c not in SKIP and not c.startswith("v10_1_")]
new_names = ["qp_vol_surp", "qp_vol_surp_mag", "qp_vol_z"]
sample_dates = pd.to_datetime(np.sort(pdf["dstr"].unique())[::20])
samp = pl.Series("d", sample_dates).cast(pl.Datetime("ms"))
lib_df = (pl.scan_parquet(DEDUP).filter(pl.col("trade_date").is_in(samp))
          .select(["trade_date", "ts_code"] + lib_factors).collect().to_pandas())
lib_df["trade_date"] = pd.to_datetime(lib_df["trade_date"])
log(f"audit lib rows: {len(lib_df)}")

acc = {n: [] for n in new_names}
for d in sample_dates:
    a = pdf[pdf["dstr"] == d.strftime("%Y-%m-%d")]
    bdf = lib_df[lib_df["trade_date"] == d]
    if len(a) < 30 or len(bdf) < 30:
        continue
    m = a.merge(bdf, on="ts_code")
    if len(m) < 30:
        continue
    R = m[new_names + lib_factors].rank()
    Cm = R.corr()
    for n in new_names:
        acc[n].append(Cm.loc[n, lib_factors])
for n in new_names:
    if not acc[n]:
        continue
    dfa = pd.DataFrame(acc[n])
    mc = dfa.abs().mean(axis=0).sort_values(ascending=False)
    top5 = mc.head(5)
    txt = ", ".join(f"{f}={dfa[f].mean():+.2f}" for f in top5.index)
    log(f"AUDIT {n:18s} max|corr|={top5.iloc[0]:.3f} | {txt}")
log(f"audit done {time.time()-t0:.0f}s")
