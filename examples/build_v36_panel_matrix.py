"""v36: 合并 qp_vol_surp 到面板与矩阵 (单变量, smoke test)."""
import json
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
DEDUP = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
H3 = AKQ / "evidence" / "h3_vol_surprise_20261013" / "factor.parquet"
PANEL_V36 = AKQ / "data" / "wavehunter_hs300_v36_20261013.parquet"
MAT34 = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
MAT36 = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v36_ADJ.parquet"
FS = AKQ / "evidence" / "v5_dynamic_select_20261011" / "run_factsets" / "v5_pool414.json"

NEW = ["qp_vol_surp"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 1. 面板 v36 ----
t0 = time.time()
n3 = pl.read_parquet(H3, columns=["trade_date", "ts_code"] + NEW)
(pl.scan_parquet(DEDUP)
 .join(pl.DataFrame(n3).lazy(), on=["trade_date", "ts_code"], how="left")
 .sink_parquet(str(PANEL_V36)))
log(f"panel v36 written {time.time()-t0:.0f}s")

# ---- 2. 矩阵 v36 ----
t0 = time.time()
mat34 = pl.read_parquet(MAT34)
base_factors = [c for c in mat34.columns if c != "trade_date"]
date_strs = [str(d)[:10] for d in mat34["trade_date"].to_list()]

px = (pl.read_parquet(DEDUP, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
      .sort(["ts_code", "trade_date"])
      .with_columns(fwd=pl.col("adj_close").shift(-21).over("ts_code")
                    / pl.col("adj_open").shift(-1).over("ts_code") - 1.0 - 0.005)
      .select(["trade_date", "ts_code", "fwd"]))
nf = pl.read_parquet(PANEL_V36, columns=["trade_date", "ts_code"] + NEW)
j = nf.join(px, on=["trade_date", "ts_code"], how="left")
pdf = j.to_pandas()
pdf["dstr"] = pd.to_datetime(pdf["trade_date"]).dt.strftime("%Y-%m-%d")

sp_map = {n: {} for n in NEW}
for d, sub in pdf.groupby("dstr", sort=False):
    fwd = sub["fwd"].to_numpy()
    for name in NEW:
        v = sub[name].to_numpy()
        m = np.isfinite(v) & np.isfinite(fwd)
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        vals, rs = v[idx], fwd[idx]
        order = np.argsort(-vals)
        sp_map[name][d] = rs[order[:10]].mean() - rs[order[-10:]].mean()

out = mat34
for name in NEW:
    col = [sp_map[name].get(d, float("nan")) for d in date_strs]
    out = out.with_columns(pl.Series(name, col))
out.write_parquet(MAT36)
log(f"matrix v36 written {time.time()-t0:.0f}s ({out.shape})")

# ---- 3. 池 v5_pool414 ----
pool = base_factors + NEW
FS.write_text(json.dumps({"tag": "v5_pool414", "factors": pool,
                          "reason": "413 dedup + qp_vol_surp (volume-confirmed momentum, AQR 2024, novel in A-shares)"},
                         ensure_ascii=False, indent=1))
log(f"pool414: {len(pool)} factors -> {FS}")
