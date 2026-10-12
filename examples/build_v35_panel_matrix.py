"""v35: 合并 3 个新因子 (qp_shadow_up / qp_shadow_lo / qp_str) 到面板与矩阵.

- 面板: data/wavehunter_hs300_v35_20261012.parquet (dedup + 3 cols, polars streaming)
- 矩阵: evidence/v34_dedup_20261010/matrix_v35_ADJ.parquet (v34 矩阵 + 3 新因子 spread, 同口径)
- 池: evidence/v5_dynamic_select_20261011/run_factsets/v5_pool416.json
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
DEDUP = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
B1 = AKQ / "evidence" / "v35_new_factors_20261012" / "new_factors_b1.parquet"
B2 = AKQ / "evidence" / "v35_new_factors_20261012" / "new_factors_b2.parquet"
PANEL_V35 = AKQ / "data" / "wavehunter_hs300_v35_20261012.parquet"
MAT34 = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
MAT35 = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v35_ADJ.parquet"
FS = AKQ / "evidence" / "v5_dynamic_select_20261011" / "run_factsets" / "v5_pool416.json"

NEW = ["qp_shadow_up", "qp_shadow_lo", "qp_str"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 1. 面板 v35 (streaming join, 可重复执行) ----
t0 = time.time()
n1 = pl.read_parquet(B1, columns=["trade_date", "ts_code", "qp_shadow_up", "qp_shadow_lo"])
n2 = pl.read_parquet(B2, columns=["trade_date", "ts_code", "qp_str"])
new = n1.join(n2, on=["trade_date", "ts_code"], how="full", coalesce=True)
(pl.scan_parquet(DEDUP)
 .join(pl.DataFrame(new).lazy(), on=["trade_date", "ts_code"], how="left")
 .sink_parquet(str(PANEL_V35)))
log(f"panel v35 written {time.time()-t0:.0f}s")

# ---- 2. 矩阵 v35 ----
t0 = time.time()
mat34 = pl.read_parquet(MAT34)
base_factors = [c for c in mat34.columns if c != "trade_date"]
date_strs = [str(d)[:10] for d in mat34["trade_date"].to_list()]

px = (pl.read_parquet(DEDUP, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
      .sort(["ts_code", "trade_date"])
      .with_columns(fwd=pl.col("adj_close").shift(-21).over("ts_code")
                    / pl.col("adj_open").shift(-1).over("ts_code") - 1.0 - 0.005)
      .select(["trade_date", "ts_code", "fwd"]))
nf = pl.read_parquet(PANEL_V35, columns=["trade_date", "ts_code"] + NEW)
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
out.write_parquet(MAT35)
log(f"matrix v35 written {time.time()-t0:.0f}s ({out.shape})")

# ---- 3. 池 v5_pool416 ----
pool = base_factors + NEW
FS.write_text(json.dumps({"tag": "v5_pool416", "factors": pool,
                          "reason": "413 dedup + 3 novel QP factors (shadow_up/shadow_lo/str)"},
                         ensure_ascii=False, indent=1))
log(f"pool416: {len(pool)} factors -> {FS}")
