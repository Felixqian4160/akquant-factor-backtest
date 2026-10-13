"""H4 augment: 在 top-15 之外, 额外加 3 只"昨日涨停"的股票.

机制: 涨停是 +246bps 20d OOS momentum. 框架自然只选 26% grid 含 LU.
       强制 augment 到 18 只/grid (15 + 3 LU), 验证能否提升 ratio.
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
PICKS_IN = AKQ / "evidence/v3_resonance_20261010/picks/V5_P413_RET_RANK_o0/picks.json"
OUT = AKQ / "evidence/h4_limitup_20261013"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 1. 检测 LU 事件 ----
t0 = time.time()
df = (pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close", "adj_high"])
      .sort(["ts_code", "trade_date"])
      .with_columns(pl.col("adj_close").shift(1).over("ts_code").alias("prev_close"))
      .with_columns(((pl.col("adj_high") - pl.col("prev_close")) / pl.col("prev_close")).alias("gap"))
      .with_columns((pl.col("gap") >= 0.095).fill_null(False).alias("lu")))
df_pd = df.select(["trade_date", "ts_code", "gap", "lu"]).to_pandas()
df_pd["dstr"] = pd.to_datetime(df_pd["trade_date"]).dt.strftime("%Y-%m-%d")
log(f"lu data {time.time()-t0:.0f}s")

# lu_by_date[d] = list of (ts_code, gap) sorted desc by gap
lu_by_date = {}
for d, sub in df_pd.groupby("dstr", sort=False):
    lu_sub = sub[sub["lu"]].sort_values("gap", ascending=False)
    lu_by_date[d] = lu_sub[["ts_code", "gap"]].values.tolist()
log(f"lu_by_date built for {len(lu_by_date)} dates")

# ---- 2. Augment picks ----
t0 = time.time()
picks = json.loads(PICKS_IN.read_text())
all_dates = sorted(set(picks.keys()) | set(lu_by_date.keys()))
date_to_idx = {d: i for i, d in enumerate(all_dates)}

N_ADD = 3
augmented = {}
for d, stocks in picks.items():
    new_picks = list(stocks)
    prev_idx = date_to_idx[d] - 1
    if prev_idx >= 0:
        prev = all_dates[prev_idx]
        if prev in lu_by_date:
            for s, gap in lu_by_date[prev]:
                if s not in new_picks:
                    new_picks.append(s)
                    if len(new_picks) >= 15 + N_ADD:
                        break
    augmented[d] = new_picks
log(f"augmented {time.time()-t0:.0f}s")

# 统计
n_orig = sum(len(v) for v in picks.values())
n_aug = sum(len(v) for v in augmented.values())
n_grids = len(picks)
n_grids_with_add = sum(1 for d, v in augmented.items() if len(v) > 15)
log(f"grids: {n_grids} | augmented: {n_orig} -> {n_aug} (added {n_aug - n_orig} stocks across {n_grids_with_add} grids)")

# 过滤 2018+
aug_2018 = {k: v for k, v in augmented.items() if k >= "2018-01-01"}
sub_dir = OUT / "H4_augment_3lu"
sub_dir.mkdir(parents=True, exist_ok=True)
(sub_dir / "picks.json").write_text(json.dumps(aug_2018, ensure_ascii=False, indent=1))
(sub_dir / "picks_meta.json").write_text(
    (AKQ / "evidence/v3_resonance_20261010/picks/V5_P413_RET_RANK_o0/picks_meta.json").read_text())
log(f"saved {sub_dir}")
