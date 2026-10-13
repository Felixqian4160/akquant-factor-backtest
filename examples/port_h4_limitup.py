"""H4: 涨停敢死队隔日反转.

机制: 上一日触及涨停的股票, 次日 (T+1) 容易反转 (论文 + A 股 retail 经验)
数据: panel v34 adj_close, adj_high (因果: 涨停检测用前 N 日数据)
测试: 0) 诊断 涨停在 picks 中频率  1) 排除"昨日涨停"股票 → sim
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
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 1. 检测涨停 ----
t0 = time.time()
df = (pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close", "adj_high"])
      .sort(["ts_code", "trade_date"]))
# prev close
df = df.with_columns([
    pl.col("adj_close").shift(1).over("ts_code").alias("prev_close"),
])
# 主板 10% 涨停 (允许 0.5% wiggle, 因复权会有 0.1% 误差)
df = df.with_columns([
    pl.when(pl.col("prev_close") > 0)
      .then(pl.col("adj_high") / pl.col("prev_close") - 1.0)
      .otherwise(None).alias("gap_to_10pct"),
    pl.when(pl.col("prev_close") > 0)
      .then(pl.col("adj_close") / pl.col("prev_close") - 1.0)
      .otherwise(None).alias("ret1d"),
])
df = df.with_columns([
    (pl.col("gap_to_10pct") >= 0.095).alias("lu_yesterday"),
    (pl.col("gap_to_10pct") >= 0.195).alias("lu_chinext_yesterday"),
])
log(f"limit-up detected {time.time()-t0:.0f}s")
lu_count = df.filter(pl.col("lu_yesterday")).height
lu_chinext_count = df.filter(pl.col("lu_chinext_yesterday")).height
total = df.height
log(f"10% limit-up events: {lu_count}/{total} ({lu_count/total*100:.2f}%)")
log(f"20% limit-up events: {lu_chinext_count}/{total} ({lu_chinext_count/total*100:.2f}%)")

# ---- 2. 诊断: 涨停在 picks 中的频率 ----
t0 = time.time()
df_pd = df.select(["trade_date", "ts_code", "lu_yesterday", "lu_chinext_yesterday"]).to_pandas()
df_pd["dstr"] = pd.to_datetime(df_pd["trade_date"]).dt.strftime("%Y-%m-%d")
lu_map = {}
for d, sub in df_pd.groupby("dstr", sort=False):
    lu_map[d] = set(sub[sub["lu_yesterday"] | sub["lu_chinext_yesterday"]]["ts_code"])
log(f"lu_map built {time.time()-t0:.0f}s")

picks = json.loads(PICKS_IN.read_text())
picks_2018 = {k: v for k, v in picks.items() if k >= "2018-01-01"}
n_total = 0
n_lu = 0
n_lu_strict = 0  # at least 1 LU stock in the pick
for d, stocks in picks_2018.items():
    lu_set = lu_map.get(d, set())
    for s in stocks:
        if s in lu_set:
            n_lu += 1
    if any(s in lu_set for s in stocks):
        n_lu_strict += 1
    n_total += len(stocks)
log(f"\n=== 涨停 in picks 2018+ ===")
log(f"  total pick slots: {n_total}")
log(f"  LU stocks: {n_lu} ({n_lu/n_total*100:.2f}%)")
log(f"  grids with >=1 LU: {n_lu_strict}/{len(picks_2018)} ({n_lu_strict/len(picks_2018)*100:.2f}%)")

# ---- 3. 涨停次日 forward return (单变量 spread) ----
t0 = time.time()
df_fwd = (pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
          .sort(["ts_code", "trade_date"])
          .with_columns(fwd=pl.col("adj_close").shift(-21).over("ts_code")
                              / pl.col("adj_open").shift(-1).over("ts_code") - 1.0 - 0.005)
          .select(["trade_date", "ts_code", "fwd"]))
j = df.select(["trade_date", "ts_code", "lu_yesterday", "lu_chinext_yesterday"]).join(
    df_fwd, on=["trade_date", "ts_code"], how="left")
j = j.to_pandas()
j["dstr"] = pd.to_datetime(j["trade_date"]).dt.strftime("%Y-%m-%d")
# 按日期 group, 计算有/无涨停的 mean fwd
lu_mean, nolu_mean = [], []
for d, sub in j.groupby("dstr", sort=False):
    fwd = sub["fwd"].to_numpy()
    lu = (sub["lu_yesterday"] | sub["lu_chinext_yesterday"]).to_numpy()
    valid = np.isfinite(fwd)
    if valid.sum() < 30:
        continue
    lu_v = fwd[lu & valid]
    nolu_v = fwd[(~lu) & valid]
    if len(lu_v) > 5 and len(nolu_v) > 5:
        lu_mean.append(lu_v.mean())
        nolu_mean.append(nolu_v.mean())
lu_arr = np.array(lu_mean)
nolu_arr = np.array(nolu_mean)
log(f"\n=== 涨停次日 20d forward return (per date) ===")
log(f"  涨停组 mean {lu_arr.mean()*1e4:+.2f}bps  median {np.median(lu_arr)*1e4:+.2f}  n={len(lu_arr)}")
log(f"  非涨停组 mean {nolu_arr.mean()*1e4:+.2f}bps  median {np.median(nolu_arr)*1e4:+.2f}  n={len(nolu_arr)}")
log(f"  差 (LU - nolu) mean {(lu_arr-nolu_arr).mean()*1e4:+.2f}bps  "
    f"t={((lu_arr-nolu_arr).mean()/((lu_arr-nolu_arr).std()+1e-10)*np.sqrt(len(lu_arr))):.2f}")

# 1-17 vs 18-25 split
# Simpler: compute per-period directly
def stat_per_period(j, sd, ed):
    j2 = j[(j["dstr"] >= sd) & (j["dstr"] <= ed)]
    lu_means, nolu_means = [], []
    for d, sub in j2.groupby("dstr", sort=False):
        fwd = sub["fwd"].to_numpy()
        lu = (sub["lu_yesterday"] | sub["lu_chinext_yesterday"]).to_numpy()
        valid = np.isfinite(fwd)
        if valid.sum() < 30:
            continue
        lu_v = fwd[lu & valid]
        nolu_v = fwd[(~lu) & valid]
        if len(lu_v) > 5 and len(nolu_v) > 5:
            lu_means.append(lu_v.mean())
            nolu_means.append(nolu_v.mean())
    if not lu_means:
        return None
    lu_a = np.array(lu_means)
    nolu_a = np.array(nolu_means)
    diff = lu_a - nolu_a
    return (lu_a.mean()*1e4, nolu_a.mean()*1e4, diff.mean()*1e4, len(lu_a))

for sd, ed in [("2010-01-01", "2017-12-31"), ("2018-01-01", "2025-12-31")]:
    r = stat_per_period(j, sd, ed)
    if r:
        lu_m, nolu_m, diff_m, n = r
        log(f"  {sd[:4]}-{ed[:4]}: LU={lu_m:+.1f}bps  no-LU={nolu_m:+.1f}bps  diff={diff_m:+.1f}bps  n={n}")
log(f"screen {time.time()-t0:.0f}s")

# ---- 4. 应用 filter: 排除"昨日涨停"股票 ----
t0 = time.time()
filtered = {}
for d, stocks in picks.items():
    lu_set = lu_map.get(d, set())
    kept = [s for s in stocks if s not in lu_set]
    filtered[d] = kept
n_kept = sum(len(v) for v in filtered.values())
n_orig = sum(len(v) for v in picks.values())
log(f"\n=== 排除昨日涨停后 ===")
log(f"  picks: {n_orig} -> {n_kept} ({n_kept/n_orig*100:.1f}% kept)")

# 过滤 2018+ 段
filtered_2018 = {k: v for k, v in filtered.items() if k >= "2018-01-01"}
sub_dir = OUT / "H4_no_lu_yesterday"
sub_dir.mkdir(parents=True, exist_ok=True)
(sub_dir / "picks.json").write_text(json.dumps(filtered_2018, ensure_ascii=False, indent=1))
(sub_dir / "picks_meta.json").write_text(
    (AKQ / "evidence/v3_resonance_20261010/picks/V5_P413_RET_RANK_o0/picks_meta.json").read_text())
log(f"saved {sub_dir}")
