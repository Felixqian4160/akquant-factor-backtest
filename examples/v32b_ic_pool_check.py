"""V32b: 复核 + IC 排名选池变体
1. 精确数字核对 (STATIC vs talib_DX H2 巧合?)
2. IC 排名选池变体: H1 用 IC 排名选 top10 因子 (方向由 IC 符号定), H2 验证
3. 静态池组成打印
"""
import polars as pl
import pandas as pd
import numpy as np
from collections import Counter
import time

PANEL = "data/wavehunter_hs300_v31_refactored_v2_20260925.parquet"
NRUN = "evidence/v31_bear_factor_analysis_20260925/net20_cache.parquet"
OUTDIR = "evidence/v32_ic_voting_20260925"
W_S, W_E = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
REBAL_STEP = 20
K_NOM = 10

net20 = pl.read_parquet(NRUN)
net20 = net20.filter(pl.col('trade_date').is_between(
    pl.lit(W_S.to_pydatetime()), pl.lit(W_E.to_pydatetime())))
all_dates = sorted(net20['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
P = len(rebal_dates)
rebal_pd = [pd.Timestamp(d) for d in rebal_dates]

net20_at = {}
for p, d in enumerate(rebal_dates):
    sub = net20.filter(pl.col('trade_date') == d)
    net20_at[p] = dict(zip(sub['ts_code'].to_list(), sub['net20'].to_list()))

# load IC matrix from previous run
ic_df_all = pd.read_csv(f"{OUTDIR}/ic_matrix_periods.csv", index_col=0)
print(f"IC matrix: {ic_df_all.shape}")
print(f"date cols: {list(ic_df_all.columns[:3])} ... {list(ic_df_all.columns[-2:])}")

split = pd.read_csv("evidence/v31_bear_factor_analysis_20260925/bear_factor_split_validation.csv")

# --- 1. 精确数字: STATIC H2 vs talib_DX H2 ---
print("\n=== 1. 精确数字核对 ===")
merged_idx = ic_df_all.index.tolist()
# STATIC pool = top10 by ret_h1
sv = split.dropna(subset=['ret_h1','ret_h2']).sort_values('ret_h1', ascending=False).head(10)
print("STATIC pool (top10 by H1):")
for _, r in sv.iterrows():
    print(f"  {r['factor']:16s} dir={r['dir_h1']} ret_h1={r['ret_h1']*100:+.2f}% ret_h2={r['ret_h2']*100:+.2f}%")
print(f"talib_DX: ret_h1={split[split['factor']=='talib_DX']['ret_h1'].iloc[0]*100:+.4f}%, ret_h2={split[split['factor']=='talib_DX']['ret_h2'].iloc[0]*100:+.4f}%")

# --- 2. IC 排名选池变体 (H1 IC 排名) ---
print("\n=== 2. IC 排名选池变体 ===")
# H1 periods = 0..17
h1_cols = ic_df_all.columns[:18]
h1_ic = ic_df_all[h1_cols].mean(axis=1)  # factor -> H1 mean IC
h1_ic_valid = h1_ic.dropna()
print(f"valid factors with H1 IC: {len(h1_ic_valid)}")
top_ic = h1_ic_valid.abs().sort_values(ascending=False).head(12)
print("H1 |IC| top 12:")
for f, v in top_ic.items():
    print(f"  {f:20s} H1 IC={h1_ic_valid[f]:+.4f} (|IC|={v:.4f})")

# need nominations — rebuild minimal for pool factors only
POOL_IC = top_ic.index.tolist()[:10]
print(f"\nIC 池 (top10 by |H1 IC|): {POOL_IC}")
# directions: sign of H1 IC
dir_ic = {f: ('hi' if h1_ic_valid[f] >= 0 else 'lo') for f in POOL_IC}
print("directions:", dir_ic)

# build nominations for these factors
def build_nom(fac_list):
    nom = {}
    df = pl.read_parquet(PANEL, columns=['trade_date','ts_code'] + fac_list)
    df = df.filter(pl.col('trade_date').is_between(
        pl.lit(W_S.to_pydatetime()), pl.lit(W_E.to_pydatetime())))
    df = df.join(net20, on=['trade_date','ts_code'], how='left')
    for fac in fac_list:
        sub = df.select(['trade_date','ts_code',fac,'net20']).drop_nulls()
        ranked = sub.with_columns([
            pl.col(fac).rank(descending=True).over('trade_date').alias('rk_hi'),
            pl.col(fac).rank().over('trade_date').alias('rk_lo'),
        ])
        hi = ranked.filter(pl.col('rk_hi') <= K_NOM).group_by('trade_date').agg(pl.col('ts_code').alias('codes'))
        lo = ranked.filter(pl.col('rk_lo') <= K_NOM).group_by('trade_date').agg(pl.col('ts_code').alias('codes'))
        hi_map = dict(zip(hi['trade_date'].to_list(), [set(c) for c in hi['codes'].to_list()]))
        lo_map = dict(zip(lo['trade_date'].to_list(), [set(c) for c in lo['codes'].to_list()]))
        nom[fac] = [(hi_map.get(d, set()), lo_map.get(d, set())) for d in rebal_dates]
    return nom

def run_vote(pool, fixed_dir, M=3, max_stocks=20, min_stocks=5):
    rets = []
    for p in range(P):
        votes = Counter()
        for f in pool:
            hi_set, lo_set = nom_all[f][p]
            s = hi_set if fixed_dir[f] == 'hi' else lo_set
            for c in s:
                votes[c] += 1
        if not votes:
            rets.append(0.0); continue
        sel = [c for c, v in votes.items() if v >= M]
        if len(sel) < min_stocks:
            sel = [c for c, _ in votes.most_common(min_stocks)]
        if len(sel) > max_stocks:
            sel = [c for c, _ in votes.most_common(max_stocks)]
        rv = [net20_at[p][c] for c in sel if c in net20_at[p] and np.isfinite(net20_at[p][c])]
        rets.append(float(np.mean(rv)) if rv else 0.0)
    return np.array(rets)

# build nominations for all factors needed (static pool + IC pool)
static_pool = sv['factor'].tolist()
static_dir = dict(zip(sv['factor'], sv['dir_h1']))
all_needed = list(set(static_pool + POOL_IC + ['talib_DX']))
nom_all = build_nom(all_needed)
print(f"nominations built for {len(all_needed)} factors")

# STATIC exact
r_static = run_vote(static_pool, static_dir)
h2_static = np.prod([1+r for r in r_static[18:]]) - 1
total_static = np.prod([1+r for r in r_static]) - 1
print(f"\nSTATIC exact: total={total_static*100:+.4f}% H2={h2_static*100:+.4f}%")

# talib_DX solo exact
r_dx = run_vote(['talib_DX'], {'talib_DX': 'hi'})
h2_dx = np.prod([1+r for r in r_dx[18:]]) - 1
print(f"talib_DX solo exact: total={np.prod([1+r for r in r_dx])*100-100:+.4f}% H2={h2_dx*100:+.4f}%")

# IC 池
r_ic = run_vote(POOL_IC, dir_ic)
h2_ic = np.prod([1+r for r in r_ic[18:]]) - 1
total_ic = np.prod([1+r for r in r_ic]) - 1
yr_ic = {}
for y in (2022, 2023, 2024):
    idx = [i for i in range(P) if rebal_pd[i].year == y]
    yr_ic[y] = (np.prod([1+r_ic[i] for i in idx]) - 1)
print(f"\nIC 池 (H1 IC top10 + sign 方向) exact: total={total_ic*100:+.4f}% H2={h2_ic*100:+.4f}%")
print(f"  yearly: {({k: f'{v*100:+.1f}%' for k, v in yr_ic.items()})}")

# 保存 IC 池结果
ic_res = pd.DataFrame({'period': range(P), 'date': [d.date() for d in rebal_pd], 'ret': r_ic})
ic_res.to_csv(f"{OUTDIR}/STATIC_IC_pool_periods.csv", index=False)
print(f"\nSaved: {OUTDIR}/STATIC_IC_pool_periods.csv")

# 对照汇总
print("\n=== 汇总 ===")
print(f"benchmark:               -13.42%")
print(f"talib_DX solo (H2):      {h2_dx*100:+.2f}%")
print(f"STATIC_H1top10 (H2):     {h2_static*100:+.2f}%")
print(f"STATIC_ICtop10 (H2):     {h2_ic*100:+.2f}%")
