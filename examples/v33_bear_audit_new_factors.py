"""V33 熊市 lift audit — 17 个新因子 (2022-2024)

完全复用 v31_bear_factor_analysis.py 的口径:
  - net20 = close[t+21]/open[t+1] - 1 - 0.005
  - 37 期 (每 20 交易日调仓), k=10/20/30, hi/lo 双方向
  - IC = 日度 rank corr (Spearman) vs net20
  - 复利净收益 + 胜率 + MDD + 分年

输入: V33 panel (458 cols)
输出: evidence/v33_bear_audit_20261003/
  - new_factor_ranking.csv (17 因子完整指标)
  - combined_vs_v31.csv (与 v31 412 因子合并排名对比)
  - period_returns_k20.parquet
"""
from __future__ import annotations

import time
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUT = Path("evidence/v33_bear_audit_20261003")
V31_RANKING = Path("evidence/v31_bear_factor_analysis_20260925/bear_factor_ranking_2022_2024.csv")
W_START, W_END = _date(2022, 1, 1), _date(2024, 12, 31)
COST = 0.005
K_LIST = (10, 20, 30)
REBAL_STEP = 20

NEW17 = ['winner_ratio', 'efficiency_ratio', 'fractal_dimension',
         'alpha191_040', 'alpha191_095', 'mom12m_jt', 'maxret_bcw',
         'accruals_sloan', 'idiovola_clmx', 'gp_novymarx',
         'overnight_intraday_spread', 'skew21_lottery', 'pvcorr_21',
         'kurt21_returns', 'coskew60', 'hl_52w_disposition', 'resmom_6m']


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def in_window(col):
    c = col.cast(pl.Date)
    return (c >= W_START) & (c <= W_END)


OUT.mkdir(parents=True, exist_ok=True)
log("=" * 80)
log("V33 熊市 lift audit — 17 个新因子 (2022-2024)")
log("=" * 80)

# --- 1. net20 ---
log("Computing net20...")
base = (pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'close'])
        .sort(['ts_code', 'trade_date'])
        .with_columns([
            pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
            pl.col('close').shift(-21).over('ts_code').alias('close_t21'),
        ])
        .with_columns((pl.col('close_t21') / pl.col('open_t1') - 1.0 - COST).alias('net20'))
        .select(['trade_date', 'ts_code', 'net20']))
win = base.filter(in_window(pl.col('trade_date')))
all_dates = sorted(win['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
rebal_dates_d = [d.date() for d in rebal_dates]
log(f"  dates: {len(all_dates)}, rebalances: {len(rebal_dates)} ({rebal_dates[0]} → {rebal_dates[-1]})")

# benchmark
bm = win.filter(pl.col('trade_date').cast(pl.Date).is_in(rebal_dates_d)).group_by('trade_date').agg(
    pl.col('net20').mean().alias('ret')).sort('trade_date')
bm_total = float((1 + bm['ret'].fill_null(0)).cum_prod()[-1] - 1)
log(f"  benchmark (equal-weight) 3y: {bm_total*100:+.2f}%")

# --- 2. load 17 factors + net20 ---
log("Loading 17 factors...")
df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code'] + NEW17)
df = df.filter(in_window(pl.col('trade_date')))
df = df.join(win, on=['trade_date', 'ts_code'], how='left')

results = []
period_records = []
t0 = time.time()

for fac in NEW17:
    sub = df.select(['trade_date', fac, 'net20']).drop_nulls()

    # IC
    subr = sub.with_columns([
        pl.col(fac).rank().over('trade_date').alias('rf'),
        pl.col('net20').rank().over('trade_date').alias('rn'),
    ])
    ic_df = subr.group_by('trade_date').agg([pl.len().alias('n'), pl.corr('rf', 'rn').alias('ic')]).filter(pl.col('n') >= 20)
    ic_mean = ic_df['ic'].mean()
    ic_std = ic_df['ic'].std()
    icir = (ic_mean / ic_std) if (ic_std and ic_std > 0) else np.nan
    ic_df = ic_df.with_columns(pl.col('trade_date').dt.year().alias('yr'))
    ic_y = {int(r['yr']): r['ic'] for r in ic_df.group_by('yr').agg(pl.col('ic').mean()).to_dicts()}

    # backtest at rebal dates
    subw = sub.filter(pl.col('trade_date').cast(pl.Date).is_in(rebal_dates_d))
    agg_exprs = []
    for k in K_LIST:
        agg_exprs += [
            pl.col('net20').sort_by(pl.col(fac)).tail(k).mean().alias(f'hi_{k}'),
            pl.col('net20').sort_by(pl.col(fac)).head(k).mean().alias(f'lo_{k}'),
        ]
    pr = subw.group_by('trade_date').agg(agg_exprs).sort('trade_date')
    pr_pd = pr.to_pandas().set_index('trade_date')

    ret = {}
    for k in K_LIST:
        for d in ('hi', 'lo'):
            s = pr_pd[f'{d}_{k}'].dropna()
            ret[f'{d}_{k}'] = float((1 + s).prod() - 1) if len(s) else np.nan

    best_dir = 'hi' if (ret['hi_20'] >= ret['lo_20']) else 'lo'
    s_best = pr_pd[f'{best_dir}_20'].dropna()
    win_rate = float((s_best > 0).mean()) if len(s_best) else np.nan
    nav = np.concatenate(([1.0], (1 + s_best).cumprod().values)) if len(s_best) else np.array([1.0, 1.0])
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    yr_ret = {}
    if len(s_best):
        idx = pd.to_datetime(s_best.index)
        for y in (2022, 2023, 2024):
            sv = s_best[idx.year == y]
            yr_ret[y] = float((1 + sv).prod() - 1) if len(sv) else np.nan

    for d in ('hi', 'lo'):
        for dt, v in pr_pd[f'{d}_20'].dropna().items():
            period_records.append({'factor': fac, 'dir': d, 'trade_date': dt, 'ret': float(v)})

    best_ret20 = max(ret['hi_20'], ret['lo_20']) if not (np.isnan(ret['hi_20']) or np.isnan(ret['lo_20'])) else ret['hi_20']
    results.append({
        'factor': fac, 'n_ic_dates': len(ic_df),
        'ic_mean': float(ic_mean) if not np.isnan(ic_mean) else np.nan,
        'icir': float(icir) if not np.isnan(icir) else np.nan,
        'ic_2022': ic_y.get(2022, np.nan), 'ic_2023': ic_y.get(2023, np.nan), 'ic_2024': ic_y.get(2024, np.nan),
        'ret_hi_10': ret['hi_10'], 'ret_hi_20': ret['hi_20'], 'ret_hi_30': ret['hi_30'],
        'ret_lo_10': ret['lo_10'], 'ret_lo_20': ret['lo_20'], 'ret_lo_30': ret['lo_30'],
        'best_dir': best_dir, 'best_ret20': best_ret20,
        'win20': win_rate, 'mdd20': mdd, 'n_periods': len(s_best),
        'y2022': yr_ret.get(2022, np.nan), 'y2023': yr_ret.get(2023, np.nan), 'y2024': yr_ret.get(2024, np.nan),
    })
    log(f"  {fac:28s} ret20={best_ret20*100:+7.2f}% ({best_dir}) ic={ic_mean:+.4f} win={win_rate*100:.0f}% mdd={mdd*100:6.1f}%")

log(f"\nLoop done in {time.time()-t0:.0f}s")

res = pd.DataFrame(results).sort_values('best_ret20', ascending=False).reset_index(drop=True)
res.insert(0, 'rank', range(1, len(res) + 1))
res.to_csv(OUT / "new_factor_ranking.csv", index=False)
pd.DataFrame(period_records).to_parquet(OUT / "period_returns_k20.parquet")

# --- 3. 对比 v31 全库 ---
log("\n=== 对比 v31 412 因子 ===")
v31 = pd.read_csv(V31_RANKING)
merged = pd.concat([
    v31[['factor', 'family', 'best_dir', 'best_ret20', 'win20', 'mdd20', 'ic_mean']].assign(source='v31_412', v31_rank=v31['rank']),
    res[['factor', 'best_dir', 'best_ret20', 'win20', 'mdd20', 'ic_mean']].assign(source='v33_new17', family='v33_new', v31_rank=np.nan),
], ignore_index=True)
merged = merged.sort_values('best_ret20', ascending=False).reset_index(drop=True)
merged.insert(0, 'combined_rank', range(1, len(merged) + 1))
merged.to_csv(OUT / "combined_vs_v31.csv", index=False)

# 新因子在合并榜的位置
new_in_combined = merged[merged['source'] == 'v33_new17'][['combined_rank', 'factor', 'best_dir', 'best_ret20', 'ic_mean']]
log("\n17 新因子在合并榜 (430 因子) 的位置:")
for _, r in new_in_combined.iterrows():
    log(f"  #{int(r['combined_rank']):>3} {r['factor']:28s} {r['best_dir']} ret20={r['best_ret20']*100:+7.2f}% ic={r['ic_mean']:+.4f}")

# --- 4. summary ---
valid = res.dropna(subset=['best_ret20'])
log("\n" + "=" * 80)
log("SUMMARY (17 new factors)")
log("=" * 80)
log(f"Mean best_ret20:   {valid['best_ret20'].mean()*100:+.2f}%")
log(f"Median best_ret20: {valid['best_ret20'].median()*100:+.2f}%")
log(f"Positive: {(valid['best_ret20']>0).sum()}/17 ({(valid['best_ret20']>0).mean()*100:.1f}%)")
log(f"vs benchmark {bm_total*100:+.2f}%")
# v31 全库均值对比
log(f"\nv31 全库 412 因子 mean: {v31['best_ret20'].mean()*100:+.2f}%")
# 与 v31 Top10 比较
v31_top10 = v31.head(10)
log(f"v31 Top10 mean: {v31_top10['best_ret20'].mean()*100:+.2f}%")
log(f"新因子前 10 mean: {valid.head(10)['best_ret20'].mean()*100:+.2f}%")

log(f"\nSaved: {OUT}/new_factor_ranking.csv, combined_vs_v31.csv, period_returns_k20.parquet")
log("DONE")
