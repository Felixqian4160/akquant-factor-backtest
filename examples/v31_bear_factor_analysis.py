"""V31 熊市因子分析 (2022-2024): IC 排名 + 组合回测收益

对标牛市阶段的因子分析方法:
  1. IC 排名: 每个因子的日度 rank IC (Spearman) → mean / ICIR / 分年
  2. 组合回测: 每个因子按高/低方向 top-k 组合 (k=10/20/30) 复利净收益
     - T+1 raw open 买入 / T+21 raw close 卖出 (net20, 已含 0.5% 往返成本)
     - 每 20 个交易日调仓, 保留表现更好的方向
  3. 敏感度: k=10/20/30 三档 + high/low 双方向 = 组合规模/方向敏感度

窗口: 2022-01-01 ~ 2024-12-31 (用户铁律: 熊市只看这 3 年)
因子: v2 面板 412 个 (390 原始 + 22 新学术)
"""

import polars as pl
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import date as _date

PANEL = "data/wavehunter_hs300_v31_refactored_v2_20260925.parquet"
OUTDIR = "evidence/v31_bear_factor_analysis_20260925"
W_START, W_END = "2022-01-01", "2024-12-31"
_W_D1, _W_D2 = _date(2022, 1, 1), _date(2024, 12, 31)
COST = 0.005
K_LIST = (10, 20, 30)
REBAL_STEP = 20


def in_window(col):
    """trade_date (datetime) 落在 2022-2024 窗口内的 mask"""
    c = col.cast(pl.Date)
    return (c >= _W_D1) & (c <= _W_D2)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

os.makedirs(OUTDIR, exist_ok=True)
log("=" * 80)
log("V31 熊市因子分析: IC 排名 + 组合回测收益 (2022-2024)")
log("=" * 80)

# --- 0. sanity test ---
_t = pl.DataFrame({'d':[1,1,1,2,2,2],'f':[3.,2.,1.,6.,5.,4.],'x':[10.,20.,30.,40.,50.,60.]})
_r = _t.group_by('d').agg([
    pl.col('x').sort_by(pl.col('f')).tail(2).mean().alias('hi2'),
    pl.col('x').sort_by(pl.col('f')).head(2).mean().alias('lo2'),
]).sort('d')
assert _r['hi2'].to_list() == [15.0, 45.0] and _r['lo2'].to_list() == [25.0, 55.0]
log("sanity OK")

# --- 1. net20 (T+1 open -> T+21 close, -0.5% cost) ---
log("Computing net20...")
base = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'close']).sort(['ts_code','trade_date'])
base = base.with_columns([
    pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
    pl.col('close').shift(-21).over('ts_code').alias('close_t21'),
]).with_columns(
    (pl.col('close_t21') / pl.col('open_t1') - 1.0 - COST).alias('net20')
).select(['trade_date','ts_code','net20'])
base.write_parquet(f"{OUTDIR}/net20_cache.parquet")

win = base.filter(in_window(pl.col('trade_date')))
all_dates = sorted(win['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
rebal_dates_d = [d.date() for d in rebal_dates]
log(f"  window dates: {len(all_dates)} ({all_dates[0]} → {all_dates[-1]})")
log(f"  rebalance dates: {len(rebal_dates)}")

# --- 2. factor list ---
cols = pl.scan_parquet(PANEL).collect_schema().names()
SKIP = {'trade_date','ts_code','open','high','low','close','vol','amount'}
FACTORS = [c for c in cols if c not in SKIP]

NEW22 = {'l_size','l_size3','l_turnm','l_turna','l_ami','l_dtvm','l_dtva','l_vdtv',
         'r_tv','r_beta','p_m1','p_m3','p_m6','p_m11','p_m24','p_mchg',
         'p_52w','p_mdr','p_pr','p_season','v_bm','v_ep'}

def family(f):
    if f in NEW22: return 'new_academic'
    if f.startswith('alpha_'): return 'alpha'
    if f.startswith('gtja_'): return 'gtja'
    if f.startswith('talib_'): return 'talib'
    return 'base'

log(f"  factors: {len(FACTORS)} (new: {len(NEW22)})")

# --- 3. benchmark: equal-weight all-stock net20 at rebal dates ---
bm = win.filter(pl.col('trade_date').cast(pl.Date).is_in(rebal_dates_d)).group_by('trade_date').agg(
    pl.col('net20').mean().alias('ret')
).sort('trade_date')
bm_nav = (1 + bm['ret'].fill_null(0)).cum_prod()
bm_total = float(bm_nav[-1] - 1)
log(f"  benchmark (equal-weight) 3y: {bm_total*100:+.2f}%")

# --- 4. main loop ---
results = []
period_records = []  # (factor, direction, date, ret) for k=20
BATCH = 40
t0 = time.time()
n_done = 0

for bstart in range(0, len(FACTORS), BATCH):
    batch = FACTORS[bstart:bstart+BATCH]
    df = pl.read_parquet(PANEL, columns=['trade_date','ts_code'] + batch)
    df = df.filter(in_window(pl.col('trade_date')))
    df = df.join(win, on=['trade_date','ts_code'], how='left')

    for fac in batch:
        sub = df.select(['trade_date', fac, 'net20']).drop_nulls()

        # --- IC (daily rank IC over all window dates) ---
        subr = sub.with_columns([
            pl.col(fac).rank().over('trade_date').alias('rf'),
            pl.col('net20').rank().over('trade_date').alias('rn'),
        ])
        ic_df = subr.group_by('trade_date').agg([
            pl.len().alias('n'),
            pl.corr('rf','rn').alias('ic'),
        ]).filter(pl.col('n') >= 20)
        ic_mean = ic_df['ic'].mean()
        ic_std = ic_df['ic'].std()
        icir = (ic_mean / ic_std) if (ic_std and ic_std > 0) else np.nan
        ic_df = ic_df.with_columns(pl.col('trade_date').dt.year().alias('yr'))
        ic_y = {int(r['yr']): r['ic'] for r in ic_df.group_by('yr').agg(pl.col('ic').mean()).to_dicts()}

        # --- backtest at rebal dates ---
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
            for d in ('hi','lo'):
                s = pr_pd[f'{d}_{k}'].dropna()
                ret[f'{d}_{k}'] = float((1+s).prod() - 1) if len(s) else np.nan

        best_dir = 'hi' if (ret['hi_20'] >= ret['lo_20'] if not (np.isnan(ret['hi_20']) or np.isnan(ret['lo_20'])) else True) else 'lo'
        s_best = pr_pd[f'{best_dir}_20'].dropna()
        win_rate = float((s_best > 0).mean()) if len(s_best) else np.nan
        nav = np.concatenate(([1.0], (1 + s_best).cumprod().values)) if len(s_best) else np.array([1.0, 1.0])
        peak = np.maximum.accumulate(nav)
        mdd = float((nav / peak - 1).min())
        # yearly
        yr_ret = {}
        if len(s_best):
            idx = pd.to_datetime(s_best.index)
            for y in (2022, 2023, 2024):
                sv = s_best[idx.year == y]
                yr_ret[y] = float((1+sv).prod() - 1) if len(sv) else np.nan

        # period returns for k20 both dirs (for chart)
        for d in ('hi','lo'):
            for dt, v in pr_pd[f'{d}_20'].dropna().items():
                period_records.append({'factor': fac, 'dir': d, 'trade_date': dt, 'ret': float(v)})

        results.append({
            'factor': fac, 'family': family(fac),
            'n_ic_dates': len(ic_df),
            'ic_mean': float(ic_mean) if ic_mean is not None and not np.isnan(ic_mean) else np.nan,
            'icir': float(icir) if icir is not None and not np.isnan(icir) else np.nan,
            'ic_2022': ic_y.get(2022, np.nan), 'ic_2023': ic_y.get(2023, np.nan), 'ic_2024': ic_y.get(2024, np.nan),
            'ret_hi_10': ret['hi_10'], 'ret_hi_20': ret['hi_20'], 'ret_hi_30': ret['hi_30'],
            'ret_lo_10': ret['lo_10'], 'ret_lo_20': ret['lo_20'], 'ret_lo_30': ret['lo_30'],
            'best_dir': best_dir, 'best_ret20': max(ret['hi_20'], ret['lo_20']) if not (np.isnan(ret['hi_20']) or np.isnan(ret['lo_20'])) else ret['hi_20'],
            'win20': win_rate, 'mdd20': mdd, 'n_periods': len(s_best),
            'y2022': yr_ret.get(2022, np.nan), 'y2023': yr_ret.get(2023, np.nan), 'y2024': yr_ret.get(2024, np.nan),
        })
        n_done += 1

    # partial save
    pd.DataFrame(results).to_csv(f"{OUTDIR}/_partial_ranking.csv", index=False)
    log(f"  progress: {n_done}/{len(FACTORS)} factors, elapsed {time.time()-t0:.0f}s")

log(f"Loop done: {n_done} factors in {time.time()-t0:.0f}s")

# --- 5. assemble + save ---
res = pd.DataFrame(results).sort_values('best_ret20', ascending=False).reset_index(drop=True)
res.insert(0, 'rank', range(1, len(res)+1))
res.to_csv(f"{OUTDIR}/bear_factor_ranking_2022_2024.csv", index=False)
log(f"Saved: {OUTDIR}/bear_factor_ranking_2022_2024.csv")

pd.DataFrame(period_records).to_parquet(f"{OUTDIR}/period_returns_k20.parquet")
log(f"Saved: {OUTDIR}/period_returns_k20.parquet")

# --- 6. summary stats ---
valid = res.dropna(subset=['best_ret20'])
log("=" * 80)
log("SUMMARY")
log("=" * 80)
log(f"Total factors: {len(res)}, valid: {len(valid)}")
log(f"Mean best_ret20:   {valid['best_ret20'].mean()*100:+.2f}%")
log(f"Median best_ret20: {valid['best_ret20'].median()*100:+.2f}%")
log(f"Positive: {(valid['best_ret20']>0).sum()}/{len(valid)} ({(valid['best_ret20']>0).mean()*100:.1f}%)")
log(f"> +10%: {(valid['best_ret20']>0.10).sum()}, > +20%: {(valid['best_ret20']>0.20).sum()}, > +50%: {(valid['best_ret20']>0.50).sum()}")
log(f"Benchmark (equal-weight 3y): {bm_total*100:+.2f}%")
log("")
log("Top 20 by best_ret20:")
top20 = valid.head(20)[['rank','factor','family','best_dir','best_ret20','win20','mdd20','ic_mean','icir','y2022','y2023','y2024']]
for _, r in top20.iterrows():
    log(f"  #{r['rank']:>3} {r['factor']:30s} {r['family']:12s} {r['best_dir']} ret20={r['best_ret20']*100:+7.2f}% win={r['win20']*100:.0f}% mdd={r['mdd20']*100:6.2f}% ic={r['ic_mean']:+.4f}")

log("")
log("Top 15 by |IC|:")
res['abs_ic'] = res['ic_mean'].abs()
top_ic = res.dropna(subset=['abs_ic']).sort_values('abs_ic', ascending=False).head(15)
for _, r in top_ic.iterrows():
    log(f"  {r['factor']:30s} ic={r['ic_mean']:+.4f} icir={r['icir']:+.3f} ret20={r['best_ret20']*100:+7.2f}%")

log("")
log("New 22 academic factors rank:")
for _, r in valid[valid['family']=='new_academic'].iterrows():
    log(f"  #{r['rank']:>3} {r['factor']:14s} {r['best_dir']} ret20={r['best_ret20']*100:+7.2f}% ic={r['ic_mean']:+.4f}")

log("")
log(f"IC vs backtest rank corr (spearman): {res[['ic_mean','best_ret20']].dropna().corr(method='spearman').iloc[0,1]:.3f}")
log("DONE")
