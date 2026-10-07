"""干跑: 用 3 个因子验证单因子分析全链路 (IC + 回测 + 组装)"""
import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "data/wavehunter_hs300_v31_refactored_v2_20260925.parquet"
W_START, W_END = date(2022,1,1), date(2024,12,31)
COST = 0.005
K_LIST = (10, 20, 30)
REBAL_STEP = 20

def in_window(col):
    c = col.cast(pl.Date)
    return (c >= W_START) & (c <= W_END)

# net20
base = pl.read_parquet(PANEL, columns=['trade_date','ts_code','open','close']).sort(['ts_code','trade_date'])
base = base.with_columns([
    pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
    pl.col('close').shift(-21).over('ts_code').alias('close_t21'),
]).with_columns(
    (pl.col('close_t21') / pl.col('open_t1') - 1.0 - COST).alias('net20')
).select(['trade_date','ts_code','net20'])

win = base.filter(in_window(pl.col('trade_date')))
all_dates = sorted(win['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
rebal_dates_d = [d.date() for d in rebal_dates]
print(f"win dates: {len(all_dates)}, rebal: {len(rebal_dates)}")

# 3 test factors
FACS = ['alpha_alpha065', 'gtja_gtja_154', 'l_ami']
df = pl.read_parquet(PANEL, columns=['trade_date','ts_code'] + FACS)
df = df.filter(in_window(pl.col('trade_date')))
df = df.join(win, on=['trade_date','ts_code'], how='left')
print(f"df: {df.shape}")

for fac in FACS:
    sub = df.select(['trade_date', fac, 'net20']).drop_nulls()
    # IC
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
    ic_df2 = ic_df.with_columns(pl.col('trade_date').dt.year().alias('yr'))
    ic_y = {int(r['yr']): r['ic'] for r in ic_df2.group_by('yr').agg(pl.col('ic').mean()).to_dicts()}

    # backtest
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

    best_dir = 'hi' if ret['hi_20'] >= ret['lo_20'] else 'lo'
    s_best = pr_pd[f'{best_dir}_20'].dropna()
    win_rate = float((s_best > 0).mean())
    nav = np.concatenate(([1.0], (1 + s_best).cumprod().values))
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    yr_ret = {}
    idx = pd.to_datetime(s_best.index)
    for y in (2022, 2023, 2024):
        sv = s_best[idx.year == y]
        yr_ret[y] = float((1+sv).prod() - 1) if len(sv) else np.nan

    print(f"\n{fac}:")
    print(f"  IC mean={ic_mean:+.4f} icir={icir:+.3f} n_dates={len(ic_df)}")
    print(f"  IC by year: {ic_y}")
    print(f"  hi_20={ret['hi_20']*100:+.2f}% lo_20={ret['lo_20']*100:+.2f}% best={best_dir} win={win_rate*100:.0f}% mdd={mdd*100:.2f}%")
    print(f"  yearly: {yr_ret}")
    # period records 测试
    recs = []
    for d in ('hi','lo'):
        for dt, v in pr_pd[f'{d}_20'].dropna().items():
            recs.append({'factor': fac, 'dir': d, 'trade_date': dt, 'ret': float(v)})
    print(f"  period_records: {len(recs)} rows")

print("\n✅ 干跑完成 — 全链路 OK")
