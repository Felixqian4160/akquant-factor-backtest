"""V31 fix: 修复 v1 两个 bug, 输出 v2 panel
Bug 1: OHLCV (open/high/low/close/vol/amount) 未包含在输出面板
Bug 2: max_horizontal(null, x) 把 null 静默填成 x (l_size NaN→0, v_ep NaN→1)
       修复: 用 when/then/otherwise(None) 做 null-safe 下限保护

输出: data/wavehunter_hs300_v31_refactored_v2_20260925.parquet
  = 390 原始 stockpick + 22 新学术因子 + OHLCV + keys (420 列)
"""

import polars as pl
import pandas as pd
import numpy as np
import time
import os

PANEL_IN = "data/wavehunter_hs300_with_talib_20260924.parquet"
PANEL_OUT = "data/wavehunter_hs300_v31_refactored_v2_20260925.parquet"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 80)
log("V31 fix v2: null-safe 修复 + 加入 OHLCV")
log("=" * 80)

t0 = time.time()

# === Column sets ===
all_cols = pl.scan_parquet(PANEL_IN).collect_schema().names()
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]
OHLCV = ['open', 'high', 'low', 'close', 'vol', 'amount']

# === Load minimal cols for computing new factors ===
log("Loading minimal cols...")
BASIC = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
         'cap', 'circ_cap', 'pe', 'pb', 'bps']
df = pl.read_parquet(PANEL_IN, columns=BASIC)
df = df.sort(['ts_code', 'trade_date'])
log(f"  Loaded: {df.shape}")

# daily_ret + market_ret
df = df.with_columns(pl.col('close').pct_change().over('ts_code').alias('daily_ret'))
mkt = (
    df.group_by('trade_date')
    .agg(pl.col('close').mean().alias('mkt_close'))
    .sort('trade_date')
    .with_columns(pl.col('mkt_close').pct_change().alias('mkt_ret'))
)
df = df.join(mkt.select('trade_date', 'mkt_ret'), on='trade_date', how='left')

# === Liquidity (null-safe) ===
log("Liquidity factors (null-safe)...")
df = df.with_columns([
    pl.when(pl.col('circ_cap') > 0).then(pl.col('circ_cap').log()).otherwise(None).alias('l_size'),
    pl.when(pl.col('circ_cap') > 0).then((pl.col('amount') / pl.col('circ_cap'))).otherwise(None).alias('daily_turnover'),
])
df = df.with_columns([
    (pl.col('l_size') ** 3).alias('l_size3'),
    pl.col('daily_turnover').rolling_mean(21).over('ts_code').log1p().alias('l_turnm'),
    pl.col('daily_turnover').rolling_mean(252).over('ts_code').log1p().alias('l_turna'),
])
df = df.with_columns([
    pl.when(pl.col('amount') > 0).then(pl.col('daily_ret').abs() / pl.col('amount')).otherwise(None).alias('abs_ret_amt'),
])
df = df.with_columns([
    pl.col('abs_ret_amt').rolling_mean(21).over('ts_code').alias('l_ami'),
])
df = df.drop('abs_ret_amt')
df = df.with_columns([
    pl.col('amount').rolling_mean(21).over('ts_code').log1p().alias('l_dtvm'),
    pl.col('amount').rolling_mean(252).over('ts_code').log1p().alias('l_dtva'),
    pl.col('amount').rolling_std(120).over('ts_code').log1p().alias('l_vdtv'),
])

# === Risk ===
log("Risk factors...")
df = df.with_columns(pl.col('daily_ret').rolling_std(60).over('ts_code').alias('r_tv'))

log("  r_beta (pandas groupby rolling cov/var)...")
_beta_df = df.select(['ts_code', 'trade_date', 'daily_ret', 'mkt_ret']).to_pandas()
_beta_df = _beta_df.sort_values(['ts_code', 'trade_date'])
_beta_df['_beta'] = _beta_df.groupby('ts_code').apply(
    lambda g: g['daily_ret'].rolling(60).cov(g['mkt_ret']) / g['mkt_ret'].rolling(60).var().replace(0, np.nan),
    include_groups=False
).reset_index(level=0, drop=True).values
df = df.join(
    pl.from_pandas(_beta_df[['ts_code', 'trade_date', '_beta']].rename(columns={'_beta': 'r_beta'})),
    on=['ts_code', 'trade_date'], how='left',
)

# === Past Returns ===
log("Past Returns factors...")
df = df.with_columns([
    pl.col('close').shift(21).over('ts_code').pct_change(21).alias('p_m1'),
    pl.col('close').shift(21).over('ts_code').pct_change(63).alias('p_m3'),
    pl.col('close').shift(21).over('ts_code').pct_change(126).alias('p_m6'),
    pl.col('close').shift(21).over('ts_code').pct_change(231).alias('p_m11'),
    pl.col('close').shift(21).over('ts_code').pct_change(504).alias('p_m24'),
])
df = df.with_columns([
    (pl.col('close').pct_change(126).shift(21).over('ts_code')
     - pl.col('close').pct_change(252).shift(21).over('ts_code')).alias('p_mchg'),
    (pl.col('close') / pl.col('close').rolling_max(252).over('ts_code')).alias('p_52w'),
    pl.col('daily_ret').rolling_max(21).over('ts_code').alias('p_mdr'),
    pl.when(pl.col('close') > 0).then(pl.col('close').log()).otherwise(None).alias('p_pr'),
    (pl.col('trade_date').dt.month() == 12).cast(pl.Int8).alias('p_season'),
])

# === Value (null-safe) ===
log("Value factors (null-safe)...")
df = df.with_columns([
    pl.when(pl.col('close') > 0).then(pl.col('bps') / pl.col('close')).otherwise(None).alias('v_bm'),
    pl.when(pl.col('pe') > 0).then(1.0 / pl.col('pe')).otherwise(None).alias('v_ep'),
])

NEW_FACTORS = ['l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami',
                'l_dtvm', 'l_dtva', 'l_vdtv',
                'r_tv', 'r_beta',
                'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
                'p_52w', 'p_mdr', 'p_pr', 'p_season',
                'v_bm', 'v_ep']
log(f"New factors computed: {len(NEW_FACTORS)}")

# === Load original cols + OHLCV, join ===
log("Loading original stockpick + OHLCV...")
df_orig = pl.read_parquet(PANEL_IN, columns=['trade_date', 'ts_code'] + OHLCV + STOCKPICK_COLS)
log(f"  Loaded: {df_orig.shape}")

df_final = df_orig.join(
    df.select(['trade_date', 'ts_code'] + NEW_FACTORS),
    on=['trade_date', 'ts_code'], how='left',
)
log(f"  Joined: {df_final.shape}")

# === Save ===
log(f"Saving: {PANEL_OUT}")
df_final.write_parquet(PANEL_OUT)
log(f"  Size: {os.path.getsize(PANEL_OUT)/1e9:.2f} GB")
log(f"  Expect cols: {2 + len(OHLCV) + len(STOCKPICK_COLS) + len(NEW_FACTORS)} = {len(df_final.columns)} actual")

# === Quick verify ===
log("Verify...")
dv = pl.read_parquet(PANEL_OUT)
log(f"  Shape: {dv.shape}")
for c in OHLCV:
    assert c in dv.columns, f"OHLCV missing: {c}"
log("  OHLCV: ✅ all present")
log(f"  l_size nulls: {dv['l_size'].is_null().sum()} (expect >0 to match circ_cap nulls)")
log(f"  v_ep nulls:   {dv['v_ep'].is_null().sum()}")
log(f"  l_turnm nulls:{dv['l_turnm'].is_null().sum()}")
log("=" * 80)
log("✅ v2 done")
log("=" * 80)