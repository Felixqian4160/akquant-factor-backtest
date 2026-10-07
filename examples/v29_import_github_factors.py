"""V29: 移植 yingwang/trade + FinRL-Adaptive 因子公式到 HS300 panel
- 测试每个公式在我们的面板上能否构造
- 检查哪些数据缺失
- 在 2022-2024 上跑 lift audit

数据源: data/wavehunter_hs300_with_talib_20260924.parquet (419 cols)
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

# Define formulas to import
FORMULAS = {
    # === From yingwang/trade (quant/signals/factors.py) ===
    "ying_momentum_skip21_63": "skip 21d, pct_change(63), cs-zscore",
    "ying_momentum_skip21_126": "skip 21d, pct_change(126), cs-zscore",
    "ying_momentum_skip21_252": "skip 21d, pct_change(252), cs-zscore",
    "ying_mean_reversion_20": "-Bollinger z-score 20d",
    "ying_trend_50_200": "SMA(50)/SMA(200), cs-zscore",
    "ying_short_term_reversal_5": "-rolling_sum(ret, 5), cs-zscore",
    "ying_high_proximity_252": "price / rolling_max(252), cs-zscore",
    "ying_volatility_contraction": "-rolling_std(10)/rolling_std(63)",
    "ying_volatility_63": "-rolling_std(63)*sqrt(252)",
    "ying_trend_persistence_21": "rolling autocorr of returns(21)",
    # === From FinRL-Adaptive (strategies/) ===
    "finrl_bollinger_upper_20": "BB upper 20d, 2std",
    "finrl_bollinger_lower_20": "BB lower 20d, 2std",
    "finrl_zscore_20": "price - MA(20) / STD(20)",
    "finrl_atr_14": "TA-Lib ATR(14)",
    "finrl_rsi_14": "TA-Lib RSI(14)",
    "finrl_macd_hist": "TA-Lib MACD histogram",
    "finrl_adx_14": "TA-Lib ADX(14)",
    "finrl_sma_50_200_ratio": "SMA(50)/SMA(200)",
}

# Data availability check
all_cols = pl.scan_parquet(PANEL).collect_schema().names()
has_close = 'close' in all_cols
has_open = 'open' in all_cols
has_high = 'high' in all_cols
has_low = 'low' in all_cols
has_volume = 'vol' in all_cols
has_amount = 'amount' in all_cols

print(f"Data availability:")
print(f"  close: {has_close}")
print(f"  open: {has_open}")
print(f"  high: {has_high}")
print(f"  low: {has_low}")
print(f"  volume: {has_volume}")
print(f"  amount: {has_amount}")

# Compute formulas
print(f"\n=== Computing imported formulas ===")
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'vol', 'amount'] + [c for c in all_cols if c.startswith('talib_')])
df = df.sort(['ts_code', 'trade_date'])
df_pd = df.to_pandas()
print(f"  Loaded: {df_pd.shape}")

# Per-stock formulas (need shift to avoid lookahead)
import warnings
warnings.filterwarnings('ignore')

results = []
t0 = time.time()

def per_stock_transform(df_pd, fn, fac_name):
    """Apply per-stock transformation, output cross-sectional z-score per date."""
    out = df_pd[['ts_code', 'trade_date']].copy()
    fn(df_pd, out)
    # cross-sectional z-score per date
    out[fac_name + '_z'] = out.groupby('trade_date')[fac_name].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return out[[fac_name, fac_name + '_z']]

# === yingwang formulas ===
print("\n--- yingwang/trade ---")

# 1. momentum skip21 63d
def ying_mom_63(df_in, out):
    out['ying_mom_skip21_63'] = df_in.groupby('ts_code')['close'].transform(
        lambda x: x.shift(21).pct_change(63)
    )
print("ying_mom_skip21_63 ...")
t = time.time()
out1 = per_stock_transform(df_pd, ying_mom_63, 'ying_mom_skip21_63')
print(f"  ok: {out1['ying_mom_skip21_63'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 2. mean reversion (Bollinger z-score)
def ying_mr_20(df_in, out):
    def _mr(g):
        m = g.rolling(20).mean()
        s = g.rolling(20).std().replace(0, 1)
        return -(g - m) / s
    out['ying_mr_20'] = df_in.groupby('ts_code')['close'].transform(_mr)
print("ying_mr_20 ...")
t = time.time()
out2 = per_stock_transform(df_pd, ying_mr_20, 'ying_mr_20')
print(f"  ok: {out2['ying_mr_20'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 3. trend SMA(50)/SMA(200)
def ying_trend_50_200(df_in, out):
    def _tr(g):
        s = g.rolling(50).mean()
        l = g.rolling(200).mean().replace(0, np.nan)
        return s / l
    out['ying_trend_50_200'] = df_in.groupby('ts_code')['close'].transform(_tr)
print("ying_trend_50_200 ...")
t = time.time()
out3 = per_stock_transform(df_pd, ying_trend_50_200, 'ying_trend_50_200')
print(f"  ok: {out3['ying_trend_50_200'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 4. short_term_reversal_5
def ying_str_5(df_in, out):
    def _str(g):
        r = g.pct_change()
        return -r.rolling(5).sum()
    out['ying_str_5'] = df_in.groupby('ts_code')['close'].transform(_str)
print("ying_str_5 ...")
t = time.time()
out4 = per_stock_transform(df_pd, ying_str_5, 'ying_str_5')
print(f"  ok: {out4['ying_str_5'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 5. high_proximity_252
def ying_hp_252(df_in, out):
    def _hp(g):
        h = g.rolling(252).max()
        return g / h.replace(0, np.nan)
    out['ying_hp_252'] = df_in.groupby('ts_code')['close'].transform(_hp)
print("ying_hp_252 ...")
t = time.time()
out5 = per_stock_transform(df_pd, ying_hp_252, 'ying_hp_252')
print(f"  ok: {out5['ying_hp_252'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 6. volatility_contraction
def ying_vc(df_in, out):
    def _vc(g):
        r = g.pct_change()
        sv = r.rolling(10).std()
        lv = r.rolling(63).std().replace(0, np.nan)
        return -(sv / lv)
    out['ying_vc'] = df_in.groupby('ts_code')['close'].transform(_vc)
print("ying_vc ...")
t = time.time()
out6 = per_stock_transform(df_pd, ying_vc, 'ying_vc')
print(f"  ok: {out6['ying_vc'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# 7. volatility 63d (low-vol anomaly)
def ying_vol_63(df_in, out):
    def _v(g):
        r = g.pct_change()
        return -(r.rolling(63).std() * np.sqrt(252))
    out['ying_vol_63'] = df_in.groupby('ts_code')['close'].transform(_v)
print("ying_vol_63 ...")
t = time.time()
out7 = per_stock_transform(df_pd, ying_vol_63, 'ying_vol_63')
print(f"  ok: {out7['ying_vol_63'].notna().mean()*100:.1f}% non-null, {time.time()-t:.1f}s")

# === FinRL-Adaptive formulas (already in talib) ===
print("\n--- FinRL-Adaptive (talib native) ---")
talib_map = {
    "finrl_atr_14": "talib_ATR",
    "finrl_rsi_14": "talib_RSI",
    "finrl_adx_14": "talib_ADX",
    "finrl_boll_upper_20": "talib_BBANDS_0",
    "finrl_boll_lower_20": "talib_BBANDS_2",
}
for fac_name, talib_col in talib_map.items():
    if talib_col in all_cols:
        print(f"  {fac_name} = {talib_col} ✅")
    else:
        print(f"  {fac_name} = {talib_col} ❌ MISSING")

# === Compute net20 ===
print(f"\n=== Computing net20 for bear lift ===")
df_pd['open_t1'] = df_pd.groupby('ts_code')['open'].shift(-1)
df_pd['close_t21'] = df_pd.groupby('ts_code')['close'].shift(-21)
df_pd['net20'] = (df_pd['close_t21'] / df_pd['open_t1'] - 1) - 0.005
df_pd['date'] = pd.to_datetime(df_pd['trade_date']).dt.date

mask = (df_pd['date'] >= date(2022, 1, 1)) & (df_pd['date'] <= date(2024, 12, 31))
df_22y = df_pd[mask].copy().dropna(subset=['net20'])

# Merge all factors
factors_df = pd.concat([
    out1[['ying_mom_skip21_63', 'ying_mom_skip21_63_z']],
    out2[['ying_mr_20', 'ying_mr_20_z']],
    out3[['ying_trend_50_200', 'ying_trend_50_200_z']],
    out4[['ying_str_5', 'ying_str_5_z']],
    out5[['ying_hp_252', 'ying_hp_252_z']],
    out6[['ying_vc', 'ying_vc_z']],
    out7[['ying_vol_63', 'ying_vol_63_z']],
], axis=1)
df_22y = pd.concat([df_22y.reset_index(drop=True), factors_df.reset_index(drop=True)], axis=1)

# === Lift audit for each imported formula ===
print(f"\n=== Bear Lift Audit (2022-2024) for imported formulas ===")
results = []
for fac_name in ['ying_mom_skip21_63', 'ying_mr_20', 'ying_trend_50_200',
                  'ying_str_5', 'ying_hp_252', 'ying_vc', 'ying_vol_63']:
    sub = df_22y[['date', fac_name, 'net20']].dropna()
    if len(sub) < 100:
        continue
    sub['_r'] = sub.groupby('date')[fac_name].rank(pct=True, na_option='keep')
    valid = sub['_r'].notna()
    top = sub[valid & (sub['_r'] >= 0.8)]['net20'].mean()
    bot = sub[valid & (sub['_r'] <= 0.2)]['net20'].mean()
    lift = top - bot
    results.append({
        'factor': fac_name,
        'lift': lift,
        'top_q': top,
        'bot_q': bot,
        'n': len(sub),
    })

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(results_df.to_string(index=False))