"""V29B: 设计 A 股特有熊市因子

基础字段 (已有): open/high/low/close/volume/amount/pct_chg

A 股特有现象:
- 涨跌停 10%/20%
- T+1 制度
- 跳空缺口常见
- 政策/事件驱动
- 小票高换手

我们设计 8 个 A 股特有因子:
1. gap_down_1d: 负跳空 → 反转信号
2. lower_shadow_ratio: 下影线 → 探底回升
3. upper_shadow_ratio: 上影线 → 冲高回落
4. amplitude_20: 振幅 → 高波动
5. close_position: 收盘位置 → 强弱
6. volume_reversal_5: 量价背离
7. consecutive_down_3d: 连续下跌
8. distance_to_60d_low: 距 60 日新低距离
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

print("=== V29B: A 股特有因子设计 ===")

t0 = time.time()
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg'])
df_pd = df.to_pandas()
df_pd = df_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
print(f"Loaded: {df_pd.shape}, time: {time.time()-t0:.1f}s")

# Compute per-stock formulas (vectorized via groupby)
print("\n=== Computing A-share-specific factors ===")

# 1. gap_down_1d = (open - prev_close) / prev_close, negative = down gap
print("1. gap_down_1d ...")
df_pd['prev_close'] = df_pd.groupby('ts_code')['close'].shift(1)
df_pd['gap_pct'] = (df_pd['open'] - df_pd['prev_close']) / df_pd['prev_close']
# Invert: more negative = more bearish gap → more reversal potential
df_pd['a_gap_down'] = -df_pd['gap_pct']

# 2. lower_shadow_ratio = (close - low) / (high - low)
# Higher = close near high = strong recovery from low (reversal signal)
print("2. lower_shadow_ratio ...")
df_pd['hl_range'] = df_pd['high'] - df_pd['low']
df_pd['a_lower_shadow'] = np.where(df_pd['hl_range'] > 0,
                                    (df_pd['close'] - df_pd['low']) / df_pd['hl_range'],
                                    np.nan)

# 3. upper_shadow_ratio = (high - close) / (high - low)
# Higher = close near low = weak close after rally (reversal down signal)
print("3. upper_shadow_ratio ...")
df_pd['a_upper_shadow'] = np.where(df_pd['hl_range'] > 0,
                                    (df_pd['high'] - df_pd['close']) / df_pd['hl_range'],
                                    np.nan)

# 4. amplitude_20 = rolling 20d hl_range / close
# Higher = more volatile
print("4. amplitude_20 ...")
df_pd['a_amp_20'] = df_pd.groupby('ts_code').apply(
    lambda g: (g['high'] - g['low']).rolling(20).mean() / g['close']
).reset_index(level=0, drop=True)

# 5. close_position = (close - open) / (high - low)
# Range [-1, 1]: positive = bullish, negative = bearish
print("5. close_position ...")
df_pd['a_close_pos'] = np.where(df_pd['hl_range'] > 0,
                                 (df_pd['close'] - df_pd['open']) / df_pd['hl_range'],
                                 np.nan)

# 6. volume_reversal_5 = (volume[t] / mean(volume[t-5:t])) * sign(close[t] - close[t-5])
# High vol + down close = bearish, high vol + up close = bullish
print("6. volume_reversal_5 ...")
df_pd['vol_ma5'] = df_pd.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(5).mean())
df_pd['ret_5d'] = df_pd.groupby('ts_code')['close'].transform(lambda x: x.pct_change(5))
df_pd['a_vol_rev_5'] = np.where(df_pd['vol_ma5'] > 0,
                                 (df_pd['vol'] / df_pd['vol_ma5']) * df_pd['ret_5d'],
                                 np.nan)

# 7. consecutive_down_3d = count of last 3 days close < prev_close
print("7. consecutive_down_3d ...")
df_pd['down_day'] = (df_pd['close'] < df_pd['prev_close']).astype(int)
df_pd['a_consec_down_3'] = df_pd.groupby('ts_code')['down_day'].transform(
    lambda x: x.rolling(3).sum()
)

# 8. distance_to_60d_low = (close - rolling_min(low, 60)) / close
# Higher = close near 60d high (uptrend); Negative = below
print("8. distance_to_60d_low ...")
df_pd['a_dist_60d_low'] = df_pd.groupby('ts_code').apply(
    lambda g: (g['close'] - g['low'].rolling(60).min()) / g['close']
).reset_index(level=0, drop=True)

# === Compute net20 ===
print(f"\n=== Computing net20 ===")
df_pd['open_t1'] = df_pd.groupby('ts_code')['open'].shift(-1)
df_pd['close_t21'] = df_pd.groupby('ts_code')['close'].shift(-21)
df_pd['net20'] = (df_pd['close_t21'] / df_pd['open_t1'] - 1) - 0.005
df_pd['date'] = pd.to_datetime(df_pd['trade_date']).dt.date

mask = (df_pd['date'] >= date(2022, 1, 1)) & (df_pd['date'] <= date(2024, 12, 31))
df_22y = df_pd[mask].copy().dropna(subset=['net20'])
print(f"2022-2024 samples: {len(df_22y)}")

# === Bear Lift Audit for each A-share factor ===
print(f"\n=== Bear Lift Audit (2022-2024) ===")
factors_a = ['a_gap_down', 'a_lower_shadow', 'a_upper_shadow',
              'a_amp_20', 'a_close_pos', 'a_vol_rev_5',
              'a_consec_down_3', 'a_dist_60d_low']

results = []
for fac in factors_a:
    sub = df_22y[['date', fac, 'net20']].dropna()
    if len(sub) < 100:
        continue
    sub['_r'] = sub.groupby('date')[fac].rank(pct=True, na_option='keep')
    valid = sub['_r'].notna()
    top = sub[valid & (sub['_r'] >= 0.8)]['net20'].mean()
    bot = sub[valid & (sub['_r'] <= 0.2)]['net20'].mean()
    # Per-day corr
    pd_corr = sub[valid].groupby('date').apply(
        lambda g: g[fac].corr(g['net20']) if g[fac].std() > 0 else 0,
        include_groups=False
    ).mean()
    results.append({
        'factor': fac,
        'lift': top - bot,
        'top_q_mean': top,
        'bot_q_mean': bot,
        'mean_corr': pd_corr,
        'n_samples': len(sub),
    })

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"\n=== A 股特有因子 bear lift (2022-2024) ===")
print(results_df.to_string(index=False))

# === Compare with existing top factors ===
print(f"\n=== 比较: 与之前 alpha_alpha065 对比 ===")
print(f"alpha_alpha065: lift=+9.94%, top_q_mean=+6.68%, bot_q_mean=-3.26%")
print(f"V29B 最佳: lift={results_df.iloc[0]['lift']*100:.2f}%, top_q={results_df.iloc[0]['top_q_mean']*100:.2f}%, bot_q={results_df.iloc[0]['bot_q_mean']*100:.2f}%")

# Save
results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v29b_ashare_factors_lift.csv', index=False)
print(f"\nSaved to evidence/v29b_ashare_factors_lift.csv")