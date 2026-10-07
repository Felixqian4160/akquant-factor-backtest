"""V30: 移植 Quantactix A 股学术因子 (Liquidity + Risk + Past Returns + 部分 Value)
约 20 个简单因子, 用 HS300 panel 数据构造

来源: Quantactix/ChinaAShareEquityCharacteristics (Liu-Stambaugh-Tian 2019)

要移植的因子:
  Liquidity: size, size3, turnm, turna, ami, dtvm, dtva, vdtv
  Risk: tv (total volatility), beta, m1 (1 month momentum)
  Past Returns: m1, m3, m6, m11, m24, mchg, 52w, mdr, pr, season
  Value: bm, ep
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

print("=== V30: Quantactix 学术因子移植到 HS300 panel ===\n")

t0 = time.time()
df = pl.read_parquet(PANEL).select([
    'trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
    'pct_chg', 'cap', 'circ_cap', 'pe', 'pb', 'bps', 'roe', 'roa'
])
df_pd = df.to_pandas()
df_pd = df_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
print(f"Loaded: {df_pd.shape}, time: {time.time()-t0:.1f}s")

# === Compute formulas ===
print("\n=== Computing Quantactix-style academic factors ===")

# === Liquidity ===
print("\n--- Liquidity ---")

# size = log(流通市值)
print("1. size ...")
df_pd['l_size'] = np.log(df_pd['circ_cap'].clip(lower=1))

# size3 = size^3
print("2. size3 ...")
df_pd['l_size3'] = df_pd['l_size'] ** 3

# turnm = 月换手率 (近 21d) = 日成交额/流通市值 求平均
# Note: amount is 成交额 (in yuan), circ_cap is 流通市值
# 我们用近 21d (月) rolling mean
print("3. turnm (月换手率) ...")
df_pd['daily_turnover'] = df_pd['amount'] / df_pd['circ_cap'].clip(lower=1)
df_pd['l_turnm'] = df_pd.groupby('ts_code')['daily_turnover'].transform(
    lambda x: np.log(x.rolling(21).mean().clip(lower=1e-9))
)

# turna (年换手率) = 近 252d 平均日换手率
print("4. turna (年换手率) ...")
df_pd['l_turna'] = df_pd.groupby('ts_code')['daily_turnover'].transform(
    lambda x: np.log(x.rolling(252).mean().clip(lower=1e-9))
)

# ami (Amihud) = abs(ret)/成交额, 月平均
print("5. ami (Amihud 非流动性) ...")
df_pd['daily_ret'] = df_pd['close'].pct_change()
df_pd['l_ami'] = df_pd.groupby('ts_code').apply(
    lambda g: (g['daily_ret'].abs() / g['amount'].clip(lower=1)).rolling(21).mean()
).reset_index(level=0, drop=True)

# dtvm (月成交额, log)
print("6. dtvm ...")
df_pd['l_dtvm'] = np.log(df_pd.groupby('ts_code')['amount'].transform(
    lambda x: x.rolling(21).mean().clip(lower=1)
))

# dtva (年成交额, log)
print("7. dtva ...")
df_pd['l_dtva'] = np.log(df_pd.groupby('ts_code')['amount'].transform(
    lambda x: x.rolling(252).mean().clip(lower=1)
))

# vdtv (成交额波动率, 120d std)
print("8. vdtv ...")
df_pd['l_vdtv'] = df_pd.groupby('ts_code')['amount'].transform(
    lambda x: np.log(x.rolling(120).std().clip(lower=1))
)

# === Risk ===
print("\n--- Risk ---")

# tv = total volatility = daily ret rolling std (60d)
print("9. tv ...")
df_pd['r_tv'] = df_pd.groupby('ts_code')['daily_ret'].transform(
    lambda x: x.rolling(60).std()
)

# beta (60d rolling cov with market, 简化版)
print("10. beta (60d) ...")
# Compute market return: equal-weight daily mean close
market_pd = df_pd.groupby('trade_date')['close'].mean().reset_index()
market_pd.columns = ['trade_date', 'mkt_close']
market_pd['mkt_ret'] = market_pd['mkt_close'].pct_change()
df_pd = df_pd.merge(market_pd[['trade_date', 'mkt_ret']], on='trade_date', how='left')

def calc_beta_60(g):
    cov = g['daily_ret'].rolling(60).cov(g['mkt_ret'])
    var = g['mkt_ret'].rolling(60).var()
    return cov / var.replace(0, np.nan)

df_pd['r_beta'] = df_pd.groupby('ts_code').apply(
    lambda g: calc_beta_60(g)
).reset_index(level=0, drop=True)

# === Past Returns ===
print("\n--- Past Returns ---")

# m1, m3, m6, m11, m24 = past 1m, 3m, 6m, 11m, 24m momentum
for label, days in [('m1', 21), ('m3', 63), ('m6', 126), ('m11', 231), ('m24', 504)]:
    print(f"{label} (past {days}d) ...")
    df_pd[f'p_{label}'] = df_pd.groupby('ts_code')['close'].transform(
        lambda x: x.shift(21).pct_change(days)
    )

# mchg = momentum change = (m6 - m12) i.e. recent momentum vs older
print("mchg (m6 - m12) ...")
df_pd['p_mchg'] = df_pd.groupby('ts_code').apply(
    lambda g: g['close'].pct_change(126).shift(21) - g['close'].pct_change(252).shift(21)
).reset_index(level=0, drop=True)

# 52w = 52周新高 (relative to 252d rolling max)
print("52w (52周新高接近度) ...")
df_pd['p_52w'] = df_pd.groupby('ts_code').apply(
    lambda g: g['close'] / g['close'].rolling(252).max()
).reset_index(level=0, drop=True)

# mdr = maximum daily return in past 21d
print("mdr (近 21d 最大日收益) ...")
df_pd['p_mdr'] = df_pd.groupby('ts_code')['daily_ret'].transform(
    lambda x: x.rolling(21).max()
)

# pr = share price (log)
print("pr (股价对数) ...")
df_pd['p_pr'] = np.log(df_pd['close'].clip(lower=0.01))

# season = 12月效应 (dummy for month==12)
print("season ...")
df_pd['p_season'] = (pd.to_datetime(df_pd['trade_date']).dt.month == 12).astype(int)

# === Value ===
print("\n--- Value ---")

# bm = book-to-market = bps / cap (bvs per share / market cap per share)
# bps = 每股净资产
print("bm (账面市值比) ...")
df_pd['v_bm'] = df_pd['bps'] / df_pd['close'].clip(lower=0.01)

# ep = earnings to price = 1 / pe
print("ep (盈利收益率) ...")
df_pd['v_ep'] = 1.0 / df_pd['pe'].clip(lower=1)

# === Compute net20 ===
print(f"\n=== Computing net20 ===")
df_pd['open_t1'] = df_pd.groupby('ts_code')['open'].shift(-1)
df_pd['close_t21'] = df_pd.groupby('ts_code')['close'].shift(-21)
df_pd['net20'] = (df_pd['close_t21'] / df_pd['open_t1'] - 1) - 0.005
df_pd['date'] = pd.to_datetime(df_pd['trade_date']).dt.date

mask = (df_pd['date'] >= date(2022, 1, 1)) & (df_pd['date'] <= date(2024, 12, 31))
df_22y = df_pd[mask].copy().dropna(subset=['net20'])
print(f"2022-2024 samples: {len(df_22y)}")

# === Bear Lift Audit ===
print(f"\n=== Bear Lift Audit (2022-2024) ===")
factors_q = [
    'l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami',
    'l_dtvm', 'l_dtva', 'l_vdtv',
    'r_tv', 'r_beta',
    'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
    'p_52w', 'p_mdr', 'p_pr', 'p_season',
    'v_bm', 'v_ep',
]

results = []
for fac in factors_q:
    sub = df_22y[['date', fac, 'net20']].dropna()
    if len(sub) < 100:
        continue
    sub['_r'] = sub.groupby('date')[fac].rank(pct=True, na_option='keep')
    valid = sub['_r'].notna()
    top = sub[valid & (sub['_r'] >= 0.8)]['net20'].mean()
    bot = sub[valid & (sub['_r'] <= 0.2)]['net20'].mean()
    lift = top - bot
    # Per-day corr
    pd_corr = sub[valid].groupby('date').apply(
        lambda g: g[fac].corr(g['net20']) if g[fac].std() > 0 else 0,
        include_groups=False
    ).mean()
    results.append({
        'factor': fac,
        'lift': lift,
        'top_q_mean': top,
        'bot_q_mean': bot,
        'mean_corr': pd_corr,
        'n_samples': len(sub),
    })

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"\n=== Quantactix 学术因子 bear lift (2022-2024) ===")
print(results_df.to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift'] > 0.01).sum()}")
print(f"lift > 0.005: {(results_df['lift'] > 0.005).sum()}")
print(f"lift < -0.01: {(results_df['lift'] < -0.01).sum()}")
print(f"lift < -0.005: {(results_df['lift'] < -0.005).sum()}")
print(f"Mean lift: {results_df['lift'].mean()*100:.3f}%")
print(f"Top 5 mean: {results_df.head(5)['lift'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v30_quantactix_factors_lift.csv', index=False)
print(f"\nSaved to evidence/v30_quantactix_factors_lift.csv")