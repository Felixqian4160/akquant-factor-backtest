"""Step 0-2: 用户合同熊市策略框架 (向量化版)
- Step 0: 物理分层 (v10_1_zig=label, idx=timing, 404=stock-pick, 规则=filter)
- Step 1: 因子预处理
- Step 2: 单变量熊市 lift audit (404 因子, 整体 2022-2024) — 不删除, 只筛选

向量化计算: 用 np.argsort 一次性给每 (date, stock) 排序
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

# ========== Step 0: 物理分层 ==========
all_cols = pl.scan_parquet(PANEL).collect_schema().names()
LABEL_COLS = [c for c in all_cols if c.startswith('v10_1_')]
TIMING_COLS = [c for c in all_cols if c.startswith('idx_')]
print(f"Layer | Count | Cols")
print(f"Label | {len(LABEL_COLS)} | {LABEL_COLS}")
print(f"Timing | {len(TIMING_COLS)} | {TIMING_COLS}")

STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'cap', 'circ_cap', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate',
            'volume', 'amount')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]
print(f"StockPick | {len(STOCKPICK_COLS)} (gtja + alpha + talib + base_fin)")
print(f"  alpha: {len([c for c in STOCKPICK_COLS if c.startswith('alpha_')])}")
print(f"  gtja: {len([c for c in STOCKPICK_COLS if c.startswith('gtja_')])}")
print(f"  talib: {len([c for c in STOCKPICK_COLS if c.startswith('talib_')])}")
print(f"  base: {len([c for c in STOCKPICK_COLS if c not in STOCKPICK_PREFIXES])}")

# ========== Step 1: 预处理 ==========
print(f"\n=== Loading panel ===")
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close'] + STOCKPICK_COLS)
df_pd = df.to_pandas()
df_pd['trade_date'] = pd.to_datetime(df_pd['trade_date'])
df_pd = df_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
df_pd['open_t1'] = df_pd.groupby('ts_code')['open'].shift(-1)
df_pd['close_t21'] = df_pd.groupby('ts_code')['close'].shift(-21)
df_pd['net20'] = (df_pd['close_t21'] / df_pd['open_t1'] - 1) - 0.005
df_pd['date'] = df_pd['trade_date'].dt.date
mask = (df_pd['date'] >= date(2022, 1, 1)) & (df_pd['date'] <= date(2024, 12, 31))
df_22y = df_pd[mask].copy().dropna(subset=['net20'])
print(f"2022-2024 samples: {len(df_22y)}, dates: {df_22y['date'].nunique()}")

# ========== Step 2: 单变量 lift audit (向量化) ==========
# 对每个 (date, fac), 计算 rank within date, 然后 quintile mean net20
print(f"\n=== Step 2: VECTORIZED bear lift audit (404 factors) ===")
df_22y = df_22y.sort_values(['date', 'ts_code']).reset_index(drop=True)
dates = df_22y['date'].unique()
date_idx = df_22y['date'].values

results = []
for fac in STOCKPICK_COLS:
    vals = df_22y[fac].values
    if np.isnan(vals).all() or len(np.unique(vals[~np.isnan(vals)])) < 10:
        continue
    # Per-date rank
    rank = np.full(len(vals), np.nan)
    for d in dates:
        mask_d = date_idx == d
        sub_vals = vals[mask_d]
        sub_rank = pd.Series(sub_vals).rank(pct=True, na_option='keep').values
        rank[mask_d] = sub_rank
    df_22y['_rank'] = rank
    valid = ~np.isnan(rank)
    top_mask = valid & (rank >= 0.8)
    bot_mask = valid & (rank <= 0.2)
    if top_mask.sum() < 50 or bot_mask.sum() < 50:
        continue
    net = df_22y['net20'].values
    top_q_mean = net[top_mask].mean()
    bot_q_mean = net[bot_mask].mean()
    lift = top_q_mean - bot_q_mean
    # Per-day corr
    per_day_corr = []
    for d in dates:
        dmask = (date_idx == d) & valid
        if dmask.sum() < 5:
            continue
        r = np.corrcoef(rank[dmask], net[dmask])[0, 1]
        if not np.isnan(r):
            per_day_corr.append(r)
    mean_corr = np.mean(per_day_corr) if per_day_corr else 0
    results.append({
        'factor': fac,
        'lift_top_q': lift,
        'top_q_mean': top_q_mean,
        'bot_q_mean': bot_q_mean,
        'mean_corr': mean_corr,
        'n_top': top_mask.sum(),
        'n_bot': bot_mask.sum(),
    })

results_df = pd.DataFrame(results).sort_values('lift_top_q', ascending=False)
print(f"Valid factors: {len(results_df)}")

print(f"\n=== TOP 30 ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== BOTTOM 30 ===")
print(results_df.tail(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift_top_q'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift_top_q'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift_top_q'] > 0.10).sum()}")
print(f"lift > 0.20: {(results_df['lift_top_q'] > 0.20).sum()}")
print(f"lift < -0.01: {(results_df['lift_top_q'] < -0.01).sum()}")
print(f"lift < -0.05: {(results_df['lift_top_q'] < -0.05).sum()}")
print(f"Mean lift: {results_df['lift_top_q'].mean()*100:.3f}%")
print(f"Median lift: {results_df['lift_top_q'].median()*100:.3f}%")
print(f"Top 10 mean lift: {results_df.head(10)['lift_top_q'].mean()*100:.3f}%")
print(f"Top 20 mean lift: {results_df.head(20)['lift_top_q'].mean()*100:.3f}%")
print(f"Top 50 mean lift: {results_df.head(50)['lift_top_q'].mean()*100:.3f}%")
print(f"Top 80 mean lift: {results_df.head(80)['lift_top_q'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v28_bear_lift_404factors_2022_2024.csv', index=False)
print(f"\nSaved CSV")