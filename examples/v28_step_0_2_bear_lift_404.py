"""Step 0-2 + Step 3: 用户合同熊市策略框架
- Step 0: 物理分层 (v10_1_zig=label, idx=timing, 404=stock-pick, 规则=filter)
- Step 1: 因子预处理 (404 个选股层)
- Step 2: 族内去冗余 404 → ~80
- Step 3: 市场状态识别 (idx 4)

先验证: 373 个因子在 2022-2024 整体熊市 lift audit
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

# ========== Step 0: 物理分层 ==========
LABEL_COLS = [c for c in pl.scan_parquet(PANEL).columns if c.startswith('v10_1_')]
print(f"Label layer (v10_1_*): {len(LABEL_COLS)} cols")

# Lazy load panel
df_lazy = pl.scan_parquet(PANEL)

# Get columns
all_cols = df_lazy.collect_schema().names()
TIMING_COLS = [c for c in all_cols if c.startswith('idx_')]
print(f"Timing layer (idx_*): {len(TIMING_COLS)} cols: {TIMING_COLS}")

# Stock-pick layer = 404 (gtja + alpha + talib + base)
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES)
                  or c in ('pe', 'pb', 'bps', 'cap', 'circ_cap', 'roe', 'roe_waa', 'roa',
                           'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
                           'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
                           'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate',
                           'volume', 'amount')]
print(f"Stock-pick layer: {len(STOCKPICK_COLS)} cols")

# ========== Step 1: 因子预处理 ==========
# Compute net20 for 2022-2024
print(f"\nLoading panel...")
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close'] + STOCKPICK_COLS)
print(f"  Shape: {df.shape}")
df_pd = df.to_pandas()
df_pd['trade_date'] = pd.to_datetime(df_pd['trade_date'])
df_pd = df_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
df_pd['open_t1'] = df_pd.groupby('ts_code')['open'].shift(-1)
df_pd['close_t21'] = df_pd.groupby('ts_code')['close'].shift(-21)
df_pd['net20'] = (df_pd['close_t21'] / df_pd['open_t1'] - 1) - 0.005
df_pd['date'] = df_pd['trade_date'].dt.date

# 2022-2024 整体
mask = (df_pd['date'] >= date(2022, 1, 1)) & (df_pd['date'] <= date(2024, 12, 31))
df_22y = df_pd[mask].copy().dropna(subset=['net20'])
print(f"  2022-2024 samples: {len(df_22y)}, dates: {df_22y['date'].nunique()}")

# ========== Step 2: 单变量熊市 lift audit (404 个因子, 未经去冗余) ==========
print(f"\n=== Step 2: Single-variable bear lift audit (404 factors, 2022-2024) ===")
results = []
for fac in STOCKPICK_COLS:
    sub = df_22y[['date', fac, 'net20']].dropna()
    if sub[fac].nunique() < 10:
        continue
    try:
        sub['pct_rank'] = sub.groupby('date')[fac].rank(pct=True, na_option='keep')
        sub = sub.dropna(subset=['pct_rank'])
        if len(sub) < 100:
            continue
        top_q = sub[sub['pct_rank'] >= 0.8]['net20']
        bot_q = sub[sub['pct_rank'] <= 0.2]['net20']
        if len(top_q) < 10 or len(bot_q) < 10:
            continue
        lift = top_q.mean() - bot_q.mean()
        # Per-day corr
        per_day_corr = sub.groupby('date').apply(
            lambda g: g['pct_rank'].corr(g['net20']) if g['pct_rank'].std() > 0 else 0
        )
        # Top 10 per day
        per_day_top10 = sub.groupby('date').apply(
            lambda g: g.nlargest(10, 'pct_rank')['net20'].mean()
        )
        results.append({
            'factor': fac,
            'lift_top_q': lift,
            'top_q_mean': top_q.mean(),
            'bot_q_mean': bot_q.mean(),
            'mean_corr': per_day_corr.mean(),
            'top10_per_day': per_day_top10.mean(),
            'n_samples': len(sub),
        })
    except Exception:
        continue

results_df = pd.DataFrame(results).sort_values('lift_top_q', ascending=False)
print(f"\nValid factors: {len(results_df)}")
print(f"\n=== TOP 30 by lift (2022-2024) ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== BOTTOM 30 ===")
print(results_df.tail(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total factors: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift_top_q'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift_top_q'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift_top_q'] > 0.10).sum()}")
print(f"lift < -0.01: {(results_df['lift_top_q'] < -0.01).sum()}")
print(f"lift < -0.05: {(results_df['lift_top_q'] < -0.05).sum()}")
print(f"Mean lift: {results_df['lift_top_q'].mean()*100:.3f}%")
print(f"Median lift: {results_df['lift_top_q'].median()*100:.3f}%")
print(f"Top 10 mean lift: {results_df.head(10)['lift_top_q'].mean()*100:.3f}%")
print(f"Top 20 mean lift: {results_df.head(20)['lift_top_q'].mean()*100:.3f}%")
print(f"Top 50 mean lift: {results_df.head(50)['lift_top_q'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v27_v28_bear_lift_404factors_2022_2024.csv', index=False)
print(f"\nSaved CSV")