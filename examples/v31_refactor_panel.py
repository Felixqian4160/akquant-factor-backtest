"""V31: 因子面板重构 (Step 0 + Step 1)
- Step 0: 物理分层 (4 个文件/列分组)
- Step 1: 因子预处理 (去极值 + 缺失值 + 中性化 + Z-score)

不删除任何因子, 只生成"preprocessed"版本供后续使用
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time
import os

PANEL_IN = "data/wavehunter_hs300_with_talib_20260924.parquet"
PANEL_OUT = "data/wavehunter_hs300_preprocessed_20260925.parquet"
PANEL_TIMING = "data/idx_timing_20260925.parquet"  # 单独保存 idx 4
PANEL_LABEL = "data/v10_1_zig_labels_20260925.parquet"  # 单独保存 v10_1_zig

print("=" * 80)
print("V31: 因子面板重构 — Step 0 + Step 1")
print("=" * 80)

# === Step 0: 物理分层 ===
print("\n=== Step 0: 物理分层 ===")
all_cols = pl.scan_parquet(PANEL_IN).collect_schema().names()

# 标签层 (v10_1_zig) — 物理隔离, 单独保存
LABEL_COLS = [c for c in all_cols if c.startswith('v10_1_')]
print(f"标签层 (v10_1_zig): {len(LABEL_COLS)} 列 → {PANEL_LABEL}")
print(f"  字段: {LABEL_COLS}")

# 择时层 (idx) — 单独保存
TIMING_COLS = [c for c in all_cols if c.startswith('idx_')]
print(f"\n择时层 (idx): {len(TIMING_COLS)} 列 → {PANEL_TIMING}")
print(f"  字段: {TIMING_COLS}")

# 选股层 (gtja + alpha + talib + base_fin)
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'cap', 'circ_cap', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate',
            'volume', 'amount')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]
print(f"\n选股层 (gtja + alpha + talib + base_fin): {len(STOCKPICK_COLS)} 列")
print(f"  alpha: {len([c for c in STOCKPICK_COLS if c.startswith('alpha_')])}")
print(f"  gtja: {len([c for c in STOCKPICK_COLS if c.startswith('gtja_')])}")
print(f"  talib: {len([c for c in STOCKPICK_COLS if c.startswith('talib_')])}")
print(f"  base: {len([c for c in STOCKPICK_COLS if not c.startswith(STOCKPICK_PREFIXES)])}")

# 过滤层 (规则, 不入 panel)
print(f"\n过滤层 (规则): ST / 流动性 / 涨跌停 / 上市天数")
print(f"  规则不存储, 在 sim 中实现")

# === Save isolated layers ===
print(f"\n=== Save isolated layers ===")
df = pl.read_parquet(PANEL_IN)

# v10_1_zig labels — 单独保存
df.select(['trade_date', 'ts_code'] + LABEL_COLS).write_parquet(PANEL_LABEL)
print(f"  Saved: {PANEL_LABEL}")

# idx timing — 单独保存 (加 idx_zscore 标准化版本)
df_idx = df.select(['trade_date', 'ts_code'] + TIMING_COLS).sort(['ts_code', 'trade_date'])
df_idx_pd = df_idx.to_pandas()
for c in TIMING_COLS:
    if c == 'idx_close':
        # 保留原始 close, 加 log return
        df_idx_pd['idx_ret_1d'] = np.log(df_idx_pd.groupby('ts_code')['idx_close'].transform(lambda x: x.pct_change() + 1))
    else:
        # Z-score per date
        df_idx_pd[c + '_z'] = df_idx_pd.groupby('trade_date')[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
df_idx_pl = pl.from_pandas(df_idx_pd)
df_idx_pl.write_parquet(PANEL_TIMING)
print(f"  Saved: {PANEL_TIMING}")

# === Step 1: 因子预处理 (404 选股层) ===
print(f"\n=== Step 1: 因子预处理 (404 选股层) ===")
print("\n规则:")
print("  1. 去极值: 3σ clip")
print("  2. 缺失值: cap (市值) 中位数填充")
print("  3. 中性化: 对 log(cap) OLS 取残差")
print("  4. 标准化: 截面 Z-score")
print("  5. 财务因子滞后: 滞后 60d (季度)")
print("  6. 技术因子: T 日收盘后, T+1 开盘交易 (已经在数据层完成)")

t0 = time.time()
df_sp = df.select(['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount']
                  + STOCKPICK_COLS).sort(['ts_code', 'trade_date'])
df_sp_pd = df_sp.to_pandas()
print(f"\nLoaded stock-pick panel: {df_sp_pd.shape}, time: {time.time()-t0:.1f}s")

# Compute log_cap for neutralization
df_sp_pd['log_cap'] = np.log(df_sp_pd['cap'].clip(lower=1e6))

# === Process each factor ===
print(f"\nProcessing {len(STOCKPICK_COLS)} factors...")

# Identify financial factors (need lag)
FIN_FACTOR_LIST = ('pe', 'pb', 'bps', 'roe', 'roe_waa', 'roa',
                   'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
                   'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
                   'quick_ratio', 'debt_to_assets', 'assets_turn')

processed_cols = []
t1 = time.time()
for fi, fac in enumerate(STOCKPICK_COLS):
    col_data = df_sp_pd[fac].copy()

    # 1. 缺失值: cap (市值) 中位数填充 (cross-sectional median by date)
    if col_data.isna().any():
        medians = df_sp_pd.groupby('trade_date')['cap'].transform('median')
        col_data = col_data.fillna(medians)
        # 如果还有 NaN (整日没有 cap), 用全局中位数
        if col_data.isna().any():
            col_data = col_data.fillna(col_data.median())

    # 2. 财务因子滞后 60d
    if fac in FIN_FACTOR_LIST:
        col_data = df_sp_pd.groupby('ts_code')[fac].shift(60)
        # 再次填充
        if col_data.isna().any():
            medians = df_sp_pd.groupby('trade_date')['cap'].transform('median')
            col_data = col_data.fillna(medians)
            if col_data.isna().any():
                col_data = col_data.fillna(col_data.median())

    # 3. 去极值: 3σ clip (per date)
    def clip_3sigma(g):
        m = g.mean()
        s = g.std()
        if s == 0 or np.isnan(s):
            return g
        return g.clip(lower=m - 3*s, upper=m + 3*s)
    col_data = df_sp_pd.groupby('trade_date').apply(
        lambda g: pd.Series(clip_3sigma(col_data.loc[g.index]).values, index=g.index),
        include_groups=False
    ) if False else df_sp_pd.groupby('trade_date')[fac].transform(clip_3sigma)

    # 4. 中性化: 对 log_cap OLS 取残差 (per date)
    def neutralize(g):
        if g.std() == 0 or np.isnan(g.std()):
            return g * 0
        log_cap_g = df_sp_pd.loc[g.index, 'log_cap']
        if log_cap_g.std() == 0 or np.isnan(log_cap_g.std()):
            return g * 0
        # OLS: g = a + b * log_cap + residual
        x = log_cap_g.values
        y = g.values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 30:
            return pd.Series(np.nan, index=g.index)
        x_v = x[valid]
        y_v = y[valid]
        x_mean = x_v.mean()
        y_mean = y_v.mean()
        denom = ((x_v - x_mean) ** 2).sum()
        if denom == 0:
            return pd.Series(np.nan, index=g.index)
        beta = ((x_v - x_mean) * (y_v - y_mean)).sum() / denom
        alpha = y_mean - beta * x_mean
        resid = y - alpha - beta * x
        return pd.Series(resid, index=g.index)

    neutralized = df_sp_pd.groupby('trade_date').apply(
        lambda g: neutralize(col_data.loc[g.index]),
        include_groups=False
    )

    # 5. 标准化: 截面 Z-score (per date)
    zscore = df_sp_pd.groupby('trade_date')[fac].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    ) if False else neutralized.groupby(neutralized.index).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    ) if False else df_sp_pd.groupby('trade_date').apply(
        lambda g: (neutralized.loc[g.index] - neutralized.loc[g.index].mean()) / (neutralized.loc[g.index].std() + 1e-9),
        include_groups=False
    )

    # 重命名: 加 _pre 后缀
    new_name = fac + '_pre'
    df_sp_pd[new_name] = zscore.values if hasattr(zscore, 'values') else zscore
    processed_cols.append(new_name)

    if (fi + 1) % 50 == 0:
        print(f"  [{fi+1}/{len(STOCKPICK_COLS)}] {time.time()-t1:.1f}s elapsed")

print(f"\nProcessed {len(processed_cols)} factors in {time.time()-t1:.1f}s")

# === Save preprocessed panel ===
print(f"\n=== Save preprocessed panel ===")
out_cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'cap', 'log_cap'] + processed_cols
df_out = pl.from_pandas(df_sp_pd[out_cols])
df_out.write_parquet(PANEL_OUT)
print(f"  Saved: {PANEL_OUT} ({os.path.getsize(PANEL_OUT)/1e9:.2f} GB)")
print(f"  Columns: {len(df_out.columns)}")

# Summary
print(f"\n" + "=" * 80)
print(f"✅ 重构完成")
print(f"=" * 80)
print(f"\n输出文件:")
print(f"  标签层 (11 列, 物理隔离): {PANEL_LABEL}")
print(f"  择时层 (4 列, 单独保存): {PANEL_TIMING}")
print(f"  选股层 ({len(processed_cols)} 列, 预处理后): {PANEL_OUT}")
print(f"\n预处理流程:")
print(f"  1. 缺失值: cap 中位数填充")
print(f"  2. 财务因子滞后: 60d (1 季度)")
print(f"  3. 去极值: 3σ clip")
print(f"  4. 中性化: 对 log(cap) OLS 残差")
print(f"  5. Z-score: 截面标准化")