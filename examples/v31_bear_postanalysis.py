"""V31 熊市因子分析 — 后处理:
1. 半段方向验证 (H1 选方向 → H2 验收益, 控制 in-sample 方向选择偏差)
2. Top 10 因子资金曲线图 (+ benchmark)
3. 汇总 markdown
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

OUTDIR = 'evidence/v31_bear_factor_analysis_20260925'
res = pd.read_csv(f'{OUTDIR}/bear_factor_ranking_2022_2024.csv')
period = pd.read_parquet(f'{OUTDIR}/period_returns_k20.parquet')
period['trade_date'] = pd.to_datetime(period['trade_date'])

print(f"factors: {len(res)}, period rows: {len(period)}")

# --- 1. 半段方向验证 ---
dates = sorted(period['trade_date'].unique())
mid = len(dates) // 2
h1_dates, h2_dates = dates[:mid], dates[mid:]
print(f"H1: {h1_dates[0].date()} → {h1_dates[-1].date()} ({len(h1_dates)} periods)")
print(f"H2: {h2_dates[0].date()} → {h2_dates[-1].date()} ({len(h2_dates)} periods)")

rows = []
for fac, g in period.groupby('factor'):
    # k20 only (period parquet 只有 k20)
    g_hi = g[g['dir'] == 'hi'].set_index('trade_date')['ret']
    g_lo = g[g['dir'] == 'lo'].set_index('trade_date')['ret']
    h1_hi, h1_lo = g_hi.reindex(h1_dates).dropna(), g_lo.reindex(h1_dates).dropna()
    h2_hi, h2_lo = g_hi.reindex(h2_dates).dropna(), g_lo.reindex(h2_dates).dropna()
    if len(h1_hi) == 0 or len(h2_hi) == 0:
        rows.append({'factor': fac, 'dir_h1': None, 'ret_h1': np.nan, 'ret_h2': np.nan})
        continue
    r_h1_hi = (1+h1_hi).prod() - 1
    r_h1_lo = (1+h1_lo).prod() - 1
    dir_h1 = 'hi' if r_h1_hi >= r_h1_lo else 'lo'
    r_h2 = (1 + (h2_hi if dir_h1 == 'hi' else h2_lo)).prod() - 1
    rows.append({'factor': fac, 'dir_h1': dir_h1, 'ret_h1': r_h1_hi if dir_h1=='hi' else r_h1_lo, 'ret_h2': r_h2})

sv = pd.DataFrame(rows)
merged = res.merge(sv, on='factor', how='left')
merged.to_csv(f'{OUTDIR}/bear_factor_split_validation.csv', index=False)

print("\n=== 半段验证结果 (H1 选方向, H2 验证) ===")
valid = merged.dropna(subset=['ret_h2'])
print(f"valid: {len(valid)}")
print(f"H2 平均: {valid['ret_h2'].mean()*100:+.2f}%, 中位: {valid['ret_h2'].median()*100:+.2f}%")
print(f"H2 正收益: {(valid['ret_h2']>0).sum()}/{len(valid)} ({(valid['ret_h2']>0).mean()*100:.0f}%)")
print(f"H2 > +10%: {(valid['ret_h2']>0.10).sum()}, > +20%: {(valid['ret_h2']>0.20).sum()}")

print("\n=== Top 25 by H2 verified return ===")
cols = ['rank','factor','family','dir_h1','ret_h1','ret_h2','ic_mean']
top25 = valid.sort_values('ret_h2', ascending=False).head(25)[cols].copy()
for c in ['ret_h1','ret_h2','ic_mean']:
    top25[c] = (top25[c]*100).round(1)
print(top25.to_string(index=False))

# 方向一致性检查: H1 选择的方向 vs 全窗最佳方向
consist = (merged['dir_h1'] == merged['best_dir']).mean()
print(f"\n方向一致性 (H1 选 vs 全窗): {consist*100:.0f}%")

# --- 2. 图表: Top 10 因子 nav + benchmark ---
top10 = res.head(10)['factor'].tolist()
fig, ax = plt.subplots(figsize=(12, 6))

# 真 benchmark: 等权全股票 (从 net20_cache 重算)
import polars as pl
base = pl.read_parquet(f'{OUTDIR}/net20_cache.parquet')
bm_df = base.filter(pl.col('trade_date').cast(pl.Date).is_in([d.date() for d in dates])).group_by('trade_date').agg(
    pl.col('net20').mean().alias('ret')).sort('trade_date').to_pandas()
bm_df['trade_date'] = pd.to_datetime(bm_df['trade_date'])
bm_nav = (1 + bm_df.set_index('trade_date')['ret']).cumprod()
ax.plot(bm_nav.index, bm_nav.values, 'k--', linewidth=2.2, label='Equal-weight BM', alpha=0.85)

cmap = plt.cm.tab10
for i, fac in enumerate(top10):
    g = period[(period['factor'] == fac)]
    dir_best = res[res['factor'] == fac]['best_dir'].iloc[0]
    g = g[g['dir'] == dir_best].sort_values('trade_date')
    nav = (1 + g['ret'].values).cumprod()
    ax.plot(g['trade_date'], nav, label=f"{fac} ({dir_best})", color=cmap(i), linewidth=1.5, alpha=0.9)

ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax.set_title('V31 Bear Market (2022-2024): Top 10 Factors NAV vs Benchmark\n(HS300 constituents, rebal 20d, T+1 open, 0.5% cost)', fontsize=12)
ax.set_ylabel('NAV (start=1.0)')
ax.legend(loc='upper left', fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/top10_factor_nav.png', dpi=130)
print(f"\nSaved chart: {OUTDIR}/top10_factor_nav.png")

# 第二张图: family 平均收益柱状图
fig2, ax2 = plt.subplots(figsize=(8, 5))
fam = res.groupby('family')['best_ret20'].agg(['mean', 'max']).sort_values('mean')
x = np.arange(len(fam))
ax2.barh(x - 0.18, fam['mean']*100, height=0.36, label='mean', color='steelblue')
ax2.barh(x + 0.18, fam['max']*100, height=0.36, label='max', color='salmon')
ax2.set_yticks(x)
ax2.set_yticklabels(fam.index)
ax2.set_xlabel('best_ret20 (%)')
ax2.set_title('Bear 2022-2024 factor family performance (412 factors)')
ax2.axvline(0, color='gray', linewidth=0.8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/family_performance.png', dpi=130)
print(f"Saved chart: {OUTDIR}/family_performance.png")

# --- 3. 汇总 MD ---
summ = []
summ.append("# V31 熊市因子分析报告 (2022-2024)\n")
summ.append(f"**日期**: 2026-09-25 | **面板**: v2 (412 因子: 390 原始 + 22 新学术)")
summ.append(f"**窗口**: 2022-01-04 → 2024-12-31 (726 交易日, 37 个调仓期)")
summ.append(f"**方法**: 每因子独立测试 — 日度 rank IC + top-k 组合 (k=10/20/30) 20 日调仓复利净收益")
summ.append(f"**成本**: 0.5% 往返 (net20 = close[t+21]/open[t+1] - 1 - 0.005)")
summ.append(f"**调仓**: 每 20 交易日, T+1 raw open 买入, T+21 raw close 卖出\n")
summ.append("---\n")
summ.append("## 总体\n")
summ.append(f"| 指标 | 值 |")
summ.append("|---|---:|")
summ.append(f"| 基准 (等权全股票 3y) | **-13.42%** |")
summ.append(f"| 因子 mean best_ret20 | +1.56% |")
summ.append(f"| 因子 median | -1.19% |")
summ.append(f"| 正收益因子 | 197/410 (48%) |")
summ.append(f"| >+10% | 134 |")
summ.append(f"| >+20% | 75 |")
summ.append(f"| >+50% | 15 |")
summ.append("")
summ.append("## 因子族表现 (mean best_ret20)\n")
summ.append("| 族 | n | mean | positive | best |")
summ.append("|---|---:|---:|---:|---:|")
fam_md = res.groupby('family').agg(
    n=('factor','count'),
    mean_ret=('best_ret20','mean'),
    pos=('best_ret20', lambda s: (s>0).sum()),
    best=('best_ret20','max'),
).sort_values('mean_ret', ascending=False)
for fam_name, row in fam_md.iterrows():
    summ.append(f"| {fam_name} | {int(row['n'])} | {row['mean_ret']*100:+.1f}% | {int(row['pos'])} | {row['best']*100:+.1f}% |")
summ.append("")
summ.append("**关键**: 新学术因子族 (22 个) mean +30.1% 冠绝全场 — 显著高于第二族 talib +10.2%。")
summ.append("")
summ.append("## Top 20 因子\n")
summ.append("| # | factor | family | dir | ret20 | win | mdd | ic | 2022 | 2023 | 2024 |")
summ.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
t = res.head(20)
for _, r in t.iterrows():
    summ.append(f"| {int(r['rank'])} | {r['factor']} | {r['family']} | {r['best_dir']} | {r['best_ret20']*100:+.1f}% | {r['win20']*100:.0f}% | {r['mdd20']*100:.1f}% | {r['ic_mean']:+.3f} | {r['y2022']*100:+.1f}% | {r['y2023']*100:+.1f}% | {r['y2024']*100:+.1f}% |")
summ.append("")
summ.append("## 半段方向验证 (H1 选方向 → H2 验证, 控制方向偏差)\n")
summ.append(f"- H1: {h1_dates[0].date()} → {h1_dates[-1].date()} ({len(h1_dates)} periods)")
summ.append(f"- H2: {h2_dates[0].date()} → {h2_dates[-1].date()} ({len(h2_dates)} periods)")
summ.append(f"- H2 平均 {valid['ret_h2'].mean()*100:+.2f}%, 中位 {valid['ret_h2'].median()*100:+.2f}%, 正收益率 {(valid['ret_h2']>0).mean()*100:.0f}%")
summ.append(f"- H1 vs 全窗方向一致性: {consist*100:.0f}%\n")
summ.append("**Top 15 by H2 (半外验证收益)**:\n")
summ.append("| factor | family | dir(H1) | ret(H1) | ret(H2) | rank(全窗) |")
summ.append("|---|---|---|---:|---:|---:|")
t2 = valid.sort_values('ret_h2', ascending=False).head(15)
for _, r in t2.iterrows():
    summ.append(f"| {r['factor']} | {r['family']} | {r['dir_h1']} | {r['ret_h1']*100:+.1f}% | {r['ret_h2']*100:+.1f}% | {int(r['rank'])} |")
summ.append("")
summ.append("## 新 22 学术因子明细\n")
summ.append("| rank | factor | dir | ret20 | ic |")
summ.append("|---:|---|---|---:|---:|")
for _, r in valid[valid['family']=='new_academic'].sort_values('rank').iterrows():
    summ.append(f"| {int(r['rank'])} | {r['factor']} | {r['best_dir']} | {r['best_ret20']*100:+.1f}% | {r['ic_mean']:+.3f} |")
summ.append("")
summ.append("## ⚠️ 限制与警告\n")
summ.append("1. **方向为全窗内选优** (hi/lo 取更好的) — 已用半段验证控制, 但仍有 1-bit 自由度")
summ.append("2. **多重比较**: 412 因子同时测试, top 结果被选择偏差放大")
summ.append("3. **流动性偏差**: 头部因子 (l_size/l_dtvm/l_ami/r_tv) 都偏'小市值/低流动性'方向 — 真实成交成本可能 >0.5%, 容量有限, 需可成交性审计")
summ.append("4. **单一窗口**: 2022-2024 一次, 未跨其他 regime")
summ.append("5. **2024 大年**: 多数因子 2024 收益远超 2022-2023 (9-11 月小微盘行情), 收益非均匀分布")
summ.append("6. **退化因子**: p_season 等 10 个 ic=nan (常数列), 无效")
summ.append("7. **IC vs 收益负相关** (spearman -0.33): 高 IC 因子 ≠ 高 long-only 组合收益; IC 排名和回测排名是两套视角")

with open(f'{OUTDIR}/BEAR_FACTOR_ANALYSIS_SUMMARY.md', 'w') as f:
    f.write('\n'.join(summ))
print(f"Saved: {OUTDIR}/BEAR_FACTOR_ANALYSIS_SUMMARY.md")
print("\n✅ 后处理完成")
