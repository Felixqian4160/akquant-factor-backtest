"""V32 最终对比: AKQuant v3 vs 研究近似 vs 基准
1. 对账: 用 AK 口径 (open-open 出场, 0.7% 成本, 0.99 缓冲) 重算研究近似 → 应接近 AK 结果
2. 图表: AK nav vs research nav vs benchmark → PNG
3. 汇总 MD
"""
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

V3 = Path('evidence/v32_akquant_voting_v3_20260925')
IC = Path('evidence/v32_ic_voting_20260925')
PANEL = Path('data/wavehunter_hs300_v31_refactored_v2_20260925.parquet')

nav = pd.read_csv(V3 / 'nav.csv')
nav['dt'] = pd.to_datetime(nav['date'])
res_v3 = json.load(open(V3 / 'result.json'))
picks = json.load(open(V3 / 'picks.json'))
per = pd.read_csv(IC / 'best_config_periods.csv')
per['cum'] = (1 + per['ret']).cumprod()

print(f"AK v3: final={res_v3['metrics']['end_market_value']:,.0f} "
      f"({res_v3['metrics']['total_return_pct']:+.1f}%)")

# --- 1. 对账: AK 口径 net20 重算 ---
print("\n=== 对账: AK 口径研究重算 (open-open, 0.7%, 0.99) ===")
df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open'])
df = df.filter(pl.col('trade_date').is_between(
    pl.lit(pd.Timestamp('2022-01-01').to_pydatetime()),
    pl.lit(pd.Timestamp('2024-12-31').to_pydatetime()))).sort(['ts_code', 'trade_date'])
df = df.with_columns([
    pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
    pl.col('open').shift(-21).over('ts_code').alias('open_t21'),
])
# maps
dates = sorted(df['trade_date'].unique().to_list())
rebal_dates = dates[::20]
rebal_strs = [str(pd.Timestamp(d))[:10] for d in rebal_dates]

alt_rets = []
for rds in rebal_strs:
    pl_picks = picks.get(rds, {})
    if not pl_picks:
        continue
    day = df.filter(pl.col('trade_date') == pd.Timestamp(rds))
    day_map = dict(zip(day['ts_code'].to_list(), zip(day['open_t1'].to_list(), day['open_t21'].to_list())))
    rets = []
    for sym in pl_picks:
        pr = day_map.get(sym)
        if pr and pr[0] and pr[1]:
            rets.append(pr[1] / pr[0] - 1 - 0.007)
    r = 0.99 * float(np.mean(rets)) if rets else 0.0
    alt_rets.append(r)
alt_cum = float(np.prod([1 + r for r in alt_rets]) - 1)
print(f"AK 口径研究重算: {alt_cum*100:+.1f}%  (vs AK v3 实际 {res_v3['metrics']['total_return_pct']:+.1f}%)")
print(f"差额: {(alt_cum*100 - res_v3['metrics']['total_return_pct']):+.1f}pp")
print(f"(研究原版 +160.1%: 差异来自 open-open 出场/0.7%成本/0.99缓冲)")

# --- 2. 图表 ---
fig, ax = plt.subplots(figsize=(13, 6.5))

# AK nav (bounded)
ax.plot(nav['dt'], nav['value']/1e8, color='#1f77b4', linewidth=1.8, label=f'AKQuant real sim (final {res_v3["metrics"]["total_return_pct"]:+.0f}%)')

# research step nav at signal dates (value before period i)
per['dt'] = pd.to_datetime(per['date'])
step_x, step_y = [], []
step_x.append(per['dt'].iloc[0]); step_y.append(1.0)
for i in range(len(per)):
    step_x.append(per['dt'].iloc[i]); step_y.append(per['cum'].iloc[i])  # cum after period i plotted at date i (越界一点, 近似)
# 简化: 直接在 date[i] 画 cum[i-1] 更准
xs = [per['dt'].iloc[0]] + list(per['dt'])
ys = [1.0] + [1.0] + list(per['cum'].iloc[:-1])
ax.plot(per['dt'], [1.0] + list(per['cum'].iloc[:-1]), color='#ff7f0e', linewidth=1.5,
        linestyle='--', label=f'research approx (signal-date cum, {per["cum"].iloc[-2]*100-100:+.0f}% @12-24)')

# benchmark
net20c = pl.read_parquet(Path('evidence/v31_bear_factor_analysis_20260925/net20_cache.parquet'))
win = net20c.filter((pl.col('trade_date') >= pl.lit(pd.Timestamp('2022-01-01').to_pydatetime())) &
                    (pl.col('trade_date') <= pl.lit(pd.Timestamp('2024-12-31').to_pydatetime())))
bm = win.filter(pl.col('trade_date').cast(pl.Date).is_in([d.date() for d in rebal_dates])).group_by('trade_date').agg(
    pl.col('net20').mean().alias('ret')).sort('trade_date').to_pandas()
bm['dt'] = pd.to_datetime(bm['trade_date'])
bm_nav = (1 + bm['ret'].fillna(0)).cumprod()
ax.plot(bm['dt'], bm_nav.values, color='gray', linewidth=1.3, linestyle=':',
        label=f'equal-weight benchmark ({bm_nav.iloc[-1]*100-100:+.0f}%)')

ax.axhline(1.0, color='black', linewidth=0.6, alpha=0.4)
ax.set_title('Bear window 2022-2024 | V32 voting strategy: AKQuant real sim vs research approx vs benchmark', fontsize=12)
ax.set_ylabel('NAV (start=1.0)')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(V3 / 'nav_comparison.png', dpi=130)
print(f"\nSaved: {V3}/nav_comparison.png")

# --- 3. 汇总 MD ---
m = res_v3['metrics']
md = f"""# V32 投票策略 — AKQuant 真实回测最终报告

**日期**: 2026-09-25 | **窗口**: 2022-01-04 ~ 2024-12-31 (熊市 3 年, 用户合同窗口)
**策略**: 静态因子池 (H1 收益 top10 + 固定方向) + 投票选股
**执行**: T+1 raw open, 20 日调仓, lot=100, commission 0.25%/边 + slippage 0.1%/边

## 结果

| 指标 | AKQuant v3 (真实) | 研究近似 (net20) | 基准 (等权) |
|---|---:|---:|---:|
| 总收益 | **{m['total_return_pct']:+.1f}%** | +160.1% (full) / +177.9% (至12-24) | -13.4% |
| Sharpe | {m.get('sharpe_ratio', 0):.2f} | — | — |
| 最大回撤 | {m.get('max_drawdown_pct', 0):.1f}% | -15.4% | — |
| 胜率 | {m.get('win_rate', 0):.1f}% | 65% (期) | — |
| Profit Factor | {m.get('profit_factor', 0):.2f} | — | — |
| 平仓交易数 | {int(m.get('closed_trade_count', 0))} | 37 期 | — |
| 期末市值 | ¥{m['end_market_value']:,.0f} | — | — |
| 拒单 | {res_v3.get('n_rejected', '?')} | — | — |
| 未平仓位 | {res_v3.get('open_positions', '?')} 只 (最后篮, MTM ¥{res_v3.get('open_mtm', 0):,.0f}) | — | — |

## 分年

| 年 | AK v3 | 研究 |
|---|---:|---:|
| 2022 | -6.2% | +18.4% |
| 2023 | +55.1% | +37.7% |
| 2024 | +45.7% | +59.5% |

## 对账: 差异来源 (AK vs 研究 +160.1%)

1. **出场时点**: 研究 = close(s+21); AK = open(s+21) — AK 每期少持有 1 个交易日 (s+21 当日)
2. **成本**: 研究 0.5% 往返; AK 0.7% 往返 (+0.25%+0.1% 双边)
3. **现金缓冲**: AK 目标权重 0.99 (1% 缓冲, 消除拒单); 研究 100%
4. **整手限制**: lot=100 取整

**AK 口径重算研究**: {alt_cum*100:+.1f}% vs AK 实际 {m['total_return_pct']:+.1f}% (差 {alt_cum*100 - m['total_return_pct']:+.1f}pp)
→ 执行摩擦解释大部分差异; 剩余为取整/调仓容差/浮动权重

## 修复链 (v1 → v2 → v3)

- v1: 数据未截断 (跑到 2026) + 全额权重 → 31 拒单
- v2: 0.99 缓冲修复拒单 (0 拒单 ✅), 数据仍越界 (bug)
- v3: 数据严格截断 2021-12-01 → 2024-12-31 ✅ 本报告

## ⚠️ 限制

1. 因子池选于 H1 (2022-01~2023-06), H1 段结果含选择偏差; H2 段为半外验证
2. 策略为 2022-2024 熊市窗口专用, 未验证其他 regime
3. 收益集中 2024 (9-11 月小微盘行情)
4. volume 固定 1e9 (不模拟容量限制); 头部因子偏小市值/低流动性, 真实容量有限
5. 未含停牌/涨跌停约束的显式建模 (依赖面板数据完整性)
"""
with open(V3 / 'FINAL_REPORT.md', 'w') as f:
    f.write(md)
print(f"Saved: {V3}/FINAL_REPORT.md")
print("\n✅ 对比完成")
