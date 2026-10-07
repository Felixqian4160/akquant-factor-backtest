# V37 V14 + V19 hybrid (V27 bear router) — 真实 AKQuant 执行

## 实验合同

- V33 panel, 410 选股因子
- 2010-01-04 ~ 2025-12-31, 195 调仓日 (每 20 个交易日)
- **V14 IC voting** (bull_neutral regime):
  - 因果 IC: 60 日 Spearman rolling, lag = 21
  - top-10 |IC| 因子
  - 每因子提名 top-10 股票, 票数 >= 2, 股票 5-10 只
- **V19 factor-return voting** (bear regime):
  - 每个因子历史 mean T+1→T+21 net (含 0.5% 成本)
  - 保留 mean > 0.005 因子, top-10, top-10 股票, 票数 >= 3, 5-10 只
- **Bear router (V27)**:
  - 16 信号投票: close vs MAs / 4 ret signs / 3 distance / slope / drawdown
  - 当日票数 >= 2 → V2 flag
  - 10 日滚动累计 >= 4 → bear regime
- T+1 raw open, lot=100, commission=0.25%, slippage=0.10%
- 0.9 target cap

## 结果

| 指标 | 数值 |
|---|---:|
| 总收益 | **+502.42%** |
| Sharpe | **0.524** |
| MDD | -62.35% |
| 胜率 | 53.18% |
| PF | **1.434** |
| 拒单 | 87/2716 = 3.20% |
| 调仓日 | 195/195 (20 日周期维持) |
| trades | 1,226 |

### Regime 分布
- bull_neutral: 26/195 (13%)
- bear: 169/195 (87%)

V27 router 把大多数调仓日判为 bear（MA60 触发 + 多信号累计易过门槛）。

### 分年表现

```
2010: +26.9%   2017: +16.3%
2011: -37.4%   2018: -46.5%
2012: -14.5%   2019: +90.5%
2013: +16.6%   2020: +47.0%
2014: +21.0%   2021: +20.5%
2015: +128.4%  2022: -25.5%
2016: -23.5%   2023: +32.1%
              2024: +23.9%
              2025: +36.6%
```

16 年里 **11 正 5 负**（V35 是 9 正 7 负，胜率改善）。

## 对比所有 sweep

| 版本 | 收益 | Sharpe | MDD | 胜率 | PF |
|---|---:|---:|---:|---:|---:|
| **V37 V14+V19+V27** | **+502.42%** | **0.524** | -62.35% | 53.18% | **1.434** |
| V35 K10 IC=60 (no regime) | +136.38% | 0.345 | -61.40% | 51.30% | 1.202 |
| V36 K10 IC=20 (no regime) | -37.22% | 0.012 | -80.96% | 46.86% | 1.032 |
| V35 K30 (IC=60) | +32.30% | 0.198 | -58.06% | 51.98% | 1.137 |
| V34 A (all 408) | +10.18% | 0.16 | -68.06% | n/a | n/a |

差距 ≈ ¥370M（V37 vs V35），远超 AKQuant 噪声阈值 ±¥50M，是真实信号差异。

## 9 项账本审计

| 检查 | V37 |
|---|---|
| trades_exist | ✅ |
| orders_exist | ✅ |
| orders_both_sides | ✅ |
| **orders_filled_all** | ❌ (87/2716 = 3.20%) |
| qty_lot_multiple | ✅ |
| nav_bounded | ✅ |
| rebalance_dates_in_window | ✅ (195) |
| trade_pnl_identity | ✅ |
| mdd_recomputed | ✅ (-62.35%) |

## 关键解释

1. **为什么 V37 比 V35 好这么多？**
   - 169/195 调仓日走 V19 factor-return voting（不是 V14 IC voting）
   - V19 bear voting 用历史 net return 排名选因子，本身就用过去真实收益验证
   - 牛市熊市分别用不同 voting 策略，regime 自适应
   - V35 单一 V14 IC voting 在 2022/2024/2025 熊市段被拖累

2. **为什么 V27 router 这么宽松？**
   - 16 个信号（close vs MA5/10/20/60/120, MA20<MA60, MA60<MA120, ret10/20/40/60 sign, dist_ma20, dist_ma60, slope_ma60, dd_60d, dd_120d）
   - 票数 >= 2 即记 V2 flag，10 日累计 >= 4 → bear
   - 在 trend 阶段只要 2 个信号 hit 就可能触发（MA5/10 频繁低于 close）
   - 169/195 bear 比例说明 router 阈值偏低

3. **是否 production alpha？**
   - 单 seed + 单次执行，无多 seed 方差验证
   - Sharpe 0.524 仍偏低（远低于 1.0 阈值）
   - 2018 -46.5% 是 16 年最大单年回撤
   - MDD -62.35% 仍是大数

## 证据路径

```text
examples/v37_v14_picks.py         # V14 IC + V19 factor-return + V27 router picks
examples/v37_run_akquant.py       # 真实 AKQuant 执行
evidence/v37_v14_causal/V14/{picks, picks_meta, factor_manifest}.json
evidence/v37_v14_akquant/V14/{result, trades, orders, nav, ledger_audit}
```

## 后续选项

| 选项 | 内容 |
|---|---|
| A | 在 V37 上跑 5 seed 方差验证 |
| B | 在 V37 上调 V27 router 阈值（10 日累计 >= 6 → bull 期更长） |
| C | 在 V37 上做分年 regime 重新训练（用 2010-2018 训练，2019-2025 OOS） |
| D | 接受 V37 当前候选，停掉 alpha search 转向新数据 (drip/effort/limit-up) |