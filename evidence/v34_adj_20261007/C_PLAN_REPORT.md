# C 方案（全量修复）阶段报告 — 2026-10-07（修正版 v2）

> 修正历史：v1 报告的"57 倍差异"结论基于一个**混淆对比**（v34 picks 误用 lookback=60，而 v33 全系是 lookback=20）。本 v2 修正全部数字并记录排查过程。

## 1. 修复清单（全部完成）

| # | 修复 | 状态 |
|---|---|---|
| 1 | adj_factor 填充（真实复权因子） | ✅ `data/wavehunter_hs300_v34_adj_20261007.parquet` (461列) |
| 2 | 404 个技术因子重建于复权价 | ✅ `evidence/v34_adj_20261007/new_factors_adj.parquet` |
| 3 | picks builder：`_fwd_net` 复权 + `is_finite` | ✅（默认 lookback 曾误为 60，已修复为 20） |
| 4a | panel_days bug 修复（完整交易日历） | ✅ `examples/v41_run_akquant_v34.py` |
| 4b | 公司行动注入（分红+送转） | ✅ 6,563 事件本地推导 + 引擎注入 |

## 2. 关键 bug 排查记录（重要教训）

排查链：stage-A 复现派生与 builder 输出不一致（Jaccard 0.272）→ 逐层排除：
diff 值一致（delta=0）→ 分数不一致 → 反推窗口无匹配 → dump builder 内部窗口
→ **发现 `window n=60`**。

**根因**：`build_picks_v34_factorrank.py` 由 nanfix 版拷贝而来，其 `LOOKBACK_SESSIONS = 60`
未同步改为锁定的 20；首次 v34 构建（12:36）未显式传 `--lookback`，导致
**v34 picks 实为 lookback=60 版本**。修复后重建（lb=20）与 stage-A 矩阵派生完全一致（195/195 相同）。

**结论：v1 报告中"v33 vs v34 picks 差 57 倍"= lookback 60 vs 20 的混淆，不是数据修复效应。**

## 3. 修正后的结果矩阵（2010-2025，195 rebalance，raw 执行）

| run | total% | ann | Sharpe | MDD% | PF | win% |
|---|---:|---:|---:|---:|---:|---:|
| v33-lb20, raw noCA | 304,403 | 65.1% | 1.95 | 24.9 | 3.39 | 63.4 |
| v33-lb20, raw + CA | 747,100 | 74.6% | 2.18 | 24.9 | 3.52 | 64.2 |
| v34-lb20, raw noCA | 85,301 | 52.5% | 1.63 | 30.2 | 2.67 | 62.3 |
| **v34-lb20, raw + CA（新基准）** | **303,868** | **65.1%** | **1.91** | **29.9** | **2.87** | **64.0** |
| [对照] v34-lb60, raw + CA（无关变体） | 13,048 | 35.7% | 1.28 | 39.9 | 2.17 | 60.1 |

**数据修正的真实影响**：74.6% → 65.1% ann（−9.5pp）；CA 效应 ×2.46（v33）/ ×3.56（v34）。

## 4. 归因（四角对比，全部 lb=20，含 CA）

| 组合 | ann | total% |
|---|---:|---:|
| v33 因子 + raw fwd（v33 基准） | 74.6% | 747,100 |
| v33 因子 + adj fwd（p3） | 75.6% | 813,504 |
| v34 因子 + raw fwd（p2） | 68.1% | 408,123 |
| v34 因子 + adj fwd（canonical） | 65.1% | 303,868 |

四角带宽 65.1%~75.6%（±5pp）；picks Jaccard：v34 vs v33 = 0.385；fwd_net 单独效应 ~0.73；lookback 效应 0.27。

## 5. 交叉验证

| 验证 | 结果 |
|---|---|
| raw+CA vs 纯复权价（hfq） | 16年差 7.6% ✅ |
| v41 runner vs capfix（同输入） | 精确复现（+304403%）✅ |
| 引擎机制（601633 真实除权） | 股数×s、现金+=qty×c、守恒 ✅ |
| fixed-rebuild vs stage-A 派生 | 195/195 相同 ✅ |

## 6. 公司行动推导（本地推导，放弃 Tushare）

- **Tushare dividend 接口限频 1次/小时** → 354 股需 15 天，不可行（实测）
- 本地推导：`pre_close`（除权基准价）+ `total_share`（送转比例）
- 推导结果：6,563 事件（split 915 / dividend 5,648），2010+ 全部 5,678 条 0 失败
- 抽查验证：000001 ✓ / 000630 (s=5.0, cash=0.18) ✓ / 002594 (s=3.0, cash=3.97) ✓ / 601633 (s=3.0, cash=0.25) ✓

## 7. 引擎机制（实测语义）

```
CorporateAction(symbol, date, type, value)
  Split:    qty_new = qty × value          (value = 倍数, 如 5.0)
  Dividend: cash += qty × value            (value = 每股现金)
  date = 除权日（ex-date），按序注入 dividend→split
注入: monkeypatch bte.Engine 为 proxy，在 add_data() 后逐个 add_corporate_action()
```

## 8. 遗留问题

1. **多 offset（seed）正式验证**（stage 5）
2. **v34-lb20 的 2026 OOS 重跑**
3. **差异型因子无量纲化**：MOM/ATR/MACD 等"元"为单位差分因子横截面比较无经济含义（raw 和 adj 都不对），正确做法 scale-free 归一化 —— 新预注册变量
4. **教训入 skill**：拷贝 builder 脚本必须逐参数核对默认值（lookback 事故）

## 9. 产出索引（关键）

```
data/wavehunter_hs300_v34_adj_20261007.parquet            v34 面板
evidence/sweep/v34_factorrank_lb20_fixed_20261007/        v34-lb20 picks（正确版）
evidence/sweep/v34_lb20_fixed_ca_20261007/V14_0/          v34-lb20 + CA（新基准）
evidence/sweep/v34_lb20_fixed_noca_20261007/V14_0/        v34-lb20 noCA
evidence/stage_a_20261007/                                Stage-A 归因+抗扰动
examples/build_picks_v34_factorrank.py                    （LOOKBACK 已修=20）
examples/v41_run_akquant_v34.py                           新 runner
examples/derive_corporate_actions_local.py                公司行动推导
```
