# v1 因子研究工作流 — Stage 1-4 完整进度报告

**目标**: 年化 ≥ 12% / Sharpe ≥ 1.2 / MDD ≤ 12%

**完成 Stage**: 1 (data) → 2a (regime stability) → 3 (per-bull-leg) → 4 (kill-switch)

## 🎯 关键结论

| Target | Met? | 实际 (Stage 4) |
|---|---|---|
| 年化 ≥ 12% | ✅ EXCEEDED | mean **+78%** |
| Sharpe ≥ 1.2 | ✅ EXCEEDED | mean **3.14** |
| MDD ≤ 12% | ❌ NOT MET | mean **17.33%**, only 2/10 legs |

## 📊 Stage 3 vs Stage 4 对比

| Leg | S3 closed | S4 closed | S3 MDD | S4 MDD | S3 tr | S4 tr |
|---|---:|---:|---:|---:|---:|---:|
| #1 (126d) | +24.68% | +24.68% | 7.3% | 7.3% | 76 | 76 |
| #7 (123d) | +18.42% | +17.22% | 13.8% | 14.2% | 64 | 41 |
| #13 (445d) | +109.49% | +84.87% | 23.3% | 20.2% | 279 | 147 |
| #15 (118d) | +18.45% | +15.76% | 22.3% | 22.3% | 57 | 23 |
| **#17 (727d)** | **-7.79%** | **+10.88%** | **38.0%** | **35.1%** | **399** | **22** |
| #19 (106d) | +44.77% | +37.40% | 9.4% | 9.4% | 70 | 55 |
| #21 (221d) | +34.42% | +38.17% | 13.2% | 13.2% | 115 | 109 |
| #25 (324d) | +52.98% | +59.29% | 23.5% | 23.4% | 188 | 63 |
| #31 (108d) | +13.91% | +8.89% | 16.0% | 14.5% | 44 | 17 |
| #35 (205d) | +50.10% | +43.08% | 12.7% | 13.6% | 87 | 78 |

**Aggregate**:
- Mean closed_only: +35.94% → **+34.02%** (-1.92%, kill-switch 小幅损失)
- Mean MDD: 17.95% → **17.33%** (-0.62%, 改善有限)
- Mean trades: 138 → **63** (-54%, 不靠量)
- Positive legs: 9/10 → **10/10**
- **leg#17 救回**: -7.79% → +10.88% (22 trades 被 kill)

## 🔑 Kill-switch 实证发现

- **救回 leg#17** (727d 大跨 leg): -7.79% → +10.88%
- **没根本解决 MDD > 12%** in legs #7/13/15/25/31/35
- **根因**: DD 在 rebalance cycle (20d) 内发生, kill 触发时已超阈值
- **Bug 发现并修复**: `acct['total_equity']` → `acct['equity']`

## 📁 文件落盘

- `evidence/factor_research_workflow_v1_stage3_20260930_025600/per_leg_bull.json`
- `evidence/factor_research_workflow_v1_stage4_20260930_025959/per_leg_bull_killswitch.json`
- `evidence/factor_research_workflow_v1/stage_1_to_4_progress.json`
- `examples/v1_stage3_per_bull_leg.py`
- `examples/v1_stage4_kill_switch.py`

## 🎯 下一步 4 选 1

### A. Stage 5a: Regime Router
- 加 LightGBM classifier → 只在预测 bull 时跑 bull_composite
- Expected: ann↓ (bull days < 100%), MDD↓ (少在 bear/sideways 暴露)
- Effort: medium (3-5 backtests)

### B. Stage 5b: top_k Reduction
- top_k 20 → 10 (更分散)
- Expected: ann略↓, MDD↓
- Effort: **low** (1 脚本, 10 backtests, ~2 min)

### C. Stage 5c: OOS Validation
- Time split 2010-2018 / 2019-2021 / 2022-2025
- Expected: 验证 generalization
- Effort: medium (25 backtests)

### D. 完结报告
- Stage 1-4 充分 (alpha 验证 + kill 验证)
- MDD 问题已记录 (待 future improvement)
- Effort: minimal (write final report)

## 💡 Stage 4 bug + fix

```python
# 错 (我最初的代码):
equity = float(acct.get("total_equity", INITIAL_CASH))
# → 永远 fall back to INITIAL_CASH, kill 永远不触发

# 对 (fix):
equity = float(acct.get("equity", INITIAL_CASH))
# → 真实 portfolio equity, kill 正常工作
```