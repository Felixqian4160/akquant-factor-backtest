# v1 因子研究工作流 — FINAL 报告

**目标**: 年化 ≥ 12% / Sharpe ≥ 1.2 / MDD ≤ 12% (可接近)
**panel**: v2_talib (420 cols, 354 stocks, 5498 days)
**regime**: bull_legs.json (10 legs ≥100d, 2513 days)

## 🎯 FINAL Stage 对比

| Stage | closed% | ann% | sharpe | MDD% | trades | 3/3 完全达标 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 3 (baseline, top_k=20) | +35.94 | +84.6 | 3.090 | 17.95 | 138 | **2/10** |
| Stage 4 (kill rebal, top_k=20) | +34.02 | +73.9 | 3.140 | 17.33 | 63 | **2/10** |
| **Stage 5b (top_k=10, kill rebal) ⭐** | **+48.32** | **+134.1** | 2.927 | 17.67 | 35 | **2/10** |
| Stage 5c (top_k=10, on_bar kill) | +38.56 | +111.0 | 2.791 | 17.73 | 32 | 2/10 |
| ❌ Stage 6a (router, full panel) | +57.96 | +2.04 | 0.262 | 63.09 | 55 | 0/1 |

## 🏆 v1 workflow 终态 = **Stage 5b**

- **Single-variable 改动链**: panel slicing → kill-switch → top_k 20→10
- **每阶段都做了 bug fix + 真实工具验证**
- **Stage 5b 在 2/10 legs 完全达标**（leg#1 MDD 9.3%, leg#19 MDD 9.6%）
- **9/10 legs 通过 ann+sharpe target**

### ❌ 未达标原因 (诚实分析)

1. **MDD > 12%**: bull_composite 单策略 HS300 上 10 stocks 单 leg concentration (~2.8% universe)
2. **legs #13, #15, #25 仍有 21-24% MDD**: bull leg 内有大回撤段, kill-switch 不能根本解决
3. **legs #7, #21, #35 MDD 13-14%**: 略超 12%, 接近但未达

### ❌ Stage 5c / 6a 失败根因

- **Stage 5c (on_bar kill)**: kill 检查太频繁反而拖累 alpha (closed -20%)
- **Stage 6a (router full panel)**: bull_composite 的"alpha"只在已知 bull 段有效，router 在 full panel 暴露期太长（5263 days = 22.6 年），被多次大熊市打穿 → ann 退化到 +2.04%

## 📊 工作流设计 Insight

**Per-leg vs Full panel 关键差异**：
- **Per-leg backtest** = 已知 bull 时段 → bull_composite 真实 alpha (+134% ann)
- **Full panel + ground-truth router** = bull_composite 暴露在 bull leg 内的 bear 段 → alpha 退化到 +2% ann
- **Implication**: bull_composite 是 **bull-leg-internal specialist**，不是 standalone production strategy

## 🎯 真正的 production-ready v2 workflow 需要

1. **Multi-universe diversification** (HS300 + CSI500 + CSI1000)  → 真实分散
2. **Regime detection in real-time** (不是事后 ground truth) → LightGBM classifier
3. **Position sizing risk parity** → 每 leg 风险平价
4. **Multi-strategy composite** (MainWave + Reversal + MeanReversion) → cross-strategy alpha

## 📁 文件落盘清单

- `examples/v1_stage3_per_bull_leg.py`
- `examples/v1_stage4_kill_switch.py`
- `examples/v1_stage5b_topk10.py`
- `examples/v1_stage5c_onbar_kill.py`
- `examples/v1_stage6a_regime_router_full.py`
- `evidence/factor_research_workflow_v1/workflow_v1_preregistration.json`
- `evidence/factor_research_workflow_v1/stage_1_to_4_progress.json`
- `evidence/factor_research_workflow_v1/stage_1_to_4_REPORT.md`
- `evidence/factor_research_workflow_v1_stage3_*/per_leg_bull.json`
- `evidence/factor_research_workflow_v1_stage4_*/per_leg_bull_killswitch.json`
- `evidence/factor_research_workflow_v1_stage5b_*/per_leg_bull_topk10.json`
- `evidence/factor_research_workflow_v1_stage5c_*/per_leg_bull_topk10_onbarkill.json`
- `evidence/factor_research_workflow_v1_stage6a_*/regime_router_full.json`

## 🎯 Goal Status

**用户原目标**:"经过一个工作流就能找到最合适的因子群+策略, 目标年化 12% 夏普 1.2 年回撤 12%以下"

**v1 workflow 实证结论**:
- ✅ 工作流本身达到目标（5 个 stage, single-variable 纪律, 每阶段真实 backtest 验证）
- ✅ 因子群找到 (bull_composite 6 因子: talib_NATR/TRANGE + gtja_159/149/144 + mw_vol_20d)
- ✅ 策略找到 (multi-symbol top_k=10 rebal=20 + kill-switch DD>8%)
- ⚠️ **production target (ann 12% / Sharpe 1.2 / MDD 12%) 只在 per-leg bull days 成立**
- ❌ full-panel production-realistic 测试中**全部失败** (Stage 6a ann +2%, MDD -63%)

**结论**: v1 workflow **找到了 alpha 但没找到 production-ready 策略**。
- bull_composite 是 bull-leg specialist, 不是 standalone 策略
- 真正的 production alpha 需要 **multi-universe + cross-strategy composite**