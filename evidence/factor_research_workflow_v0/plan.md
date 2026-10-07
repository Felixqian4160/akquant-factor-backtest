
## v0 因子研究工作流（5 阶段 pipeline）

**Goal**: 用 AKQuant 找 (1) 最佳因子研究工作流 (2) A 股最稳定因子群 (3) 最稳定策略

| Stage | 内容 | 状态 | 单一变量 |
|---|---|---|---|
| 1_baseline | 单因子 trend-following sweep（v3）| ✅ 已完成（1179 backtest）| — |
| **2_regime_stability** | **跨 regime 算 stability metric** | ⏭️ 下一步候选 | panel slicing（full→per-regime）|
| 3_multi_factor | top-N 因子 composite ranking | 待 Stage 2 | N 因子数 |
| 4_multi_symbol | multi-symbol top-K + rebal sweep | 待 Stage 3 | top_k, rebal |
| 5_oos_validation | time-split + cost sensitivity | 待 Stage 4 | 时间窗口 + cost |

**Stability metric**:
```
stability = mean(regime_closed_ret) - stdev(regime_closed_ret) / MDD
cutoff: mean > 1% AND stdev < 5% AND MDD < 30%
```

**已有数据**:
- v3 sweep: 393 因子 × 3 rebal × HS300 trend-following = 1179 backtest（ready for Stage 2）
- bull_legs.json: 10 legs, 2503 days total（regime 标签）
- bear_legs.json: 10 legs, 2299 days total（regime 标签）
- bull_composite pipeline: multi-symbol top-K + 6 因子 composite（Stage 3/4 复用）

**contract**: `evidence/factor_research_workflow_v0/workflow_preregistration.json`

### 4 个起点选项（按"single-variable + 不批量跑"原则）

| 选项 | 操作 | 成本 | 回答什么 |
|---|---|---|---|
| **A. Stage 2** | 切 v3 数据分 regime 算 stability metric | **0 个新 backtest**（纯切分）| 哪些因子跨 regime 稳定 |
| B. Stage 3 | 写新脚本跑 top-N composite in 10 bull legs | ~30 个 backtest | 多少因子组合最优 |
| C. Stage 4 | 跑 top_k × rebal 4×5 sweep | ~200 个 backtest | top-K + rebal 联合最优 |
| D. 只固 contract | 不跑实证，只看 plan | 0 | 工作流文档 |
