# v3 共振选股框架（多窗口共振买入 + 条件失效卖出轮动）

用户概念（2026-10-10）：利用股票过去 10/20/40/60/120 天中一个或几个共振的因子，
预测未来 5/10/20 天的涨跌；买入后持有对应时间；当买入条件不存在/出现下行风险时
卖出，换入其他满足条件的股票。

## 1. 信号定义（全部因果，无未来信息）

- 因子池：去重后 413 个因子（`data/wavehunter_hs300_v34_dedup_20261010.parquet`）
- 方向修正：`dir(f,t) = sign(mean(已结算 diff 行 [t-270, t-20)))`，
  已结算 = 行号 ≤ t-21（settle-lag）；不足 30 行默认 +1
- 横截面百分位：`r(f,t,s) = rank_pct(factor值 × dir 修正)`（同日 354 只股票内）
- 多窗口持续性：`sw(f,t,s,W) = 过去 W 日 r 的均值`，W ∈ {10, 20, 40, 60, 120}
- 单因子共振判定：
  - 看多：`#{W: sw ≥ 0.65} ≥ 2`（≥2 个窗口同时处于高位）
  - 看空：`#{W: sw ≤ 0.35} ≥ 2`
- 投票：`BULL(s) = Σ_f 看多`；`BEAR(s) = Σ_f 看空`（413 因子）

## 2. 交易规则（检查网格 = 每 5 交易日）

- 买入条件：`BULL ≥ ENT_MIN` 且 `BEAR ≤ BEAR_MAX`
- 持有条件（滞回）：`BULL ≥ HOLD_MIN` 且 `BEAR ≤ BEAR_MAX`
- 卖出：持有条件失效 或 持有满 4 个网格（≈20 bars）→ 下一网格剔除
- 冷却：卖出后 ≥ 1 个网格才可回补（最优配置 3 个网格）
- 持仓：候选按 `(-BULL, BEAR, 代码)` 排序取前 10，等权；不足不补

阈值（在 2010-2017 网格上校准，2018+ 为真 OOS）：

| 参数 | 值 | 含义 |
|---|---|---|
| ENT_MIN | 129 (P95) | 买入需 ≥129 个因子共振看多 |
| HOLD_MIN | 69 (P50) | 持有需 ≥69（跌破中位数 = 条件不存在） |
| BEAR_MAX | 116 (P90) | ≥116 个因子共振看空 = 风险信号 |

## 3. 执行合同（v41 runner，与全部对照臂一致）

- T+1 NextOpen；90% 等权（0.90/n 每只）；佣金 0.25% + 滑点 0.10%/边；CA 注入
- 21 交易日硬持有上限（runner 强制，超出部分由网格剔除对齐）
- 2010-2025 全期；网格 = 全历 dates[0::5]

## 4. 配置与结果（单变量推进，全部同成本合同）

| 配置 | hold | cool | grid | 全期 ann | 全期 MDD | 段 ann | 段 MDD | 段 SR | trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V14_0 (v1) | P90 | 1 | 5 | −0.27% | 68.7% | 3.35% | −43.0% | 0.259 | 7312 |
| V3_hyst | P50 | 1 | 5 | 4.68% | 65.0% | 12.02% | −41.1% | 0.596 | 7501 |
| **V3_cd3** | P50 | **3** | 5 | **5.12%** | 64.7% | **14.73%** | **−32.1%** | **0.692** | 7387 |
| V3_g10 | P50 | 1 | 10 | 5.13% | 61.5% | 11.09% | −39.3% | 0.541 | 3637 |

段 = 2018-2025（真 OOS，阈值/方向均只用结算数据）。对照：HS300 指数段 1.75%/−45.6%；
等权全池 16.69%/−26.1%；static_2017 21.23%/−44.9%；k120 9.18%/−32.9%。

## 5. 复现命令

```bash
ENV="PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages"
# 信号（21 批，断点续传，~80s 全量）
env $ENV /usr/bin/python3.12 -u examples/v3_resonance_framework.py --signals 21
# 构建 picks（可配 hold-pctl / cooldown / grid-step / tag）
env $ENV /usr/bin/python3.12 -u examples/v3_resonance_framework.py --build --hold-pctl 0.50 --cooldown 3 --tag V3_cd3
# 回测 + 报告
env $ENV /usr/bin/python3.12 -u examples/v3_resonance_framework.py --run 1 --tag V3_cd3
env $ENV /usr/bin/python3.12 -u examples/v3_resonance_framework.py --report --tag V3_cd3
```

## 6. 脚本与产物

- 脚本：`examples/v3_resonance_framework.py`（--signals / --status / --build / --run / --report，
  可调参数：--hold-pctl / --cooldown / --grid-step / --tag）
- 信号缓存：`evidence/v3_resonance_20261010/_cache/votes_batches.npz`
  （bull/bear 投票矩阵 5502×354，可复用于任意参数组合）
- picks：`evidence/v3_resonance_20261010/picks/<tag>/`
- sims：`evidence/v3_resonance_20261010/sims/<tag>/`（result/nav/trades/orders/ledger）
- 成本结构说明：本框架 5 日网格 + 21-bar 上限 ⇒ 结构性换手较高；
  现成本合同（0.35%/边）≈ 实盘 A 股成本的 3-7 倍，敏感度需单独报告。

## 7. 已知机制特征

- 每网格等权再平衡（drift 微调）属于 runner 合同；除权重整外换手 = 成员轮换
- 21-bar 上限强制轮换是最大成本来源（占比 ~45%+ 组合年换手）
- 卖出条件从 P90 放松到 P50（"跌破共识中位才卖"）= 最大单项改进
- 冷却 3 网格消除"同名单反复回补"（601919 类 159 次往返模式）
