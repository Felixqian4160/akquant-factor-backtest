# v4 五因子共振投票（2026-10-11）

> 用户合同："投票这个策略这个方向肯定是对的……假设 5 个因子使用的方法，然后进行回测，
> 找到一条斜向上近似直线的收益曲线（波动低、回撤低、稳定累积向上）"。
> 本策略 = v3 共振投票机制 + **精选 5 因子子集**（factset）。单变量流程：集合选择 → 全集合扫描 → 参数筛选 → 多相位复核。

## 1. 信号定义（与 v3 相同，factset 版）

- 方向：`dir(f,t) = sign(mean(结算 diff[f, t-270:t-20]))`（250 日滚动，已结算口径）
- 横截面百分位：`r(f,t,s) = rank_pct(值 × dir(f,t), 同日全池)` ∈ [0,1]
- 多窗口持续性：`sw(f,t,s,W) = 过去 W 日 r 均值`，W ∈ {10, 20, 40, 60, 120}
- 单因子共振：看多 = ≥2 窗口 `sw ≥ 0.65`；看空 = ≥2 窗口 `sw ≤ 0.35`
- 投票：`BULL(s) = 看多因子数`，`BEAR(s) = 看空因子数`（5 因子 → 0..5）
- 连续得分：`score(s) = Σ_j mean_W(sw_j)`（0..5，用于候选排序）

## 2. 因子集合（2010-2017 因果口径选择，413 因子池）

| tag | 因子 | 选出理由 |
|---|---|---|
| **s1** | l_ami, alpha_alpha040, gtja_gtja_113, hl_52w_disposition, alpha_alpha006 | 2010-17 t 最高 5（mean_t=12.2）|
| **s1d** | s1 中 alpha006 → gtja_gtja_015 | t 排序 + 贪心 \|corr\|<0.5（组内 max_corr 0.47）|
| s2 | alpha 040/045/003/055/013 | 8/8 年全正 + 低相关 0.35（**实测最差，见 §5**）|
| s3 | l_ami, alpha040, gtja113, talib_ADX, hl_52w | alpha/gtja/talib/其他 各组 t 最佳 |
| s4 | 随机 5（seed=20261011）| null 对照 |
| n3 / n8 | l_ami, alpha040, gtja113 / + hl_52w, alpha006, gtja015, l_turna, gtja144 | 个数验证（t top-3 / top-8）|

## 3. 决策与执行合同

- 阈值（每个 factset 用 2010-2017 网格行自动校准）：`ENT=P95`、`HOLD=P65`、`BEAR=P90`
  （5 因子 → ENT_MIN=3、HOLD_MIN=1、BEAR_MAX=3）
- 排名：`rank_by=score`；冷却 3 网格（CD6 变体 = 6）；`max_hold_bars=60`；`n_pos=15`；网格步长 5 交易日
- 执行（v42 runner）：T+1 NextOpen / 等权 90% 仓 / 佣金 0.25% + 滑点 0.10% 每边 / CA / 60 bar 上限
- 评估段：2018-01-01 ~ 2025-12-31（选择+校准只用 2010-2017，**真 OOS**）；五相位 = 网格 offset 0-4

## 4. 核心结果（5 相位均值，2018-2025 段）

| 配置 | 收益/回撤比 mean [min,max] | 段年化 | 最差MDD | mean R² | 60d正收益 |
|---|---|---|---|---|---|
| **s1d（5因子）** | **0.661** [0.519, 0.751] | 19.67% | -33.8% | 0.896 | 67% |
| s1_CD6（5因子+冷却6）| 0.641 [0.577, 0.694] | 19.06% | -30.6% | 0.887 | 68% |
| s1d_CD6 | 0.635 [0.511, 0.776] | 19.30% | -33.3% | 0.915 | 68% |
| s1（5因子）| 0.620 [0.567, 0.684] | 18.69% | -31.3% | 0.869 | 68% |
| s3（5因子）| 0.594 [0.538, 0.695] | 17.87% | -31.2% | 0.843 | 66% |
| **n3（3因子）** | 0.508 [0.419, 0.578] | 12.50% | **-26.2%** | **0.933** | 65% |
| — SW047（413因子旧最优）| 0.422 [0.336, 0.473] | 12.56% | -36.9% | ~0.83 | 57% |

单相位补充（o0）：s4 对照 0.216、s2 0.204、n8 0.371（MDD -44.4%，R² 0.664——8 因子崩溃）。

## 5. 裁决（多相位口径）

- **5 因子是甜点**：3 因子（0.508）斜率太低；8 因子（0.371）全面退化 → 因子数不是越多越好。
- **s1/s1d/s3 全部 5/5 相位优于 413 因子旧最优**；s1 的每个相位（min 0.567）> SW047 的每个相位（max 0.473）。
- **形态最直候选 = n3（3因子）**：R² 0.933（全场最高）、MDD -26.2%（全场最浅），代价是年化 12.5%（比 5 因子低 ~7pp）。
- CD6（冷却 6）边际：对 s1 4/5 相位改善、对 s1d 均值反降 → **不列为独立改进**（数据不一致）。
- **意外负面结论**：s2（8/8 年全正 + 最低相关）实测最差 → 单因子长期一致性 ≠ 组合投票有效，必须以组合形态实测。
- 局限：三大回撤坑未消除（2023-07→2024-02 -29.4%、2018-05→2018-10 -28.9%、2021-12→2022-04 -23.3%），均为市场系统性下跌段；仓位/暴露控制（用户已搁置）是剩余主要杠杆。

## 6. 复现命令

```bash
cd /media/felix/f/quant/akquant-factor-backtest   # 全部用系统 python
/usr/bin/python3.12 -u examples/v4_select_factors.py     # 选择因子集合 (s1/s1d/s2/s3/s4)
/usr/bin/python3.12 -u examples/v4_select_extra.py       # n3/n8 集合
/usr/bin/python3.12 -u examples/v4_five_factor.py        # 全部集合 o0 (signals→build→sim→journal)
/usr/bin/python3.12 -u examples/v4_five_factor.py --phases s1,s1d,s3         # 多相位
/usr/bin/python3.12 -u examples/v4_five_factor.py --screen s1_top5_t         # 7 个单变量变体
/usr/bin/python3.12 -u examples/v4_five_factor.py --vphases V4_S1_CD6,V4_S1D_CD6,V4_S1_N3
/usr/bin/python3.12 -u examples/v4_five_factor.py --report   # 汇总 (v4_report.md)
/usr/bin/python3.12 -u examples/v4_five_factor.py --curves   # 净值曲线 (phase-mean)

# 手动单集合 (framework factset CLI)
/usr/bin/python3.12 -u examples/v3_resonance_framework.py \
  --factset evidence/v4_five_factor_20261011/run_factsets/s1_top5_t.json --signals 99
/usr/bin/python3.12 -u examples/v3_resonance_framework.py \
  --factset evidence/v4_five_factor_20261011/run_factsets/s1_top5_t.json \
  --build --tag V4_S1_TOP5_T_o0 --offset 0 \
  --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score
/usr/bin/python3.12 -u examples/v3_resonance_framework.py \
  --factset evidence/v4_five_factor_20261011/run_factsets/s1_top5_t.json \
  --run 1 --tag V4_S1_TOP5_T_o0 --runner v42 --holding-bars 60
```

## 7. 脚本与产物

- 脚本：`examples/v4_select_factors.py` / `v4_select_extra.py` / `v4_five_factor.py`；
  框架（含 `--factset`、`--ent-pctl`）：`examples/v3_resonance_framework.py`
- 集合定义：`evidence/v4_five_factor_20261011/{factorsets.json, selection.md, run_factsets/*.json}`
- 进度台账：`evidence/v4_five_factor_20261011/journal.jsonl`（逐条含参数/指标/直线度）
- 汇总：`evidence/v4_five_factor_20261011/v4_report.md`；曲线：`charts/v4_five_factor_o0.png`
- picks：`evidence/v3_resonance_20261010/picks/V4_*`；sims：`evidence/v3_resonance_20261010/sims/V4_*`
- 指标口径：段收益/年化/回撤/Sharpe + 直线度 R²（log NAV 线性拟合）+ 60/120 日滚动窗口正收益率
