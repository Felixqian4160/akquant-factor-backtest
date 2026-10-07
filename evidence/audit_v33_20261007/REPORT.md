# V33 因子库 + Factor-Return Voting 策略 — 全面审计报告

> **审计执行**: 2026-10-07 | **审计对象**: v33 panel + V33 Factor-Return Voting (lb=20)
> **审计范围**: 数据完整性、计算公式正确性、执行一致性、bug 影响量化
> **结论**: **发现 2 个严重 bug**，修复后策略表现**大幅变化**（+37%/年 → +61%/年）

---

## 0. Executive Summary

| # | 发现 | 严重度 | 影响 | 状态 |
|---|---|---|---:|---|
| 1 | **持仓 cap 机制失效** (`panel_days` bug) | 🔴 CRITICAL | 53.5% 交易变成 1-bar 强平（设计为 20-21 bars）；亏 ¥735M | 已修复 + 验证 |
| 2 | **NaN 因子污染评分** (`is_not_null` 不滤 NaN) | 🟡 MEDIUM | 28/195 日期 top-10 因子列表错误 | 已修复 + 验证 |
| 3 | 未复权价格（raw prices） | 🟡 MEDIUM | 737 个除权跳空污染 fwd 收益（1.19% 窗口） | 已记录，未修复 |
| 4 | 33 个 NaN 因子 / 2 个全 NaN 因子 | 🟡 MEDIUM | 部分因子有效值稀疏 | 已记录 |
| 5 | 17 笔超 22-bar 交易（停牌卡住） | 🟢 LOW | 停牌股无法卖出，真实行为 | 已解释 |

**修复后的正确基准**（lb=20，双重修复：capfix + nanfix）：

```
Total return:  +209,875.9%
Annualized:    +61.30%
Sharpe:        1.879
Max Drawdown:  -24.7%
Profit Factor: 3.191
Win rate:      63.2%
Trades:        1583
```

---

## 1. 审计模块与结果

### Module 1 — 面板数据完整性 ✅ PASS

```
Panel:   data/wavehunter_hs300_v33_with_new_factors_20261003.parquet
Rows:    1,298,335 | Stocks: 354 | Dates: 5,502 | Cols: 458
Period:  2004-01-02 ~ 2026-08-27
```

| 检查 | 结果 |
|---|---|
| 重复 (trade_date, ts_code) | 0 ✅ |
| OHLC 合理性 (high≥max(o,c), low≤min(o,c), high≥low) | 0 违规 ✅ |
| 负价格 / 零价格 / null 价格 | 0 / 0 / 0 ✅ |
| 日期连续性（无 >20 天异常 gap） | 0 gap ✅ |
| 每股每日股票数 | min=88, max=352（2010+ min=113） ✅ |

### Module 2 — _fwd_net 计算正确性 ✅ PASS

```
_fwd_net = close[t+21] / open[t+1] - 1 - 0.005
```

| 检查 | 结果 |
|---|---|
| polars vs pandas 独立复现 | max_diff = 0.00e+00, 0 mismatch ✅ |
| T+1 entry / T+21 exit 时点 | 精确匹配（手算验证） ✅ |
| 期末 21 行 null | 7,434 = 354×21 精确 ✅ |
| 价格合理性（茅台/平安银行抽样） | 与真实价格一致 ✅ |

⚠️ **注意**: 面板为 **未复权价**（adj_factor≡1.0, adj_close≡close）。737 个单日波动超 ±21%（A股涨跌停外）为除权跳空 → 见 Module 3。

### Module 3 — 除权污染量化 🟡 MEDIUM

| 指标 | 数值 |
|---|---|
| 极端波动日 (|ret1|>21%) | 737（722 下跌 + 15 上涨） |
| 分布 | 每年 11-59 个，均匀 |
| 受污染 _fwd_net 窗口 | 15,445 / 1,298,335 = **1.19%** |
| 受污染策略交易 | 12 / 1587 = **0.76%** |
| 影响方向 | 除权下跌为人为跳空 → 低估算净收益 |

**修复方向**: 使用复权价（需重建 panel 或 join 复权因子）。当前影响有限（0.76% 交易），记录待优化。

### Module 4 — 因子值计算抽查 ✅ PASS

| 因子 | 对照 | 结果 |
|---|---|---|
| talib_TYPPRICE / MEDPRICE / WCLPRICE / AVGPRICE / BOP | TA-Lib 公式 | diff=0.00e+00 **精确** ✅ |
| talib_RSI(14) | talib.RSI | diff=2.1e-14 **精确** ✅ |
| talib_MOM(10) | talib.MOM | diff=0.00e+00 **精确** ✅ |
| talib_ATR(14) | Wilder 经典平滑 | diff=0.000000 **精确**（talib.ATR 仅 seed 差异，100 bar 后收敛 0） ✅ |
| talib_NATR | = ATR/close×100 | 内部一致 diff=0.000000 ✅ |
| gtja_gtja_002 | -1×DELTA(inner,1) 公式 | diff=0.00e+00 **精确** ✅ |
| l_size | log(circ_cap) | corr=1.000000, mean_diff=0.0000 **精确** ✅ |
| l_ami (top因子) | 数值范围 | [5.6e-10, 6.0e-4], mean 3.7e-7 正常 ✅ |

### Module 5 — 评分复现 + 因果性 ✅ PASS

| 检查 | 结果 |
|---|---|
| diff(f,t) 复现（5 个样本因子） | 4/5 精确；gtja_gtja_002 差异 → 根因 = Module 6 的 NaN bug |
| score(f,T) 滚动窗口因果性（50 个日期抽样） | 全部 t < T ✅ 0 违规 |
| picks_meta regime vs router_map | 0 mismatch / 195 ✅ |

### Module 6 — NaN 处理 bug 🟡 已修复

**根因**:
```python
# 错误（builder 原实现）:
df = df.filter(pl.col(f).is_not_null() & ...)   # is_not_null 不过滤 NaN 浮点数!
# polars 降序 rank 把 NaN 排第 1 → NaN 股票占据 top-10 名额
```

**NaN 因子清单**（33 个含 NaN，4 个严重）:

| 因子 | NaN 数 | 占比 |
|---|---:|---:|
| gtja_gtja_030 | 1,298,335 | **100%** |
| alpha_alpha100 | 1,060,553 | **81.7%** |
| gtja_gtja_111 | 822,024 | **63.3%** |
| gtja_gtja_164 | 753,725 | **58.1%** |
| 其余 29 个 | 3 ~ 102,828 | <8% |

**影响量化**:
- top-10 因子列表不同: **28/195 日期 (14.4%)**
- 槽位变化: 42/1950 (2.15%)
- 股票池变化: 24/195 日期, 95 个股票切换
- 曾选中 NaN 因子: 130 槽位 (6.67%)，如 gtja_gtja_030 被选中 8 次（全 NaN → 零投票）

**修复**: `build_picks_v33_factorrank_nanfix.py`（`is_finite()` 替代 `is_not_null()`）

### Module 7 — 执行一致性 ✅ PASS (全部精确)

| 检查 | 结果 |
|---|---|
| Entry price = panel open × 1.001 (滑点) | **1587/1587 匹配**（max_rel 0.10%=精确滑点） ✅ |
| Exit price = panel open × 0.999 | 匹配 ✅ |
| Commission = 名义金额 × 0.25%/边 | max_rel_diff=0.0000 **精确** ✅ |
| Lot size 100 整数倍 | 1587/1587 ✅ |
| **NAV 精确对账** | Σ(fills现金流) + 期末持仓 mark = NAV change，**精确到分** ✅ |
| 期末持仓解释 | 688012.SH 725,700股（停牌股无法卖出），mark=272.72（2025-12-18 最后收盘） ✅ |
| 时区 | AKQuant 时间戳 = UTC+8 午夜边界，已确认 ✅ |

**NAV 对账恒等式**（已验证）:
```
NAV change 2,279,838,978
= Σ closed-symbols PnL (2,302,643,133)
+ 688012.SH 净贡献 (mark 197,912,904 - 成本 220,717,059 = -22,804,155)
✓ 精确成立
```

### Module 8 — 持仓 Cap 机制失效 🔴 CRITICAL 已修复

**根因**:
```python
# 错误（runner 原实现，line 258）:
panel_days = sorted({df.index[0] for sym, df in data.items()})
# df.index[0] = 每只股票的第一个日期 = 全部 2010-01-04 → 集合只有 1 个元素!

# 后果: add_trading_days() 任何输入都返回 2010-01-04
# → force_exit 条件 "2010-01-04 <= 当前日期" 恒为真
# → 每个持仓在注册次日被强制平仓!!!
```

**Debug 实锤**（插桩运行 2010-01 ~ 2010-03）:
```
DEBUG panel_days len=1 content=[datetime.date(2010, 1, 4)]   ← bug
DEBUG force_exit(2010-01-07): entries=10 stale=10            ← 次日全平
DEBUG close_position(002236.SZ) OK
...×10
```

**实际行为 vs 设计**:

| | 设计 | 实际（buggy） | 实际（capfix 后） |
|---|---|---|---|
| 主仓位持仓 | 20-21 bars | 1 bar（或 T+1 拒绝后 20 bars） | 20-21 bars ✅ |
| 1-bar 交易占比 | ~0% | **53.5% (870笔)** | 4.9% (78笔, top-up 残留) |
| 1-bar 交易总 PnL | — | **-¥735M**（win 39%） | 小额 |
| 20-bar 交易总 PnL | — | +¥12.3B（win 66%） | 主力 |

**机理**: 持仓次日的强平卖单遇到 T+1 约束时被拒绝（`Insufficient available position`），被拒的交易幸存到下个 rebalance（20 bars）。所以原版结果是"1-bar 强平"与"20-bar 幸存"的混合体，**完全偏离设计**。

**修复**: `v37_run_akquant_sweep_capfix.py`（构建完整交易日历）
```python
panel_days = sorted({d for sym, df in data.items() for d in df.index})  # FIX
```

**验证**: 修复后持仓分布 = 80.1%@20bars + 13.6%@21bars + 4.9%@1bar(top-up) ✅ 符合设计

---

## 2. 修复影响矩阵（lb=20）

| 组合 | Total | Annual | Sharpe | MDD | PF | Win | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. buggy + legacy（原 baseline） | +15,948.2% | +37.35% | 1.551 | -32.1% | 2.825 | 57.5% | 1627 |
| 2. buggy + nanfix | +13,720.5% | +36.08% | 1.523 | -33.4% | 2.763 | 57.4% | 1626 |
| 3. capfix + legacy | +304,402.9% | +65.09% | 1.951 | -24.9% | 3.393 | 63.4% | 1585 |
| **4. capfix + nanfix（推荐）** | **+209,875.9%** | **+61.30%** | **1.879** | **-24.7%** | **3.191** | **63.2%** | **1583** |

**推荐配置 = 组合 4**（双重修复）

### 组合 4 年度收益

| 年 | 收益 | | 年 | 收益 |
|---|---:|---|---|---:|
| 2010 | +14.97% | | 2018 | **-3.39%** |
| 2011 | +0.70% | | 2019 | +120.64% |
| 2012 | +32.99% | | 2020 | +165.96% |
| 2013 | +29.51% | | 2021 | +145.32% |
| 2014 | +64.69% | | 2022 | +130.19% |
| 2015 | +167.89% | | 2023 | +48.11% |
| 2016 | +9.51% | | 2024 | +81.14% |
| 2017 | +30.48% | | 2025 | +94.64% |

**仅 2018 一年为负**（-3.39%）。

---

## 3. 遗留限制与警示

| # | 限制 | 说明 |
|---|---|---|
| 1 | **Single-run (offset=0)** | 需重跑 5-10 offsets 量化方差（两个 bug 修复后） |
| 2 | **2026 OOS 需重验** | 原 +11.99% 是 buggy cap 下结果；需用修复版重跑 |
| 3 | **未复权价格** | 除权跳空 1.19% 窗口污染；方向=低估 |
| 4 | **容量限制** | NAV 增至 ¥210B（10 股组合），大额资金不可行；回测未建模冲击成本 |
| 5 | **In-sample** | 因子选择基于 2010-2025 全窗口；OOS 重验前禁止 production 宣称 |
| 6 | 停牌股 | 17 笔超 22-bar 交易为停牌无法卖出（真实行为，非 bug） |

---

## 4. 文件清单

### 审计脚本（examples/）
```
audit_v33_module1_panel.py         # 面板完整性
audit_v33_module2_fwdnet.py        # fwd_net 公式
audit_v33_module3_exrights.py      # 除权污染量化
audit_v33_module4_factors.py       # 因子值抽查
audit_v33_module5_scoring.py       # 评分复现+因果
audit_v33_module6_nan.py           # NaN bug 量化
audit_v33_module7_execution_v2.py  # 执行一致性（v2 时区修复版）
audit_v33_module8_surgical.py      # top-10 对比手术分析
v37_run_akquant_debug.py           # 插桩调试 runner
```

### 修复版本（examples/）
```
build_picks_v33_factorrank_nanfix.py    # NaN 修复版 picks builder
v37_run_akquant_sweep_capfix.py         # Cap 修复版 runner
```

### 审计产物（evidence/audit_v33_20261007/）
```
audit_01_panel_integrity.json
audit_02_fwd_net.json
audit_03_exrights.json
audit_04_factor_values.json
audit_05_scoring.json
audit_06_nan_impact.json
audit_07_execution_v2.json
audit_08_surgical.json
null_rates.json                     # 428 因子完整 null 率
debug_run/                          # 插桩运行产物（2010-01 ~ 2010-03）
REPORT.md                           # 本文件
```

### 结果数据（evidence/sweep/）
```
v33_factorrank_lb20_20261007/            # 组合 1: buggy + legacy
v33_factorrank_lb20_nanfix_20261007/     # 组合 2: buggy + nanfix（picks+akquant+result）
v33_factorrank_lb20_capfix_20261007/     # 组合 3: capfix + legacy（akquant+result）
v33_factorrank_lb20_bothfix_20261007/    # 组合 4: capfix + nanfix（推荐）
```

---

## 5. 建议后续步骤（待用户决策）

1. **重跑多 offset 验证**（组合 4 配置，offsets 0/2/4/6/8/10/12/14/16/18）→ 量化方差
2. **2026 OOS 重验**（capfix + nanfix 版本）
3. **复权价格重构**（消除除权污染）
4. **更新策略文档**（README / STRATEGY.md 中的旧数字标记 SUPERSEDED）

---

**审计完成时间**: 2026-10-07
**审计者**: Buffett (QF)
**结论**: 原 baseline 数字（+15,948% / +37.35%）产生于 2 个 bug 之上，**已被组合 4 数字取代**（+209,876% / +61.30%）。修复方向均为最小单行改动，各自独立验证。
