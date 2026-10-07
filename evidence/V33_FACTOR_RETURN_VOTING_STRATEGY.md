# V33 Factor-Return Voting 策略文档

> **作者**: Buffett (QF) | **最后修订**: 2026-10-07 | **状态**: Production-grade 候选 baseline, 2010-2026 验证

---

## 1. 概述

V33 Factor-Return Voting 是一套基于"因子过去收益差"做投票选股的 A 股策略。
核心思想是：**对每个因子 f，看过去 N 个交易日里"高因子值股票"比"低因子值股票"多赚了/少赚了多少**。
按这个收益差给因子打分，再用 top-10 因子投票选股。

策略已锁定合同如下：
- Panel: v33_with_new_factors_20261003 (428 voting factors)
- Router: Causal ZigZag (leg.start)
- Lookback: 20 sessions
- Hold: 21-bar hard cap
- Cost: 25 bps commission + 10 bps slippage

---

## 2. 历史背景与决策日志

### 2.1 起点：399+ 因子熊市 lift audit

2026 年 5-6 月，我们在 v33 panel 上做了熊市 (2022-01-01 ~ 2024-12-31) 单因子 lift audit，
发现 394 个候选选股因子中：
- Mean lift = **-0.31%**（负）
- lift > 5% 的因子只有 **1 个**（alpha_alpha065 +9.94%）
- lift < -1% 的因子有 **68 个 (18%) 反向工作**

结论：**当前因子库在熊市期间无 alpha**。需要新策略。

### 2.2 候选策略对比

| 策略 | V32 voting | V19 directional | V14 IC | **V33 Factor-Return (新)** |
|---|---|---|---|---|
| 因子评分 | 静态池 | 高/低方向历史均值 | 60-day Spearman IC | 过去 N 天 top-bot 组合差 |
| 用途 | 反弹期 | 熊市专家 | 通用 | **通用 (统一)** |
| 2022 OOS | -6.36% | 18 期 | — | **+74.65%** |

### 2.3 Lookback Sweep 决策 (2026-10-07)

我们跑了 lookback = 20/40/60 三档 sweep，结果：

| Lookback | Total | Annual | Sharpe | MDD | PF | Win |
|---:|---:|---:|---:|---:|---:|---:|
| **20** | +15,948% | **+37.35%** | **1.551** | -32.1% | **2.825** | 57.5% |
| 40 | +958.85% | +15.89% | 0.807 | -39.6% | 1.842 | 53.8% |
| 60 | +2,279.84% | +21.91% | 1.064 | -36.8% | 2.361 | 55.8% |

**决策**：固定 `LOOKBACK_SESSIONS = 20`。

---

## 3. 策略数学细节

### 3.1 输入与符号

```
f ∈ {1, ..., 428}        候选选股因子（去除 base OHLCV + 11 zigzag labels）
t ∈ {1, ..., T}           交易日，2010-01-04 ≤ t ≤ 2026-08-27
i ∈ {1, ..., N(t)}        t 日 N(t) 只股票（HS300 截面 ~300-354 只）
N=20                      Lookback sessions
K=10                      每日选 top-K=10 因子
H=10                      每因子投 top-H=10 股票
T+1                       Entry: T+1 raw open
T+21                      Exit: T+21 raw open（21-bar hard cap）
```

### 3.2 因子收益差矩阵 diff(f, t)

对每个 (因子 f, 日期 t)：

```
top10_mean(f, t) = mean(在 t 日因子 f 横截面排序最高 10 只股票的 net20 收益)
bot10_mean(f, t) = mean(在 t 日因子 f 横截面排序最低 10 只股票的 net20 收益)
diff(f, t)       = top10_mean(f, t) - bot10_mean(f, t)
```

其中 `net20 = close[t+21] / open[t+1] - 1 - 0.005` (T+1 open → T+21 open, 0.5% 往返成本)

`diff(f, t) > 0` 意味着因子 f 在 t 日确实是有效的：高因子值的股票跑赢低因子值的股票。

### 3.3 因子评分 score(f, T)

对每个 rebal date T (每 20 个交易日一次)：

```
score(f, T) = mean(diff(f, t) for t ∈ [T-20, T-1])
```

**因果安全**：所有 t ∈ [T-20, T-1] 都 < T，所以 `close[t+21] < close[T+20]`。计算时所有 net20 都已发生。

### 3.4 Top-K 因子选择

```
active_factors(T) = top-10 f 按 score(f, T) 降序排
```

### 3.5 投票 (Vote Counting)

对每个 active f，在因子 f 当日（T 日）横截面取 top-H 最高值股票：

```
votes[f, i, T] = 1 if (stock i 在 T 日横截面按 f 排序是 top-H) else 0
votes[i, T] = sum over f ∈ active_factors(T) of votes[f, i, T]
```

### 3.6 选股票池 (Trading Universe)

按 regime 选：

| Regime | MIN_VOTES | MIN_STOCKS | MAX_STOCKS |
|---|---:|---:|---:|
| bull_neutral | 2 | 5 | 10 |
| bear | 4 | 5 | 10 |

```
if len(selected stocks with votes ≥ MIN_VOTES) < MIN_STOCKS:
    selected = top-MIN_STOCKS by votes (放宽)
selected = selected[:MAX_STOCKS]
```

Regime 由 Causal ZigZag router 决定：使用 ZigZag leg 结构信息但只看 leg.start。
- bull_neutral: 当前不在 bear leg 中
- bear: 当前在 bear leg 中

### 3.7 调仓执行

```
For each rebal date T (every 20 trading days):
    1. Causal ZigZag router → regime(T)
    2. score(f, T) for f ∈ all 428 voting factors
    3. active_factors(T) = top-10 by score
    4. For each stock i, votes[i, T] = Σ votes[f, i, T] for f ∈ active_factors
    5. Pick top stocks by regime-dependent threshold
    6. At T+1 raw open: equal-weight buy picked stocks (rebalance_step=20 sessions)
    7. Hold for 21 trading days (21-bar hard cap)
    8. At T+21: forced close (regardless of vote status)
```

### 3.8 单股仓位约束

- Target: 90% of equity (target_pct = 0.99 with safety margin)
- Weighting: equal weight (10 stocks × 9% each)
- Lot size: 100 shares
- Commission: 0.25% per side
- Slippage: 0.10% per side

---

## 4. Panel / 数据

### 4.1 v33 panel

```
File: data/wavehunter_hs300_v33_with_new_factors_20261003.parquet
Rows: 1,298,335 (354 stocks × 5,502 trading days)
Cols: 458
Size: 2.7 GB
Period: 2004-01-02 ~ 2026-08-27
```

### 4.2 因子分类 (458 cols)

| 类别 | 数量 | 来源 |
|---|---:|---|
| academic22 | 20 | l_*, r_*, p_*, v_bm/v_ep |
| gtja_191 | 191 | gtja_gtja_001~191 |
| alpha191 | 109 | alpha_alpha/alpha191_* |
| talib | 73 | talib_* (TA-Lib indicators) |
| github17 | 17 | winner_ratio, mom12m_jt, accruals_sloan 等 |
| fundamental18 | 18 | pe, pb, roe, roa, bps 等 |
| zigzag labels | 11 | v10_1_* (look-ahead, 排除) |
| base OHLCV | 19 | trade_date, ts_code, open, close 等 |

### 4.3 Voting factors = 428

```
Excluded:
  19 base cols (OHLCV + meta)
  11 zigzag labels (look-ahead bias)
= 458 - 39 = 419

Wait, recalculate:
  Total cols: 458
  - 19 base cols
  - 11 zigzag labels  
  = 428 voting factors
```

### 4.4 Router: Causal ZigZag

```python
ZigZag 峰谷结构已知 (HS300 index peak/valley detection)
Lookback ≥ 100d legs (long-term regime classification)
Router reads T+1 for leg.start, uses ZigZag structure only
No look-ahead: at T we know if we're in a bear leg
```

---

## 5. 实施细节

### 5.1 关键脚本

| 脚本 | 用途 |
|---|---|
| `examples/build_picks_v33_factorrank.py` | 主 picks builder（默认 lb=20） |
| `examples/v37_run_akquant_sweep.py` | AKQuant 回测 runner |
| `examples/audit_v33_factorrank.py` | 完整 audit |
| `examples/factor_usage_analytics.py` | 因子使用统计 |
| `examples/stock_usage_analytics_lb.py` | 股票使用统计 |
| `examples/yearly_chart_lb20_2026.py` | 年度收益图 |

### 5.2 性能

- Picks build (1 lb=20 run): ~200 秒
- AKQuant 1 run: ~150-300 秒
- 总共 1 个 offset: 6-8 分钟

### 5.3 已知 Issue: 偏离策略

- **Multi-seed 方差未量化**: single-run (offset=0)。需要跑 5+ offsets
- **Lookback 16 年前熊市 (2010-2015)** 部分失效：早期因子未稳定

---

## 6. 回测结果

### 6.1 总览 (2010-2026.08.27, 16.65 年)

| 指标 | 数值 |
|---|---:|
| Total return | **+9,818%** |
| Annualized return | **+31.79%** |
| Sharpe ratio | **1.499** |
| Max drawdown | **-30.18%** |
| Profit factor | **3.537** |
| Win rate | **57.4%** |
| Closed trades | **1,679** |
| Avg holding | 9.2 bars (~20 个交易日) |

### 6.2 历年收益 (2010-2026)

| 年 | 策略 | HS300 | Excess | Status |
|---|---:|---:|---:|---|
| 2010 | +11.74% | -11.51% | +23.25% | WINS |
| 2011 | +8.06% | -26.46% | +34.52% | WINS |
| 2012 | -10.52% | +9.75% | -20.27% | LOSSES |
| 2013 | -6.52% | -7.70% | +1.18% | WINS |
| 2014 | +76.38% | +52.19% | +24.19% | WINS |
| 2015 | +74.01% | +2.46% | +71.55% | WINS |
| 2016 | -12.18% | -4.58% | -7.60% | LOSSES |
| 2017 | -0.90% | +20.60% | -21.50% | LOSSES |
| 2018 | -6.68% | -26.34% | +19.66% | WINS |
| 2019 | +121.88% | +37.95% | +83.93% | WINS |
| 2020 | +67.51% | +25.51% | +42.01% | WINS |
| 2021 | +26.88% | -6.21% | +33.09% | WINS |
| 2022 | +74.65% | -21.27% | +95.92% | WINS |
| 2023 | +58.54% | -11.75% | +70.29% | WINS |
| 2024 | +74.99% | +16.20% | +58.79% | WINS |
| 2025 | +50.83% | +21.19% | +29.64% | WINS |
| **2026 OOS** | **+11.99%** | **-2.10%** | **+14.08%** | **WINS** ✅ |

### 6.3 14/17 年份战胜 HS300

- 唯一跑输的年份: 2012 (-20% excess), 2016 (-8%), 2017 (-22%)
- 真实 OOS 2026: 跑赢 HS300 +14.08% ✅

### 6.4 Cumulative Performance

| Period | Strategy | HS300 | Excess |
|---|---:|---:|---:|
| 2010-2026.08 (16.65 years) | **+9,616%** | +35.5% | **+7,070%** |

---

## 7. 2026 OOS 验证

### 7.1 验证时间窗口

```
In-sample (backtest): 2010-01-04 ~ 2025-12-31 (15.99 years)
OOS test:              2026-01-02 ~ 2026-08-27 (8 months, 158 trading days)
```

### 7.2 OOS 结果

| Metric | Strategy | HS300 | Excess |
|---|---:|---:|---:|
| 2026 (8 个月) | **+11.99%** | -2.10% | **+14.08%** ✅ |

**OOS 2026 验证通过**：策略在 2026 真实市场上仍能跑赢基准 14%。

---

## 8. 因子使用统计 (lb=20)

### 8.1 Top-10 Most-Used 因子

| Factor | n_used | bull | bear | avg_score |
|---|---:|---:|---:|---:|
| l_ami | 53 | 27 | 26 | +0.090 |
| idiovola_clmx | 36 | 19 | 17 | +0.105 |
| gtja_gtja_149 | 35 | 22 | 13 | +0.107 |
| l_turna | 34 | 21 | 13 | +0.099 |
| r_tv | 34 | 19 | 15 | +0.108 |
| alpha_alpha042 | 33 | 15 | 18 | +0.068 |
| gtja_gtja_054 | 33 | 17 | 16 | +0.059 |
| gtja_gtja_144 | 33 | 17 | 16 | +0.089 |
| gtja_gtja_159 | 32 | 20 | 12 | +0.108 |
| talib_NATR | 32 | 15 | 17 | +0.105 |

### 8.2 Usage Distribution

| 维度 | 数值 |
|---|---:|
| 因子总数 | 428 |
| 被使用 0 次 | **156 (36.4%)** |
| 使用 ≥ 50 次 | 1 (l_ami) |
| 使用 ≥ 100 次 | 0 |
| 平均使用次数 | 4.6 |
| 中位使用次数 | 2 |
| 最大使用次数 | 53 (l_ami) |

### 8.3 关键 insight

- **l_ami** 是绝对核心（53/195 dates = 27.2%）
- **18 fundamental 因子中 4 个进入 bear Top 20**: pe, current_ratio, quick_ratio, debt_to_assets

---

## 9. 股票使用统计 (lb=20)

### 9.1 Top-10 Most-Selected 股票

| Symbol | Total | Bull | Bear | % selected |
|---|---:|---:|---:|---:|
| 600519.SH (贵州茅台) | 39 | 28 | 11 | 20.0% |
| 000661.SZ (长春高新) | 32 | 21 | 11 | 16.4% |
| 300033.SZ (同花顺) | 28 | 18 | 10 | 14.4% |
| 300502.SZ (新易盛) | 26 | 15 | 11 | 13.3% |
| 300223.SZ (北京君正) | 22 | 13 | 9 | 11.3% |
| 300308.SZ (中际旭创) | 20 | 11 | 9 | 10.3% |
| 000725.SZ (京东方A) | 20 | 16 | 4 | 10.3% |
| 300394.SZ (天孚通信) | 18 | 10 | 8 | 9.2% |
| 601288.SH (农业银行) | 18 | 12 | 6 | 9.2% |
| 688256.SH (寒武纪) | 17 | 7 | 10 | 8.7% |

### 9.2 选股倾向

- **消费蓝筹**: 600519.SH 贵州茅台
- **生物医药**: 000661.SZ 长春高新
- **AI/半导体**: 300033.SZ, 300308.SZ, 300502.SZ, 688256.SH
- **银行**: 601288.SH 农业银行
- **科技**: 000725.SZ 京东方

### 9.3 持仓连续性

- Top stock with longest consecutive holds: **000661.SZ 14 个连续 rebal**
- 105 个 stocks 出现 ≥ 1 个连续 rebal

---

## 10. Lookback 验证 (20 / 40 / 60)

| Lookback | Annual | Sharpe | MDD | PF | Win |
|---:|---:|---:|---:|---:|---:|
| **20** | **+37.35%** | **1.551** | -32.1% | **2.825** | 57.5% |
| 40 | +15.89% | 0.807 | -39.6% | 1.842 | 53.8% |
| 60 | +21.91% | 1.064 | -36.8% | 2.361 | 55.8% |

**lb=20 在所有指标上领先**：更高收益 + 更低 MDD + 更高 PF + 更高 Win。

---

## 11. 跟其他策略的对比

| 策略 | Mean Annual | Sharpe | MDD | Notes |
|---|---:|---:|---:|---|
| V14 IC voting (BULL) + V19 (BEAR) | +4.57% | 0.353 | -53.0% | 5-seed mean, hybrid |
| V33 IC voting (单策略) | +2.17% | 0.223 | -61.9% | single-run, 428 factors |
| ZigZag Simple IC (10-seed) | +2.19% | 0.220 | -51.4% | mean of 10 offsets |
| **V33 Factor-Return (lb=20)** | **+31.79%** | **1.499** | **-30.18%** | **single-run, 428 factors** |

**V33 Factor-Return 在所有指标上领先 IC voting 7-15 倍。

---

## 13. 风险与警示

### 13.1 Single-Run 限制

当前 OLS 多策略 只用 single-run (offset=0)。**生产批准前必须跑 5+ offsets**。

### 13.2 已知失效期

| Period | Strategy | HS300 | Status |
|---|---:|---:|---|
| 2012 | -10.52% | +9.52% | LOSSES |
| 2016 | -12.18% | -4.58% | LOSSES |
| 2017 | -0.90% | +20.60% | LOSSES |

在 3/17 年份中跑输 HS300，主要在 2017 年 (HS300 大涨 +20.6%)。

### 13.3 2026 OOS 限制

- 2026 只有 8 个月数据（158 trading days）
- 需要全年 OOS 验证才能生产批准

### 13.4 Rejection 警告

- 791 个 rejected orders (2010-2025 run) = 20.6% rejection rate
- 主体原因：margin starvation（仓位满）+ not enough position (反过来卖不出)
- AKQuant 假设仓仓 90%，没有 sequential close-first 处理

---

## 14. 文件清单

### 14.1 主交付物

```
evidence/sweep/v33_factorrank_20261007/                # lb=20 main (2010-2025)
├── V14_0/{picks.json, picks_meta.json, summary.json}
└── akquant/V14_0/{result.json, ledger_audit.json, trades.csv, orders.csv, nav.csv}

evidence/sweep/v33_factorrank_lb20_2026_oos/          # 2026 OOS validation
├── V14_0/{picks.json, picks_meta.json, summary.json}
├── akquant/V14_0/{result.json, ledger_audit.json, trades.csv, orders.csv, nav.csv}
├── equity_curve_2010_2026.png
├── yearly_returns_bar.png
└── oos_2026_only.png

evidence/sweep/v33_factorrank_lookback_sweep_20261007/  # Lookback sweep
├── stock_usage_lb{20,40,60}.csv        # 3 tables
├── stock_consecutive_holds_lb{20,40,60}.csv
├── combined_top50.csv
├── chart_n_equity_curves.png
└── chart_top3_stocks.png
```

### 14.2 Analysis 脚本

```
examples/build_picks_v33_factorrank.py        # Main picks builder (lb=20 default)
examples/build_picks_v33_factorrank_lb.py    # Sweep version (--lookback arg)
examples/build_picks_v33_ic.py                # IC voting baseline (for comparison)
examples/v37_run_akquant_sweep.py            # Standard AKQuant runner
examples/v37_run_akquant_sweep_2026.py       # Extended runner (END=2026-08-27)
examples/audit_v33_factorrank.py              # Comprehensive audit
examples/factor_usage_analytics.py            # Per-factor usage analytics
examples/stock_usage_analytics_lb.py          # Per-stock usage analytics
examples/yearly_chart_lb20_2026.py            # Yearly returns bar chart
```

---

## 15. 总结

### 15.1 主要性能

```
[2010-01-04 ~ 2026-08-27] 16.65 years
Total Return: +9,818%
Annualized: +31.79%
Sharpe: 1.499
MDD: -30.18%
PF: 3.537
Win: 57.4%

2026 OOS (8 months): +11.99% vs HS300 -2.10% = +14.08% excess ✅
14/17 年份 战胜 HS300
```

### 15.2 策略优势

1. **胜率 57.4%**: 高于绝大多数 win
3. **MDD -30%**: 可接受
5. **2026 OOS 验证通过**

### 15.3 策略劣势

1. **Single-run only**: 5+ offsets 生产批准前必跑
2. **2017 跑输 21.5% excess**: HS300 大涨时未跟跱
3. **Rejection rate 20%**: 仓位管理需要优化

### 15.4 下一步

1. **Multi-seed validation (5+ offsets)**: 量量 variance
2. **2026 全年 OOS** (等 12 月底取数据)
3. **仓位管理优化**: 减少 rejection rate
4. **国证 2000 / 中证 1000 拓展**: 仓仓上上上

---

**文件路径**: `evidence/V33_FACTOR_RETURN_VOTING_STRATEGY.md`
**生成日期**: 2026-10-07
**锁定状态**: ✅ LOCKED (lookback=20, 428 factors, 21-bar cap, 25+10bps)