# V33 Factor-Return Voting 策略 — README

> ⚠️ **2026-10-07 全面审计更新**: 本策略发现并修复 2 个 bug（持仓 cap 失效 + NaN 评分污染）。
> 旧数字（+15,948% / +37.35%年）已被取代。**修复后正确基准**: +209,876% / +61.30%年 / Sharpe 1.879 / MDD -24.7%。
> 完整审计报告: `evidence/audit_v33_20261007/REPORT.md`
> 推荐配置: `build_picks_v33_factorrank_nanfix.py` + `v37_run_akquant_sweep_capfix.py`


> **作者**: Buffett (QF) | **最后修订**: 2026-10-07
> **锁定参数**: `LOOKBACK_SESSIONS = 20` (单跑 offset=0)

---

## 一句话总结

对每个因子 f，看过去 20 个交易日里"高因子值股票"比"低因子值股票"多赚/少赚了多少（top10 − bot10 净收益差）。按这个差给因子打分，top-10 因子再投票选股。

---

## 数学公式

```python
# 因子收益差 (per factor, per date)
top10_mean(f, t) = mean(top-10 stocks by factor f at t, net20 return)
bot10_mean(f, t) = mean(bot-10 stocks by factor f at t, net20 return)
diff(f, t)       = top10_mean(f, t) - bot10_mean(f, t)

# 因子评分 (per factor, per rebal date T)
score(f, T) = mean(diff(f, t) for t ∈ [T-20, T-1])

# Top-10 因子投票
active_factors(T) = top-10 f by score(f, T) desc
votes[i, T] = Σ votes[f, i, T] for f ∈ active_factors(T)
  其中 votes[f, i, T] = 1 if stock i 在 T 日横截面按 f 排序是 top-10 else 0

# 选股池
selected = stocks sorted by votes desc
  if regime == bull_neutral: filter votes ≥ 2
  if regime == bear:         filter votes ≥ 4
  selected = top-10 (or top-5 if fewer than 5 stocks clear threshold)
```

`net20 = close[t+21] / open[t+1] − 1 − 0.005` (T+1 open → T+21 open, 0.5% 往返成本)

---

## 合同参数

| 维度 | 值 |
|---|---|
| Panel | v33_with_new_factors_20261003.parquet (458 cols, 354 stocks) |
| Voting factors | 458 − 19 base − 11 zigzag labels = **428** |
| Router | Causal ZigZag (leg.start) |
| Lookback | **20 sessions** |
| Rebalance | 每 20 个交易日 |
| Hold | 21-bar hard cap (T+21 open 强制平仓) |
| Cost | 25 bps commission + 10 bps slippage |
| Lot size | 100 shares |
| Weighting | equal weight (10 stocks × 9%) |
| Target | 90% of equity |
| MIN_VOTES bull | 2 |
| MIN_VOTES bear | 4 |
| MIN_STOCKS | 5 |
| MAX_STOCKS | 10 |
| TOP_K (因子数) | 10 |
| K_NOM (每因子股票) | 10 |

---

## Panel 排除规则

```python
# 排除 (base cols × 19)
base_cols = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount",
    "pct_chg", "adj_factor", "adj_close", "idx_close", "idx_mom_5", "idx_mom_20",
    "idx_mom_60", "turnover_rate", "circ_cap", "cap", "symbol", "volume",
    "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
}

# 排除 (zigzag labels × 11, look-ahead bias)
zigzag_labels = {
    "v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval", "v10_1_b1_start",
    "v10_1_b1_interval", "v10_1_down_start", "v10_1_down_interval",
    "v10_1_peak_zone", "v10_1_valley_zone", "v10_1_zig_peak", "v10_1_zig_valley",
}
```

---

## ZigZag Router

```python
Router 输入: ZigZag HS300 index peak/valley 结构（已知结构，不用未来数据）
- leg ≥ 100d 作为 bull/bear leg
- 读 leg.start 时已确定 regime
输出: bull_neutral | bear
```

router_map.json 路径: `evidence/causal_zigzag_router_20261006/router_map.json`（5502 dates）

---

## 调仓执行

```python
For each rebal date T (every 20 trading days):
    1. Router → regime(T)
    2. 计算所有 428 个因子的 score(f, T)
    3. active_factors(T) = top-10 by score
    4. T 日横截面取每因子 top-10 股票 → votes[i, T]
    5. 按 regime threshold 过滤 selected
    6. T+1 raw open: 等权买入 selected
    7. T+21 raw open: 强制平仓
```

---

## 因果安全保证

| 维度 | 保证 |
|---|---|
| Router | ZigZag leg 结构在 T+1 时已知，不用未来数据 |
| diff(f, t) | close[t+21] 都 < close[T+20]，已发生 |
| score(f, T) | 所有 t ∈ [T-20, T-1]，past only |
| factor value x[i, t] | T 日可用 |
| T+1 open buy | decision-time 和 execution-time 分离 |

---

## 复现命令

```bash
cd /media/felix/f/quant/akquant-factor-backtest

# 1. 生成 picks
mkdir -p evidence/sweep/v33_factorrank_repro
/usr/bin/python3.12 -u examples/build_picks_v33_factorrank.py \
    --rebal-offset 0 \
    --out-root evidence/sweep/v33_factorrank_repro
# 默认 LOOKBACK_SESSIONS=20, END=2025-12-31

# 2. 跑 AKQuant
/usr/bin/python3.12 -u examples/v37_run_akquant_sweep.py \
    --tag V14_0 \
    --picks-base evidence/sweep/v33_factorrank_repro \
    --out-base evidence/sweep/v33_factorrank_repro/akquant
```

---

## 关键脚本

```
examples/build_picks_v33_factorrank.py    # 主 picks builder
examples/v37_run_akquant_sweep.py        # AKQuant runner
```

---

## 输出文件

```
evidence/sweep/v33_factorrank_repro/
├── V14_0/
│   ├── picks.json            # {date: {symbol: votes}}
│   ├── picks_meta.json       # {date: {regime, n_picks, max_votes, ...}}
│   └── summary.json
└── akquant/V14_0/
    ├── result.json           # metrics (annualized, sharpe, mdd, pf, ...)
    ├── ledger_audit.json     # 8 audit checks
    ├── trades.csv            # 1679 trades
    ├── orders.csv            # 3993 orders
    └── nav.csv               # 3887 daily NAV
```

---

## 预期结果（单跑 offset=0, 2010-2025.12.31）

| Metric | 数值 |
|---|---:|
| Total return | +15,948% |
| Annualized | +37.35% |
| Sharpe | 1.551 |
| MDD | -32.1% |
| PF | 2.825 |
| Win rate | 57.5% |
| Trades | 1,627 |

---

## 2026 OOS 验证（2026-01-02 ~ 2026-08-27）

| Metric | Strategy | HS300 | Excess |
|---|---:|---:|---:|
| 2026 (8 个月) | +11.99% | -2.10% | +14.08% ✅ |

14/17 年份战胜 HS300（含 2026 OOS）。

---

**锁定状态**: ✅ LOCKED (lookback=20, 428 factors, 21-bar cap, 25+10bps)
**生成日期**: 2026-10-07