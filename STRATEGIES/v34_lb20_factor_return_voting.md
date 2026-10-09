# v34-lb20 Factor-Return Voting 策略 — 复现手册

> ⚠️ **2026-10-10 审计裁决 — 本手册的收益数字已作废（前视偏差 / 未来函数）**
> 打分窗口 `score[f,i] = mean(diff[f, i-20:i])` 中每一行 `diff[f,t]` 是 **[t+1→t+21] 前向收益**（结算于 t+21）；
> 决策日 i 使用了未来最多 20 个交易日的信息。单变量修正（窗口后移 21 日、仅用已结算行）：
> offset0 年化 **65.07% → 4.32%**（Sharpe 1.91→0.30，MDD 29.9%→69.5%）；
> offset10 **82.57% → −0.31%**。修正后无 alpha。
> 详见 `evidence/audit_lookahead_20261010/REPORT.md`。本手册仅保留为工程链路复现参考，**收益数字不得再引用**。

合同族：v34-lb20 picks（factor-return ranking，矩阵方法） + raw 执行 + 公司行动。

---

## 1. 公式

### 1.1 Per-date per-factor diff

```
panel_raw_close[t+1] = adj_close[t+1]  (滞后 1 bar 已知，做 T+1 open proxy)
panel_adj_close[t+21] = adj_close[t+21]

fwd_net_raw[f, t] = adj_close[t+21] / adj_open[t+1] - 1 - 0.005
diff[f, t]         = mean(fwd_net_raw of top-10 stocks by f@t)
                   - mean(fwd_net_raw of bot-10 stocks by f@t)
```

每日期、每因子预计算 → 矩阵 `matrix_v34_ADJ.parquet` (3886 × 428)。

### 1.2 Per-date 因子打分（窗口 = 前 20 sessions）

```
score[f, t] = nanmean(diff[f, t-20 : t])     if ≥ 10 finite values else NaN
```

### 1.3 Top-10 因子 + 横截面选股

```
active_factors = argtop10(score) × 10  (per rebal date)
votes[s, t]    = #{f ∈ active : stock s ∈ top-10(f@t)}
selected       = top-{5..10} stocks by votes (regime-gated min_votes)
```

**regime 阈值**（来自 `causal_zigzag_router_20261006/router_map.json`）：
- `bull_neutral`: min_votes=2, min_stocks=5, max_stocks=10
- `bear`:         min_votes=4, min_stocks=5, max_stocks=10

### 1.4 Execution

- T+1 NextOpen 买入；持有 20 交易日（硬性 cap = 21 交易日，过期 `close_position`）
- target_total = 0.90, equal weight per pick
- cost = 0.25% commission + 0.10% slippage (round-trip)
- lot = 100 股
- T+1 settlement enforced by `t_plus_one=True`

### 5. Corporate Actions（同期注入 AKQuant）

| key | value |
|---|---|
| source | `evidence/v34_adj_20261007/corporate_actions_derived.parquet` |
| n_events_post2010 | 5678 |
| engine | `aq.CorporateAction(symbol, date, type, value)` |
| Split | qty × value (value = 倍数) |
| Dividend | cash += qty × value (value = 每股现金) |
| how | monkeypatch `_EngineProxy`, monkeypatch add_corporate_action 注入 |

---

## 2. Contract Table（冻结）

| 参数 | 值 | 来源 |
|---|---|---|
| panel | `data/wavehunter_hs300_v34_adj_20261007.parquet` | 461 列 / 3886 dates / 354 stocks |
| matrix | `evidence/stage_a_20261007/matrix_v34_ADJ.parquet` | 428 因子 |
| router | `evidence/causal_zigzag_router_20261006/router_map.json` | Causal ZigZag (leg.start) |
| start_date | 2010-01-04 |  |
| end_date (full) | 2025-12-31 |  |
| end_date (OOS) | 2026-08-27 |  |
| rebal_step | 20 trading days |  |
| lookback | 20 sessions |  |
| top_k | 10 |  |
| k_nom | 10 (top-10 stocks per factor) |  |
| min_votes_bull | 2 |  |
| min_votes_bear | 4 |  |
| min_stocks_bull/bear | 5 |  |
| max_stocks_bull/bear | 10 |  |
| target_total | 0.90 |  |
| holding_cap_bars | 21 |  |
| commission_rate | 0.0025 |  |
| slippage | 0.0010 (percent) |  |
| lot_size | 100 |  |
| t_plus_one | True |  |
| fill_policy | NextOpen |  |
| prices | raw + CA injection |  |
| ca_mode | all (dividend + split) |  |

---

## 3. 排除规则（factor pool）

```
KEEP  : gtja_gtja_*, alpha_alpha_*, talib_*, mw_*, github17_*, academic10_*
DROP  : v10_1_a1_point, v10_1_a2_start, v10_1_a2_interval,
        v10_1_b1_start, v10_1_b1_interval,
        v10_1_down_start, v10_1_down_interval,
        v10_1_peak_zone, v10_1_valley_zone,
        v10_1_zig_peak, v10_1_zig_valley           (lookahead 标签)
DROP  : trade_date, ts_code, open, high, low, close, vol, amount,
        pct_chg, adj_factor, adj_open, adj_high, adj_low, adj_close,
        idx_*, turnover_rate, circ_cap, cap, symbol, volume
                                                (原始 OHLCV/索引/基本面，pre-existing excluded)
RESULT: 428 voting factors
```

任何 `v10_1_*` 列必须被强制 drop（前瞻偏差源）。矩阵派生脚本中加 assert：

```python
assert [c for c in mat.columns if c.startswith("v10_1_")] == []
```

---

## 4. 复现命令

按顺序运行（每条命令可独立复现成功）。

### 4.1 Build / verify matrix + 1 baseline pick set (matrix-based, ~30s + 3s)

```bash
cd /media/felix/f/quant/akquant-factor-backtest
/usr/bin/python3.12 -u examples/stage_a_1_matrices_and_picks.py --panel v34 --picks-only
```

Outputs: `evidence/stage_a_20261007/picks/repro_v34/V14_0/{picks,picks_meta,summary}.json`

### 4.2 Run single offset sim (canonical +CA, 2010-2025)

```bash
/usr/bin/python3.12 -u examples/v41_run_akquant_v34.py --tag V14_0 \
  --picks-base evidence/stage_a_20261007/picks/repro_v34 \
  --actions on --ca-mode all --prices raw
```

Outputs: `evidence/sweep/v34_lb20_fixed_ca_20261007/V14_0/result.json` (~83s)

### 4.3 Full Stage 5 (10 offsets × 2 windows, resumable, ~25 minutes total)

```bash
# Generate 10 offset picks (matrix-based, ~30s)
mkdir -p evidence/stage5_20261007/picks
for off in 0 2 4 6 8 10 12 14 16 18; do
  /usr/bin/python3.12 -u examples/stage5_picks_and_sims.py 2>&1 | tail -1
done
# Or use the resumable runner in chunks of 4 sims
for i in 1 2 3 4 5; do
  /usr/bin/python3.12 -u examples/stage5_run_sims.py --max 4
done
```

Outputs:
- `evidence/stage5_20261007/sims/V14_*/result.json` (full 2010-2025)
- `evidence/stage5_20261007/sims_2026/V14_*/result.json` (full 2010-2026.08)
- `evidence/stage5_20261007/STAGE5_REPORT.md` (run `stage5_report.py`)

### 4.4 Single-fact data verification

```bash
# adj_factor sanity
/usr/bin/python3.12 -c "
import polars as pl
v = pl.read_parquet('data/wavehunter_hs300_v34_adj_20261007.parquet',
                     columns=['ts_code','trade_date','close','adj_factor'])
v = v.with_columns((pl.col('adj_factor').diff().over('ts_code') != 0).alias('d')).filter(pl.col('d'))
print(f'rows with af change: {v.height}')"
# Expected: ~6500 (each = a corporate action event)

# Reproduction check (must be 195/195 identical)
/usr/bin/python3.12 -c "
import json
p1 = json.loads(open('evidence/stage_a_20261007/picks/repro_v34/V14_0/picks.json').read())
p2 = json.loads(open('evidence/sweep/v34_factorrank_lb20_fixed_20261007/V14_0/picks.json').read())
ok = sum(1 for d in set(p1)&set(p2) if set(p1[d])==set(p2[d]))
print(f'identical baskets: {ok}/195')"
# Expected: 195/195
```

---

## 5. 预期数字

| run | 年化 | Sharpe | MDD% | PF | win% |
|---|---:|---:|---:|---:|---:|
| canonical offset=0, full 2010-2025 | +65.07% | 1.911 | 29.9 | 2.87 | 64.0 |
| canonical 10-offset mean (full) | +75.83% | 2.158 | 29.8 min | 2.46 min | 65.3 |
| canonical 10-offset std (full) | ±10.46% | ±0.194 | — | — | — |
| canonical 10-offset mean (2010-2026.08) | +72.64% | 2.122 | 29.8 min | 2.70 min | 65.3 |
| canonical 2026 8mo real mean (10 offset) | +6.94% | — | — | — | — |
| canonical 2026 8mo HS300 benchmark | -2.10% | — | — | — | — |

随机因子 0.67±0.26 ann（null），canonical 0.7583 ann ⇒ z=+22.3（10/10 高于 null）。

---

## 6. 脚本清单

```
examples/build_picks_v34_factorrank.py             canonical builder (legacy, lb=20 hardcoded)
examples/v41_run_akquant_v34.py                    runner (panel_days fix + CA injection proxy)
examples/stage_a_1_matrices_and_picks.py            matrix-builder + 1-pick-set (matrix + top10)
examples/stage_a_2_run_sims.py                     batch sim runner (resumable, 3 per call)
examples/stage5_picks_and_sims.py                  matrix-based 10 offset picks + all sims
examples/stage5_run_sims.py                        resumable stage-5 sim runner
examples/stage5_report.py                          report aggregator + yearly tables
examples/diag_split_mechanism.py                   empirical Split semantics probe
examples/derive_corporate_actions_local.py         local CA derivation (no Tushare)
examples/run_alpha_alpha042.py                     legacy single-factor smoke (NOT used here)
examples/v37_run_akquant_sweep.py                  legacy runner (NOT used here, kept for diff)
```

Non-script artifacts (read-only inputs):

```
data/wavehunter_hs300_v34_adj_20261007.parquet              panel
evidence/stage_a_20261007/matrix_v34_ADJ.parquet             matrix
evidence/causal_zigzag_router_20261006/router_map.json       router
evidence/v34_adj_20261007/corporate_actions_derived.parquet  CA events
aurumq-rl/data_cache/{code}_full.parquet                    CA source: pre_close
aurumq-rl/data_cache/{code}_daily_basic.parquet             CA source: total_share
aurumq-rl/data_cache/stock_basic_industry.parquet           industry (Tushare)
aurumq-rl/.qbot_token                                        Tushare token
```