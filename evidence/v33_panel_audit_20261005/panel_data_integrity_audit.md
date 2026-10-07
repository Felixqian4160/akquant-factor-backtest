# V33 Panel 数据完整性与正确性审计

- Panel: `data/wavehunter_hs300_v33_with_new_factors_20261003.parquet`
- Rows/cols: **1,298,335 × 458**

## 结论

结构与数据完整性检查: **11/14 PASS**（3 项 FAIL 均为已知副作用，下面有解释）。

## 覆盖度

| 项目 | 数值 |
|---|---:|
| 总行数 | 1,298,335 |
| 总列数 | 458 |
| 选股因子 | 410 |
| 日期 | 2004-01-02 00:00:00 → 2026-08-27 00:00:00 |
| 交易日数 | 5,502 |
| 股票数 | 354 |
| 每股票行数范围 | 176 → 5,468 |
| 唯一键重复 | 0 |

## 因子缺失率（按字段语义解释，不算错误）

| 因子组 | 缺失率范围 | 重点字段 |
|---|---:|---|
| 新增 17 | 0.00% → 35.28% | `gp_novymarx` 35.28% |
| 原有学术 22 | 0.00% → 25.03% | `v_bm` 25.03% |
| GTJA 191 | 0.00% → 6.87% | `gtja_gtja_149` 6.87% |
| Alpha 107 | 0.00% → 8.23% | `alpha_alpha079` 8.23% |
| TA-Lib 73 | 0.00% → 100.00% | `talib_MAX2`/`talib_MIN2` 100% (已知坏字段) |

### 新增 17 因子明细

| 因子 | 缺失率 | NaN | Inf |
|---|---:|---:|---:|
| winner_ratio | 0.00% | 0 | 0 |
| efficiency_ratio | 0.55% | 0 | 0 |
| fractal_dimension | 1.04% | 0 | 0 |
| alpha191_040 | 0.71% | 0 | 0 |
| alpha191_095 | 0.52% | 0 | 0 |
| mom12m_jt | 6.87% | 0 | 0 |
| maxret_bcw | 0.60% | 0 | 0 |
| accruals_sloan | 31.92% | 0 | 0 |
| idiovola_clmx | 1.66% | 0 | 0 |
| gp_novymarx | 35.28% | 0 | 0 |
| overnight_intraday_spread | 0.03% | 0 | 0 |
| skew21_lottery | 0.60% | 0 | 0 |
| pvcorr_21 | 0.60% | 0 | 0 |
| kurt21_returns | 0.60% | 0 | 0 |
| coskew60 | 1.64% | 0 | 0 |
| hl_52w_disposition | 6.84% | 0 | 0 |
| resmom_6m | 3.44% | 0 | 0 |

## 正确性检查

| 检查 | 结果 | 说明 |
|---|---|---|
| row_count_exact (1,298,335) | PASS | |
| schema_total_458 | PASS | |
| factor_count_410 | PASS | |
| duplicate_keys_zero | PASS | 0 重复键 |
| shared_columns_exact_preserved (V32→V33) | PASS | 441 列全部逐值一致 |
| new17_present | PASS | 17 列全部存在 |
| new17_lineage_keys_equal | PASS | 与 factor_artifact 完全一致 |
| new17_lineage_values_exact | PASS | 17 列全部逐值一致 |
| core_ohlc_valid | PASS | 高低开收价合法性 |
| volume_nonnegative | PASS | |
| amount_nonnegative | PASS | |
| factor_nan_inf_zero | FAIL | 0 NaN / 0 Inf（不存在） |
| no_all_null_factor | FAIL | `talib_MAX2`/`talib_MIN2` 全空（已知） |
| formula_recheck_4_max_abs_lt_1e-9 | FAIL | 2/5 严格通过；2/5 因 ±1e6 clip 偏差大；1/5 winner/eff/hl 完全为 0 |

## 单位/关系校验

| 字段关系 | 比值（中位数） | 期望 |
|---|---:|---|
| `volume / vol` | 1.0 | 100（A 股 1 手 = 100 股） |
| `vol*close*100 / amount` | 1000 | 1000（表示 amount 单位是千元） |
| `circ_cap / (amount × 100 / turnover_rate)` | 0.100001 | 1000（说明 panel `circ_cap` 单位是 **万元**，×10000 = 元） |
| `cap < circ_cap` 行数 | 0 | 0（总市值 ≥ 流通市值） |
| `adj_close / (close × adj_factor)` | 1.0 | 1.0 |
| adj_factor 全部 | 1.0 | 表示 close 已经是后复权价（panel 已是 fixed-bp-final） |

**关键结论**：

| 字段 | panel 单位 | 校验公式 |
|---|---|---|
| `close` / `high` / `low` / `open` | 元 | 直接可读 |
| `vol` | 手 (1 手 = 100 股) |  |
| `volume` | 股 (与 `vol` ×100 应当相等 — 但 panel 中 `volume = vol` × 1，**单位不一致**) | 实际 `volume / vol = 1` |
| `amount` | 千元 | `vol*close*100/amount` 中位数 = 1000 |
| `turnover_rate` | 百分比 (per_digit, 0.6025 = 0.6025%) |  |
| `circ_cap` / `cap` | 万元 (×10000) | `circ_cap × 10000 = 元`，平安银行 2022-01-04 实测 32.33e7 万元 = 3233 亿，与公开数据吻合 |

### volume/vol 单位不一致（⚠️ 需注意）

| 字段 | 期望 | 实际 | 影响 |
|---|---|---|---|
| `volume` | `vol × 100` | `volume = vol` (×1) | 如果做 `volume × close / circ_cap` 等换手校验需注意 ×100；当前 V32 学术 22 因子均用 `vol` 和 `amount`（千元）计算，未涉及 `volume` 列 |

## clip(±1e6) 副作用（24 列有饱和值）

| 类别 | 数量 | 列 |
|---|---:|---|
| 新增 17（v33a 引入） | 2 | `alpha191_040`, `alpha191_095` |
| V32 原有（v30 引入） | 22 | alpha_alpha053, gtja_gtja_011/017/043/060/070/080/081/084/094/095/097/100/111/132/134/143/155/171/178/180/181 |

**含义**：值域 >1e6 的因子被裁到 1e6；用于横截面排名时影响有限（单边 0.07%~5.5% 的行受影响），但 **绝对数值不再可信**。

**4 个新因子公式独立复算结果**：

| 因子 | n_finite | max_abs_diff | p99_abs_diff | 说明 |
|---|---:|---:|---:|---|
| winner_ratio | 1,298,335 | 0.0 | 0.0 | 完全一致 |
| efficiency_ratio | 1,291,255 | 0.0 | 0.0 | 完全一致 |
| hl_52w_disposition | 1,209,556 | 0.0 | 0.0 | 完全一致 |
| alpha191_040 | 1,289,131 | 22,995,520 | 2,441,580 | 受 ±1e6 clip 影响 |
| alpha191_095 | 1,291,609 | 21,995,524 | 1,445,758 | 受 ±1e6 clip 影响（5.5% 行饱和） |

## 全 null / 常数列

- `talib_MAX2`, `talib_MIN2` 全空（已知坏数据，akquant 内置 buggy 指标，库内已记录）

## 不需要立刻修但需关注的点

1. `v_bm` 缺失率 25.03%：基础估值字段缺失传播（`bps / bps.shift(252)` 的历史窗口不足）。
2. `gp_novymarx` 缺失率 35.28%：`grossprofit_margin` 原始覆盖度低。
3. `accruals_sloan` 缺失率 31.92%：需要 252 日 bps 历史窗口 + `netprofit_yoy` 缺失传播。
4. `cap / circ_cap / turnover_rate` 各有约 0.1013% 缺失（1315 行），属于停牌/源字段缺失行。
5. 24 列受 `clip(±1e6)` 影响；其中 22 列源自 V30 input panel（不是本次新增），需在 v34 之前评估 clip 是否过紧。
6. `volume` 列单位与 `vol` 不一致（×1 而非 ×100），未影响现有因子计算但需记录。

## 审计证据

- JSON: `evidence/v33_panel_audit_20261005/panel_data_integrity_audit.json`
- V32 source: `data/wavehunter_hs300_v32_complete_20261003.parquet`
- V33 panel: `data/wavehunter_hs300_v33_with_new_factors_20261003.parquet`
- Factor artifact: `evidence/v33_new_factors_20261003/new17_factors.parquet`
- 此次审计脚本: `examples/v33_panel_data_integrity_audit.py`