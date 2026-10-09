# 其余历史选择器 — 结算滞后（未来函数）筛查

- 日期: 2026-10-10 · 背景: v34-lb20 前视审计（`REPORT.md`）后的横向普查
- 规则: 决策日 i 只能使用"结算日 ≤ i"的评估期；一期 = T+1 入场 + 20 日持有 = **滞后 21 个交易日**

## 裁决总表

| # | 选择器 | 实现文件 | 打分计算 | 结算合规 | 裁决 |
|---|---|---|---|---|---|
| 1 | v34-lb20 因子收益排名（主线） | build_picks_v34_factorrank.py / stage5 / stage_a | top10−bot10 net20，最近 20 行 | ❌ 未按结算日切（混入未来 20 天） | **FAIL**（已量化：65.07%→4.32%；82.57%→−0.31%） |
| 2 | v33 因子收益排名族 | build_picks_v33_factorrank*.py | 同一函数族（`mask = pos < i; [-lookback:]`） | ❌ 同一缺陷 | **FAIL**（结构相同，数字作废） |
| 3 | V14 IC 投票 | build_picks_v33_ic.py / v37_v14_picks.py（IC 段） | 60d 滚动 Spearman IC，`end = count_le − CAUSAL_LAG` | ✅ 正确滞后 21 | **PASS**（可用；v33 IC ≈ +2.2%/年、v37-v3 ≈ +4.6%） |
| 4 | V19 熊市因子收益投票（独立版） | bear_factor_return_vote_v19.py | 已完成期 top/bot10 均值 | ✅ `exit_date < current` 只取已完成 | **PASS**（可用） |
| 5 | v37 熊市缓存（v14_causal 版） | v37_build_bear_cache.py | 覆盖率累计均值（非 top−bot spread） | ⚠️ 轻微滞后 + **指标退化** | **弃用**（覆盖伪影，非有效信号） |
| 6 | v27balance directional 缓存 | build_bear_directional_cache_v27balance.py | high/low 各 top10 均值；`exit_pos[d] < cur_pos` | ✅ 只取已完成期 | **PASS**（修正版实现） |
| 7 | daily factor reselect（v14） | daily_factor_pipeline_v14.py | 60d 滚动 IC，消费端无结算滞后 | ❌ 当日 rolling_ic 含未来 ~20 天 IC | **FAIL**（同族各版本口径不一，需逐一复核） |
| 8 | v32 voting（std × (1−null)） | v32 系列 | 静态代理，无收益打分 | — | N/A（无此缺陷；但本身不是收益指标） |

## 关键证据（代码行）

- #1/#2：`mask = date_pos_arr < i; window = diffs_arr[mask][-lookback:]`（build_picks_v3*_factorrank.py ~L113-116）——行号在前、结算在后，未平移 21。
- #3：`end_exclusive = count_le - CAUSAL_LAG`（build_picks_v33_ic.py:154 / v37_v14_picks.py:196）✅
- #4：`filter(trade_date < current AND exit_date < current)`（bear_factor_return_vote_v19.py:314-317）✅
- #5：`cum_mean of _fwd_net where factor non-null`（v37_build_bear_cache.py:140-146）；实测 2024-01-15：407 因子分值 std=0.0005，top3 = gp_novymarx / v_bm / accruals_sloan（数据覆盖伪影，非业绩）⚠️
- #6：`eligible = [.. if exit_pos[d] < cur_pos]`（build_bear_directional_cache_v27balance.py:229）✅
- #7：`compute_rolling_ic` 无滞后；`top_factors_per_day` 直接使用当日 `_rolling_ic`（daily_factor_pipeline_v14.py:124-187, 273-276）❌

## v37 主结果说明

- `evidence/v37_v14_akquant/FINAL_REPORT.md`：总收益 **+502.42%** / Sharpe 0.524 / MDD −62.35% / PF 1.434（数字量级真实，无异常泡沫）。
- 选股主段（IC）已确认因果；熊市段视所用缓存版本而定（#5 退化 / #6 合规）——复用该线时熊市段须统一使用 #6 口径。

## 结论（"还有没有能用的信号"）

1. **因果合规、可继续研究**：V14 IC 系（真实但弱）、V19（熊市专用）、v27balance directional（熊市修正版）。
2. **与 v34-lb20 同缺陷、数字作废**：v33/v34 因子收益排名族、daily reselect v14。
3. **应弃用**：v37_v14_causal 熊市缓存（退化指标）。
4. **治理规则**：今后所有"因子业绩打分"一律按结算日规则（滞后 = 1 + 持有期）；回归测试 = "窗口最新一行结算日必须 ≤ 决策日"。
