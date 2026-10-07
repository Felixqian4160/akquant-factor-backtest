# V31 面板重构 — 自检报告

**日期**: 2026-09-25
**运行**: v2 (修复版)
**脚本**: `examples/v31_fix_panel_v2.py`

---

## 产物

| 文件 | 大小 | 内容 |
|---|---|---|
| `data/wavehunter_hs300_v31_refactored_v2_20260925.parquet` | 2.65 GB | **主面板**: 420 列 = 2 keys + 6 OHLCV + 390 原始 stockpick + 22 新学术因子 |
| `data/v10_1_zig_labels_v31_20260925.parquet` | 1.6 MB | 标签层 (11 列 v10_1_zig, 物理隔离) |
| `data/idx_timing_v31_20260925.parquet` | 3.5 MB | 择时层 (4 列 idx + idx_ret_1d) |

**旧版 (废弃)**: `data/wavehunter_hs300_v31_refactored_20260925.parquet` (v1) — 有 2 个 bug, 不再使用。

---

## 自检发现的问题 (v1 → v2 修复)

### Bug 1: OHLCV 列丢失
v1 输出时只 join 了 stockpick 因子, open/high/low/close/vol/amount 未包含。
**修复**: v2 join 时加入 OHLCV。

### Bug 2: `max_horizontal(null, x)` 静默填充 null (严重)
polars `pl.max_horizontal(a, b)` 默认 `ignore_nulls=True`, 当 a 为 null 时返回 b。
导致缺失值被静默填充为下限值:
- `l_size`: circ_cap null → log(1) = **0** (污染为"最小市值")
- `v_ep`: pe null → 1/1 = **1.0** (污染为"EP=100%")
- `daily_turnover`, `l_ami`, `p_pr` 同样受影响

**修复**: 全部改为 `pl.when(cond).then(expr).otherwise(None)` null-safe 写法。

---

## v2 验证结果 (全部通过)

### 结构验证
| 检查 | 结果 |
|---|---|
| Shape | (1,298,335, 420) ✅ |
| 列数分解 | 2 + 6 + 390 + 22 = 420 ✅ |
| OHLCV 完整 | ✅ |
| 原始 390 stockpick 全保留 | ✅ (无丢失列) |
| 行数一致 | ✅ (1,298,335) |
| 日期范围 | 2004-01-02 → 2026-08-27 ✅ |
| 股票数 | 354 ✅ |
| v10_1_ / idx_ 泄漏 | ✅ 无 |

### Null 语义验证
| 因子 | null 数 | 来源核对 |
|---|---|---|
| l_size | 1,315 | = circ_cap null 数 (精确一致) ✅ |
| v_bm | 325,034 | = bps null 数 (精确一致) ✅ |
| v_ep | 56,847 | pe 缺失/负值 ✅ |
| l_turnm | 8,395 | 窗口不足 + circ_cap 缺失 ✅ |

### 因子公式全序列验证 (000001.SZ, 独立 pandas 复算对照)
| 因子 | 匹配率 | 最大差异 |
|---|---|---|
| r_tv (60d std) | 5289/5289 | 7e-17 ✅ |
| l_ami (Amihud 21d) | 5328/5328 | 3e-21 ✅ |
| l_turnm (换手 21d) | 5328/5328 | 0 ✅ |
| p_m6 (动量 6m) | 5202/5202 | 4e-16 ✅ |
| v_bm (bps/close) | 3357/3357 | 0 ✅ |

Null 一致性: 5 个因子均 0 不一致 (两库 null 位置完全一致)。

### 范围检查
- p_52w ∈ [0.070, 1.000] ✅ (相对 252d 新高, 界内)
- p_season ∈ {0, 1} ✅

---

## 已知限制 (诚实标注)

1. **v_bm 25% NaN**: 来源 bps 财务数据本身缺失 (季度披露), 非计算 bug。
2. **l_turna 6.8% / p_52w 6.8% / p_mchg 7.4% NaN**: 252d 长窗口要求, 早期行天然缺失。
3. **r_beta 1.6% NaN**: 60d 协方差窗口不足。
4. **r_beta 用 pandas 计算**: polars 无 rolling_cov/rolling_corr, 需 groupby apply (仅此一列)。

---

## 22 个新学术因子 (Quantactix/Liu-Stambaugh-Tian)

| 族 | 因子 |
|---|---|
| Liquidity (8) | l_size, l_size3, l_turnm, l_turna, l_ami, l_dtvm, l_dtva, l_vdtv |
| Risk (2) | r_tv, r_beta |
| Past Returns (10) | p_m1, p_m3, p_m6, p_m11, p_m24, p_mchg, p_52w, p_mdr, p_pr, p_season |
| Value (2) | v_bm, v_ep |

---

## 工程教训 (polars)

1. `pl.max_horizontal(a, b)` 默认 `ignore_nulls=True` — **不要用它当 clip 下限保护**, 会把 null 填成 b。用 `pl.when().then().otherwise(None)`。
2. polars 无 `rolling_cov` / `rolling_corr` (0.20.x) — cov/var 需 pandas groupby 实现。
3. `Expr.clip(lower=)` 不存在 — 必须用 `clip(lower_bound=, upper_bound=)` 或 max/min。
4. 400 列 × 1.3M 行 `to_pandas()` 会 OOM — 因子计算只用 polars, 最后 join 也在 polars 侧完成。
5. `pl.read_parquet(columns=[...])` 投影读取是控内存关键。
