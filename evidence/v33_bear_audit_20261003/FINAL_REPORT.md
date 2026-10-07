# V33 GitHub 新因子移植 — 最终报告

**日期**: 2026-10-03
**任务**: 从 GitHub 搜索开源新因子 → 移植 → 熊市 (2022-2024) 验证
**产出**: V33 panel (458 cols) + 17 因子熊市 audit + 去重分析

---

## 1. 搜索来源 (4 方向, 2 成功)

| 方向 | 状态 | 关键发现 |
|---|---|---|
| A 股项目 | ✅ | wukan1986/ta_cn (筹码/分形), Daic115/alpha191, popbo/alphas |
| 微观结构 | ❌ (aicodee 额度) | — |
| 学术因子 | ✅ | OpenSourceAP/CrossSection 等; 13 个 polars 实现 |
| 熊市防御 | ❌ (aicodee 额度) | — |

## 2. V33 Panel 集成 (458 cols)

- V32 (441) + 17 新因子 = **V33 (458 cols)**, 1,298,335 行
- 计算: v33a slim 模式 (11 列输入, 1.7s) + v33b join 组装 (10s)
- 自检: pandas 复算 4 因子 max rel err ≤ 4.58e-13 ✅

## 3. 熊市 Audit (2022-2024, k=20, 37 期)

| 因子 | 方向 | ret20 | 2022 | 2023 | 2024 | 去重结论 |
|---|---|---|---|---|---|---|
| alpha191_095 | lo | **+58.4%** | +16.9% | +6.9% | +26.7% | ❌ 重复 (gtja_095, corr=1.000) |
| idiovola_clmx | hi | **+45.0%** | -18.3% | +9.2% | +62.5% | ⚠️ 冗余 (talib_NATR, +0.845) |
| maxret_bcw | hi | +34.9% | -22.3% | +24.5% | +39.4% | ⚠️ 部分 (gtja_159, +0.711) |
| efficiency_ratio | hi | +32.5% | -5.6% | +9.4% | +28.3% | 🔶 半新 (talib_DX, +0.554) |
| gp_novymarx | lo | +15.9% | +4.3% | -10.6% | +24.3% | ✅ 新维度 (+0.443) |
| mom12m_jt | hi | +14.5% | -7.5% | -15.7% | +46.9% | ✅ 新维度 (+0.311) |
| kurt21_returns | hi | +11.0% | — | — | — | ❌ 冗余 (kurt_filter, -0.880) |
| skew21_lottery | lo | +1.6% | — | — | — | ❌ 冗余 (skew_reversal, -0.902) |
| coskew60 | lo | -1.4% | — | — | — | ✅ 新维度 |
| overnight_intraday | hi | -3.7% | — | — | — | ❌ 冗余 (alpha_033, -0.920) |
| pvcorr_21 | lo | -3.9% | — | — | — | 🔶 半新 |
| alpha191_040 | hi | -9.6% | — | — | — | ⚠️ 部分 (gtja_052, +0.760) |
| accruals_sloan | hi | -10.4% | — | — | — | ✅ 新维度 |
| winner_ratio | hi | -11.3% | — | — | — | ⚠️ 冗余 (gtja_079, +0.926) |
| resmom_6m | hi | -26.3% | — | — | — | ✅ 新维度 |
| hl_52w_disposition | hi | -30.9% | — | — | — | ⚠️ 部分 (gtja_067, +0.661) |
| fractal_dimension | hi | -32.6% | — | — | — | ✅ 新维度 |

**基准**: 等权 -13.42% | **v31 全库 412 mean**: +1.56% | **17 新因子 mean**: +4.92% (8/17 正)

## 4. 去重分析 (全窗口 104 天均匀采样)

- **1 个精确重复**: alpha191_095 = gtja_095 (corr 1.000, 收益完全相同 +58.36%)
- **4 个高冗余** (|corr| > 0.8): winner_ratio, idiovola, overnight_intraday, skew21, kurt21
- **4 个部分重叠** (0.6-0.8): alpha191_040, maxret_bcw, hl_52w
- **8 个新维度** (< 0.6): efficiency, fractal, mom12m, accruals, gp_novymarx, pvcorr, coskew, resmom

## 5. 关键诚实结论

1. **表面最好的 4 个"新因子"全是已有维度的变体**:
   - alpha191_095 就是 gtja_095 (v31 排名 #12)
   - idiovola ≈ NATR (v31 #15, +50.2%), maxret ≈ gtja_159 (v31 #23, +31.1%)
   - efficiency ≈ DX (v31 #1, +95.8%) 半重叠
2. **真新维度在熊市窗口大多表现弱** (fractal/accruals/resmom 为负)
3. **仅 2-3 个候选有部分新意 + 正收益**: efficiency_ratio (0.554), gp_novymarx (0.443), mom12m_jt (0.311)
4. **本轮未发现实质性新熊市 alpha** — GitHub 开源因子大多已覆盖在 GTJA191/Alpha101 库中

## 6. 技术教训

1. **移植因子前必须做去重**: 新因子先与现有库全窗口 corr 检查, |corr|>0.95 直接丢弃
2. **hermes background worker scope 有 ~4GB cgroup 内存限制**: 宽表 (>4GB) 任务必须 foreground 跑或 slim-compute + join 组装
3. GTJA191 因子集已覆盖大部分公开公式 — GitHub 搜索的边际价值递减

## 7. 文件清单

```
data/wavehunter_hs300_v33_with_new_factors_20261003.parquet  (458 cols, 2.8GB)
examples/v33a_compute_new_factors.py       (slim 计算)
examples/v33b_assemble_panel.py            (join 组装)
examples/v33_bear_audit_new_factors.py     (熊市 audit)
evidence/v33_new_factors_20261003/         (v33a 报告 + new17_factors.parquet)
evidence/v33_bear_audit_20261003/          (audit + dedup + combined)
```
