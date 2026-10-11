# v4 五因子集合选择 (2010-2017 因果口径)

生成: 2026-10-11 20:51:02 | 窗口: 2010-01-01..2017-12-31 | 源: matrix_v34_dedup_ADJ.parquet

## s1_top5_t
- 理由: 2010-2017 t 最高 5 因子
- mean_t=12.2 min_t=9.6 max_corr=0.59

| factor | t | mean | pos_years |
|---|---|---|---|
| l_ami | 15.14 | 0.04219 | 6 |
| alpha_alpha040 | 14.07 | 0.02286 | 8 |
| gtja_gtja_113 | 11.72 | 0.01846 | 7 |
| hl_52w_disposition | 10.58 | 0.0232 | 7 |
| alpha_alpha006 | 9.57 | 0.01509 | 7 |

## s1d_top5_t_div
- 理由: t 排序 + 贪心 |corr| 约束
- mean_t=12.2 min_t=9.6 max_corr=0.47

| factor | t | mean | pos_years |
|---|---|---|---|
| l_ami | 15.14 | 0.04219 | 6 |
| alpha_alpha040 | 14.07 | 0.02286 | 8 |
| gtja_gtja_113 | 11.72 | 0.01846 | 7 |
| hl_52w_disposition | 10.58 | 0.0232 | 7 |
| gtja_gtja_015 | 9.55 | 0.02044 | 7 |

## s2_consist
- 理由: (pos_years,t) 排序 + 贪心 corr<0.6
- mean_t=8.8 min_t=5.8 max_corr=0.35

| factor | t | mean | pos_years |
|---|---|---|---|
| alpha_alpha040 | 14.07 | 0.02286 | 8 |
| alpha_alpha045 | 8.59 | 0.01217 | 8 |
| alpha_alpha003 | 8.08 | 0.01057 | 8 |
| alpha_alpha055 | 7.53 | 0.00974 | 8 |
| alpha_alpha013 | 5.77 | 0.00904 | 8 |

## s3_family
- 理由: alpha/gtja/talib/其他 各组 t 最佳 + 剩余最佳
- mean_t=11.6 min_t=6.3 max_corr=0.47

| factor | t | mean | pos_years |
|---|---|---|---|
| l_ami | 15.14 | 0.04219 | 6 |
| alpha_alpha040 | 14.07 | 0.02286 | 8 |
| gtja_gtja_113 | 11.72 | 0.01846 | 7 |
| talib_ADX | 6.26 | 0.01222 | 6 |
| hl_52w_disposition | 10.58 | 0.0232 | 7 |

## s4_control
- 理由: 随机 5 (seed=20261011) null 对照
- mean_t=2.1 min_t=-0.5 max_corr=0.51

| factor | t | mean | pos_years |
|---|---|---|---|
| p_m11 | 8.54 | 0.01689 | 7 |
| alpha_alpha_custom_argmax_recent | -0.35 | -0.00063 | 4 |
| gtja_gtja_025 | 1.04 | 0.00218 | 4 |
| gtja_gtja_129 | -0.5 | -0.00078 | 4 |
| talib_LINEARREG_SLOPE | 1.66 | 0.00284 | 4 |
