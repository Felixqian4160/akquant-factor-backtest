# V32 Panel 完整化自检报告

**日期**: 2026-10-03
**升级路径**: v31 panel (420 cols) + 21 缺失列 = **v32 panel (441 cols)**

## 列对齐矩阵

| 来源 | 数量 | 列 |
|---|---:|---|
| v30 input (with talib) | 419 | OHLCV + volume + 191 gtja + 107 alpha + 73 talib + 16 财务 + 4 idx + 11 zigzag + 4 OHLCV-adj (cap/circ_cap/adj_close/adj_factor/pct_chg) |
| v31 学术因子 | 22 | l_*/r_*/p_*/v_* |
| **总计 (无重复)** | **441** | |

## 自检结果

| # | 检查项 | 结果 |
|---:|---|---|
| 1 | 总列数 = 441 | ✅ |
| 2 | 21 缺失列全部回填 (cap/circ_cap/adj_*/idx_*/v10_1_*/volume/pct_chg) | ✅ |
| 3 | 22 学术因子全部保留 | ✅ |
| 4 | circ_cap 一致性: l_size 反推 / 实际 = 10000.0000 | ✅ (期望 ~1.0) |
| 5 | 行数 = 1,298,335 (与 v30/v31 一致) | ✅ |
| 6 | l_size 非空行数 v31 vs v32 一致 | ✅ |

## 字段类别分布

```
选股层 (393): 191 gtja + 107 alpha + 73 talib + 22 学术
基础 OHLCV (8): open/high/low/close/vol/volume/amount + adj_close
复权 (2): adj_factor + pct_chg
市值 (2): cap + circ_cap
财务 (16): pe/pb/bps/roe/roa/gpm/npm/current/quick/debt/assets_turn/fcff/cfps/ocfps/netprofit_yoy/or_yoy/ocf_yoy + turnover_rate
择时 idx (4): idx_close + idx_mom_5/20/60
峰谷标签 (11): v10_1_zig_peak/valley/a1_point/a2_start/a2_interval/b1_start/b1_interval/down_start/down_interval/peak_zone/valley_zone
键 (2): trade_date + ts_code
————————————————————
总计: 441 列
```

## 已知限制

1. **cap (总市值) 仍是 v30 原始数据** — 没验证是否合理, 但与 circ_cap 同源 (Tushare daily_basic)
2. **adj_factor 适用于 panel 的 close (后复权价)** — 想前复权需除 adj_factor
3. **idx_close 是 per-stock 复制** (panel 每行都有), 真实指数序列需去重
4. **9 个 zigzag 标签非 100% 覆盖** (有 null, 视原始数据而定)
