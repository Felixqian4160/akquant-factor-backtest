"""全因子因果记分卡（v34 428 factors）— 描述性筛查（结算合规口径）。

对每个因子，在 2010-2025 上统计其 top10-bot10 net20 价差（矩阵 diff）：
  - 全期均值 / std / 描述性 t 值
  - 分年正收益年数（16 年里有多少年 mean>0）
  - 牛/熊态拆分（router_map）
  - 截至最后一个数据日、已结算 20 期窗口的近期得分（用于对照）
注意：
  - diff = 多空价差（top10 − bot10）；long-only 实际使用需后续补 top 侧单边统计
  - 重叠窗口 → t 值仅作描述性参考，不是推断统计
  - 本表用于"选什么因子、怎么用"的筛查（先看数据），不是 alpha 验收

输出: evidence/audit_lookahead_20261010/factor_scorecard_causal.csv (+ stdout 摘要)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
ROUTER = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = AKQ / "evidence" / "audit_lookahead_20261010" / "factor_scorecard_causal.csv"

mat = pl.read_parquet(MAT)
factors = [c for c in mat.columns if c != "trade_date"]
M = mat.select(factors).to_numpy().astype(np.float64)
dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
years = np.array([int(d[:4]) for d in dates])
router = json.loads(ROUTER.read_text())
regime = np.array([router.get(d, "bull_neutral") for d in dates])

sel = np.array([("2010-01-01" <= d <= "2025-12-31") for d in dates])
Mw = M[sel]
yrs = years[sel]
reg = regime[sel]

n = np.isfinite(Mw).sum(axis=0)
with np.errstate(all="ignore"):
    mean = np.nanmean(Mw, axis=0)
    std = np.nanstd(Mw, axis=0)
tstat = np.where(n > 0, mean / (std / np.sqrt(np.maximum(n, 1)) + 1e-12), np.nan)

year_list = sorted(set(yrs.tolist()))
pos_years = np.zeros(len(factors), dtype=int)
for y in year_list:
    m = np.nanmean(Mw[yrs == y], axis=0)
    pos_years += (np.isfinite(m) & (m > 0)).astype(int)

bull = reg == "bull_neutral"
with np.errstate(all="ignore"):
    mean_bull = np.nanmean(Mw[bull], axis=0)
    mean_bear = np.nanmean(Mw[~bull], axis=0)

iL = len(dates) - 1
recent = np.nanmean(M[iL - 41:iL - 21], axis=0)

out = pl.DataFrame({
    "factor": factors, "n": n, "mean": mean, "std": std, "t": tstat,
    "pos_years": pos_years, "mean_bull": mean_bull, "mean_bear": mean_bear,
    "recent20_settled": recent,
}).sort("mean", descending=True)
out.write_csv(OUT)

vals = mean[np.isfinite(mean)]
print(f"factors={len(factors)}  valid={len(vals)}")
print(f"mean of means={vals.mean():+.4f}  median={np.median(vals):+.4f}  std={vals.std():.4f}")
print(f"positive-mean factors: {int((vals > 0).sum())} / {len(vals)}")
for th in (0.005, 0.01, 0.02):
    print(f"  mean > {th:.3f}: {int((vals > th).sum())}")
print(f"pos_years >= 12/16: {int((pos_years >= 12).sum())}   >= 10/16: {int((pos_years >= 10).sum())}")
print(f"both bull>0 and bear>0: {int(((mean_bull > 0) & (mean_bear > 0)).sum())}")
print()
print(f"{'factor':<30s} {'mean':>8s} {'t':>6s} {'posY':>5s} {'bull':>8s} {'bear':>8s} {'recent':>8s}")
for r in out.head(30).iter_rows(named=True):
    print(f"{r['factor']:<30s} {r['mean']:+8.4f} {r['t']:>6.1f} {int(r['pos_years']):>4d} "
          f"{r['mean_bull']:+8.4f} {r['mean_bear']:+8.4f} {r['recent20_settled']:+8.4f}")
print()
print("bottom 10:")
for r in out.tail(10).iter_rows(named=True):
    print(f"{r['factor']:<30s} {r['mean']:+8.4f} {r['t']:>6.1f} {int(r['pos_years']):>4d} "
          f"{r['mean_bull']:+8.4f} {r['mean_bear']:+8.4f}")
