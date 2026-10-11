"""v4 五因子集合选择 — 2010-2017 因果口径 (用户合同 2026-10-11).

目标: 在 413 个 dedup 因子中挑选 5 因子集合, 用于"5 因子共振投票"策略测试。
纪律: 选择统计只用 2010-2017 (矩阵差分序列, 已结算口径); 2018+ 为后续 OOS。

集合定义:
  s1_top5_t      : t 值最高 5 (纯收益强度)
  s1d_top5_t_div : t 排序 + 贪心相关性约束 (|corr|<0.5, 不够放宽 0.6/0.7)
  s2_consist     : (pos_years, t) 排序 + 贪心 corr<0.6 (持续稳定性)
  s3_family      : alpha/gtja/talib/其他 各组 t 最佳 + 剩余最佳 1 个
  s4_control     : 随机 5 (seed=20261011) — null 对照

输出: evidence/v4_five_factor_20261011/{factorsets.json, selection.md, run_factsets/<set>.json}
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
OUT = AKQ / "evidence" / "v4_five_factor_20261011"
FS = OUT / "run_factsets"
W0, W1 = "2010-01-01", "2017-12-31"
SEED = 20261011


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FS.mkdir(parents=True, exist_ok=True)
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    dates = np.array([str(d)[:10] for d in mat["trade_date"].to_list()])
    sel = (dates >= W0) & (dates <= W1)
    X = mat.select(factors).to_numpy().astype(np.float64)[sel]
    yrs = np.array([d[:4] for d in dates[sel]])
    log(f"matrix rows {len(dates)}, window rows {int(sel.sum())}, factors {len(factors)}")

    n = np.isfinite(X).sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    t = np.where(n > 2, mean / (std / np.sqrt(np.maximum(n, 1)) + 1e-12), np.nan)
    py = np.zeros(len(factors), dtype=int)
    for y in sorted(set(yrs.tolist())):
        m = np.nanmean(X[yrs == y], axis=0)
        py += (np.isfinite(m) & (m > 0)).astype(int)
    valid = n > 500
    idx = {f: i for i, f in enumerate(factors)}
    log(f"valid factors: {int(valid.sum())}")

    df = pd.DataFrame(X, columns=factors)
    corr = df.corr(min_periods=100).abs().to_numpy()
    log("corr matrix done")

    def greedy(ranked, th):
        picked = []
        for f in ranked:
            if not valid[idx[f]] or f in picked:
                continue
            if all(corr[idx[f], idx[g]] < th for g in picked):
                picked.append(f)
            if len(picked) == 5:
                break
        return picked

    order_t = sorted([f for f in factors if valid[idx[f]]],
                     key=lambda f: -(t[idx[f]] if np.isfinite(t[idx[f]]) else -1e9))
    s1 = order_t[:5]
    s1d = greedy(order_t, 0.5)
    if len(s1d) < 5:
        s1d = greedy(order_t, 0.6)
    if len(s1d) < 5:
        s1d = greedy(order_t, 0.7)

    order_c = sorted([f for f in factors if valid[idx[f]]],
                     key=lambda f: (-py[idx[f]], -(t[idx[f]] if np.isfinite(t[idx[f]]) else -1e9)))
    s2 = greedy(order_c, 0.6)
    if len(s2) < 5:
        s2 = greedy(order_c, 0.7)

    sup = {}
    for f in [f for f in factors if valid[idx[f]]]:
        p = f.split("_")[0]
        k = p if p in ("alpha", "gtja", "talib") else "other"
        sup.setdefault(k, []).append(f)
    best = {k: max(v, key=lambda f: t[idx[f]] if np.isfinite(t[idx[f]]) else -1e9)
            for k, v in sup.items()}
    s3 = list(best.values())[:4]
    for f in order_t:
        if len(s3) >= 5:
            break
        if f in s3:
            continue
        if all(corr[idx[f], idx[g]] < 0.8 for g in s3):
            s3.append(f)

    rng = np.random.default_rng(SEED)
    s4 = list(rng.choice(order_t, size=5, replace=False))

    def set_stats(fs):
        ts = [t[idx[f]] for f in fs]
        c_max = max((corr[idx[a], idx[b]] for i, a in enumerate(fs) for b in fs[i + 1:]), default=0.0)
        return {"mean_t": float(np.mean(ts)), "min_t": float(np.min(ts)),
                "sum_py": int(sum(py[idx[f]] for f in fs)), "max_corr": float(c_max),
                "per_factor": {f: {"t": round(float(t[idx[f]]), 2), "mean": round(float(mean[idx[f]]), 5),
                                   "py": int(py[idx[f]])} for f in fs}}

    sets = {
        "s1_top5_t": {"reason": "2010-2017 t 最高 5 因子", "factors": s1},
        "s1d_top5_t_div": {"reason": "t 排序 + 贪心 |corr| 约束", "factors": s1d},
        "s2_consist": {"reason": "(pos_years,t) 排序 + 贪心 corr<0.6", "factors": s2},
        "s3_family": {"reason": "alpha/gtja/talib/其他 各组 t 最佳 + 剩余最佳", "factors": s3},
        "s4_control": {"reason": f"随机 5 (seed={SEED}) null 对照", "factors": s4},
    }
    for k, v in sets.items():
        v["stats"] = set_stats(v["factors"])
        log(f"{k}: {v['factors']}")
        log(f"    mean_t={v['stats']['mean_t']:.1f} min_t={v['stats']['min_t']:.1f} "
            f"sum_py={v['stats']['sum_py']}/40 max_corr={v['stats']['max_corr']:.2f}")

    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "window": f"{W0}..{W1}",
               "source": str(MAT), "sets": sets}
    (OUT / "factorsets.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    for k, v in sets.items():
        (FS / f"{k}.json").write_text(json.dumps(
            {"tag": k, "factors": v["factors"], "reason": v["reason"]}, ensure_ascii=False, indent=1))

    md = ["# v4 五因子集合选择 (2010-2017 因果口径)", "",
          f"生成: {payload['generated']} | 窗口: {W0}..{W1} | 源: {MAT.name}", ""]
    for k, v in sets.items():
        md += [f"## {k}", f"- 理由: {v['reason']}",
               f"- mean_t={v['stats']['mean_t']:.1f} min_t={v['stats']['min_t']:.1f} "
               f"max_corr={v['stats']['max_corr']:.2f}", "",
               "| factor | t | mean | pos_years |", "|---|---|---|---|"]
        for f in v["factors"]:
            s = v["stats"]["per_factor"][f]
            md.append(f"| {f} | {s['t']} | {s['mean']} | {s['py']} |")
        md.append("")
    (OUT / "selection.md").write_text("\n".join(md))
    log(f"saved -> {OUT/'factorsets.json'} + run_factsets/*.json + selection.md")


if __name__ == "__main__":
    main()
