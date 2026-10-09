"""因子去重（横截面 rank 相关聚类）— 投票公平性前置步骤。

方法:
  - 抽样 100 个交易日（2010-2025），逐日计算 428 因子的横截面 Spearman 相关
    （rank 后 Pearson，pairwise 完整样本，min_periods=100）
  - 跨日平均相关矩阵；按 |ρ| ≥ 0.95 做连通分量聚类（同时报告 0.90）
  - 每个重复组保留 |scorecard 全期均值| 最大的因子作为"唯一代表"（1 组 = 1 票）

输出: evidence/audit_lookahead_20261010/factor_dedup_groups.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
SCORECARD = AKQ / "evidence" / "audit_lookahead_20261010" / "factor_scorecard_causal.csv"
OUT = AKQ / "evidence" / "audit_lookahead_20261010" / "factor_dedup_groups.json"

mat = pl.read_parquet(MAT)
factors = [c for c in mat.columns if c != "trade_date"]
dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
F = len(factors)

sel = [i for i, d in enumerate(dates) if "2010-01-01" <= d <= "2025-12-31"]
take = [sel[int(x)] for x in np.linspace(0, len(sel) - 1, 100)]
sub_dates = [dates[i] for i in take]
print(f"factors={F}; sample dates={len(sub_dates)} ({sub_dates[0]}..{sub_dates[-1]})")

rdt = pl.Series("trade_date", [pd.Timestamp(d).to_pydatetime() for d in sub_dates]).dt.cast_time_unit("ms")
day = (
    pl.read_parquet(V34, columns=["trade_date", "ts_code"] + factors)
    .filter(pl.col("trade_date").is_in(rdt))
    .with_columns(pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("d"))
)
pdf = day.select(["d"] + factors).to_pandas()
print(f"panel subset: {pdf.shape}")

acc = np.zeros((F, F))
wgt = np.zeros((F, F))
nd = 0
for _, grp in pdf.groupby("d", sort=True):
    R = grp[factors].rank(method="average", na_option="keep")
    C = R.corr(min_periods=100).to_numpy()
    m = np.isfinite(C)
    acc[m] += C[m]
    wgt[m] += 1
    nd += 1
meanC = acc / np.maximum(wgt, 1)
print(f"mean correlation matrix done over {nd} dates")


def components(thr: float):
    iu = np.triu_indices(F, 1)
    vals = np.abs(meanC[iu])
    mask = vals >= thr
    pairs = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    parent = list(range(F))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(F):
        groups.setdefault(find(i), []).append(i)
    multi = [sorted(g) for g in groups.values() if len(g) > 1]
    multi.sort(key=lambda g: -len(g))
    return multi, len(pairs)


# scorecard for representative choice
sc = pd.read_csv(SCORECARD).set_index("factor")
sc_abs = sc["mean"].abs().to_dict()


def pick_rep(members_idx):
    names = [factors[i] for i in members_idx]
    best = max(names, key=lambda n: sc_abs.get(n, 0.0) or 0.0)
    return best


out = {"n_factors": F, "thresholds": {}}
for thr in (0.95, 0.90):
    multi, n_pairs = components(thr)
    groups_out = []
    rep_set = set()
    for g in multi:
        names = [factors[i] for i in g]
        rep = pick_rep(g)
        rep_set.add(rep)
        groups_out.append({"members": sorted(names), "representative": rep, "size": len(g)})
    singleton_rep = [f for f in factors if f not in {m for g in groups_out for m in g["members"]}]
    reps = sorted(rep_set | set(singleton_rep))
    out["thresholds"][str(thr)] = {
        "n_edges": n_pairs,
        "n_groups_multi": len(groups_out),
        "n_merged_factors": sum(len(g["members"]) - 1 for g in groups_out),
        "n_representatives": len(reps),
        "groups": groups_out,
        "representatives": reps,
    }
    print(f"|rho|>={thr}: edges={n_pairs} multi-groups={len(groups_out)} "
          f"merged={sum(len(g['members'])-1 for g in groups_out)} -> representatives={len(reps)}")
    if thr == 0.95:
        print("\nlargest groups (0.95):")
        for g in groups_out[:15]:
            print(f"  [{g['size']}] rep={g['representative']:<28s} members={g['members'][:8]}{' ...' if g['size']>8 else ''}")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"\nsaved {OUT}")
