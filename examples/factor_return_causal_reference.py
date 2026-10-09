"""因果版「因子收益排名投票」打分 — 参考实现（settle-lag 修正版）。

唯一规则
--------
一期评估的结果要 21 个交易日后才结算（1 天 T+1 入场 + 20 天持有期），
所以决策日 i（收盘后决策、i+1 开盘执行）只能使用 t ≤ i−21 的期：

  矩阵行  diff[f,t] = top10(f@t) − bot10(f@t) 的 net20（区间 [t+1, t+21]）平均差
  因果分数 score[f,i] = nanmean( diff[f, i−41 : i−21] )    # 20 期，全部已结算
  原实现(错) nanmean( diff[f, i−20 : i] )                  # 每行结算于 行号+21 → 混入未来 20 天

防御性工程做法：把矩阵整体下移 21 行存储（行号 = 结算日），
消费端保持"最近 20 行"的直觉写法也不会再犯。

用法
----
  python3.12 -u examples/factor_return_causal_reference.py                # as-of 最后数据日
  python3.12 -u examples/factor_return_causal_reference.py --asof 2024-06-28

只读演示：打印可用评估期窗口、top-10 因子、以及该日投票选股结果（口径演示，非交易建议）。
注意：修正后该策略经 A/B 验证无 alpha（见 evidence/audit_lookahead_20261010/REPORT.md）。
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"

SETTLE = 21       # 1 (T+1 entry) + 20 (holding) → 一期评估滞后 21 个交易日结算
LOOKBACK = 20
TOP_K = 10
K_NOM = 10
MIN_VOTES_BULL = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="决策日 YYYY-MM-DD（默认=数据最后一天）")
    args = ap.parse_args()

    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy()
    dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    i = dates.index(args.asof) if args.asof else len(dates) - 1

    print(f"决策日 i = {dates[i]} (行号 {i}, 数据区间 {dates[0]}..{dates[-1]})")
    print()

    # ── 可用（已结算）窗口 ──
    hi, lo = i - SETTLE, max(0, i - SETTLE - LOOKBACK)
    win_dates = dates[lo:hi]
    print(f"[因果窗口] 最近 {len(win_dates)} 期、全部已结算: {win_dates[0]} .. {win_dates[-1]}")
    print(f"  最新一期的结算日 = {dates[(hi - 1) + SETTLE]}  (≤ 决策日 ✓)")
    print()

    # ── 原实现窗口（禁用）：展示其依赖未来数据 ──
    bad_lo, bad_hi = max(0, i - LOOKBACK), i
    sd_idx = (bad_hi - 1) + SETTLE
    sd_str = dates[sd_idx] if sd_idx < len(dates) else f"(需 i+{SETTLE}，超出数据范围)"
    print(f"[原实现窗口 · 已禁用] {dates[bad_lo]} .. {dates[bad_hi - 1]}")
    print(f"  其中最新一期的结算日 = {sd_str} → 依赖未来 {sd_idx - i} 个交易日的价格 ✗")
    print()

    # ── 因果分数 ──
    win = M[lo:hi]
    with np.errstate(all="ignore"):
        counts = np.isfinite(win).sum(axis=0)
        scores = np.where(counts >= 10, np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
    ranked = sorted([(fi, scores[fi]) for fi in range(len(factors)) if np.isfinite(scores[fi])],
                    key=lambda kv: -kv[1])
    print(f"[top-{TOP_K} 因子 · 因果分数 = 已结算 {len(win_dates)} 期均值]")
    for k, (fi, s) in enumerate(ranked[:TOP_K], 1):
        print(f"  {k:2d}. {factors[fi]:<30s} score={s:+.4f}")
    print()

    # ── 投票选股（用 f@i，数据 ≤ i，因果） ──
    active = [factors[fi] for fi, _ in ranked[:TOP_K]]
    day = pl.read_parquet(V34, columns=["trade_date", "ts_code"] + active).filter(
        pl.col("trade_date").dt.strftime("%Y-%m-%d") == dates[i]
    )
    if day.height == 0:
        print("(该日无面板数据，跳过投票)")
        return
    codes = day["ts_code"].to_list()
    values = day.select(active).to_numpy()
    votes = np.zeros(day.height, dtype=np.int32)
    for j in range(len(active)):
        col = values[:, j]
        valid = np.flatnonzero(np.isfinite(col))
        if len(valid) < K_NOM:
            continue
        votes[valid[np.argpartition(col[valid], -K_NOM)[-K_NOM:]]] += 1
    order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
    sel = [k for k in order if int(votes[k]) >= MIN_VOTES_BULL]
    if len(sel) < 5:
        sel = order[:5]
    sel = sel[:10]
    print(f"[若该日调仓 → 投票选股（min_votes={MIN_VOTES_BULL}，bull 默认阈值）]")
    for k in sel:
        print(f"  {codes[k]}  votes={int(votes[k])}")
    print()
    print("注：修正后该策略经单变量 A/B 验证无 alpha（65.07%→4.32% / 82.57%→−0.31%）；")
    print("    本输出仅演示因果口径的运转方式，非交易建议。")


if __name__ == "__main__":
    main()
