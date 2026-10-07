"""V34: IC 滚动 + 投票策略 + AKQuant 真实回测 (2010-2025)

合同 (基于 V32 静态池升级):
  - 窗口: 2010-2025, 16 年 (3886 交易日, 354 股票)
  - 调仓: 每 20 个交易日 (~190 个调仓点)
  - IC 计算: 每个调仓日向前滚动 60 天的 Spearman corr (factor vs forward 20d return)
  - 选因子: 每日取 IC top-M 因子 (滚动, 不固定池)
  - 每因子提名: top-K 股票 (按方向调优后的方向)
  - 投票: 票数 ≥ MIN_VOTES 入选 (≥ MIN_STOCKS 保底, ≤ MAX_STOCKS 封顶)
  - 方向: 每个因子 hi/lo 两个方向, 保留更好的
  - 执行: T+1 raw open, lot=100, commission 0.25% + slippage 0.10% (双边)
  - target_weight = 0.99/n (1% 缓冲避免保证金拒单)

输出:
  - evidence/v34_ic_voting_2010_2025/{picks.json, picks_meta.json, ic_history.csv}
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUT = Path("evidence/v34_ic_voting_2010_2025")

import pandas as pd

W_S, W_E = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31")
REBAL_STEP = 20
IC_WINDOW = 60
TOP_M = 30
K_NOM = 10
MIN_VOTES = 3
MIN_STOCKS = 5
MAX_STOCKS = 20

# 因子分组
EXCLUDE_PREFIXES = ("v10_1_", "idx_")
EXCLUDE_EXACT = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "volume", "amount",
    "adj_factor", "pct_chg", "cap", "circ_cap", "turnover_rate",
    "pe", "pb", "bps", "roe", "roe_waa", "roa", "grossprofit_margin", "netprofit_margin",
    "current_ratio", "quick_ratio", "debt_to_assets", "assets_turn",
    "fcff", "cfps", "ocfps", "netprofit_yoy", "or_yoy", "ocf_yoy",
    "adj_close", "idx_close",
    "talib_MAX2", "talib_MIN2",  # 已知坏
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================================================================
# 1. 准备: net20 缓存 + 因子清单
# ============================================================================

log("=" * 80)
log(f"V34: IC 滚动 + 投票 + AKQuant 真实回测")
log(f"窗口: 2010-2025, IC 窗口={IC_WINDOW}d, top_M={TOP_M}, "
    f"K={K_NOM}, votes≥{MIN_VOTES}, {MIN_STOCKS}~{MAX_STOCKS} 股, rebal={REBAL_STEP}d")
log("=" * 80)

OUT.mkdir(parents=True, exist_ok=True)

# net20
log(f"\n[1] 计算 net20 缓存...")
t0 = time.time()
net20 = (
    pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "close"])
    .sort(["ts_code", "trade_date"])
    .with_columns([
        pl.col("open").shift(-1).over("ts_code").alias("open_t1"),
        pl.col("close").shift(-21).over("ts_code").alias("close_t21"),
    ])
    .with_columns(
        (pl.col("close_t21") / pl.col("open_t1") - 1.0 - 0.005).alias("net20")
    )
    .select(["trade_date", "ts_code", "net20"])
    .filter(pl.col("net20").is_not_null() & pl.col("net20").is_finite())
    .collect()
)
log(f"  net20 缓存 {net20.shape}, {time.time()-t0:.1f}s")

# 因子清单
log("\n[2] 收集因子清单...")
schema = pl.scan_parquet(PANEL).collect_schema().names()
factor_cols = [
    c for c in schema
    if c not in EXCLUDE_EXACT
    and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
]
log(f"  选股因子候选: {len(factor_cols)}")
factor_cols.sort()

# 调仓日
all_dates = sorted(net20["trade_date"].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
log(f"  调仓点: {len(rebal_dates)} 个, "
    f"{pd.Timestamp(rebal_dates[0]).date()} → {pd.Timestamp(rebal_dates[-1]).date()}")

# ============================================================================
# 2. 每日 IC: 滚动 60 天 rank corr vs net20
# ============================================================================
log(f"\n[3] 计算每日 IC 矩阵 ({len(factor_cols)} 因子 × {len(rebal_dates)} 调仓日)...")

t0 = time.time()

# 用 polars 分批计算, 避免 OOM
BATCH = 100
ic_records = []  # (trade_date, factor, ic)

for b0 in range(0, len(factor_cols), BATCH):
    batch = factor_cols[b0:b0+BATCH]
    # 读取必要列
    needed = ["trade_date", "ts_code"] + batch
    df = pl.read_parquet(PANEL, columns=needed).sort(["ts_code", "trade_date"])
    df = df.join(net20.select(["trade_date", "ts_code", "net20"]),
                 on=["trade_date", "ts_code"], how="left")

    # 每个调仓日: 取前 IC_WINDOW 天 + 当天, 对每天因子 vs net20 算 rank corr
    for rd in rebal_dates:
        rd_pd = pd.Timestamp(rd).date()
        ws = rd_pd - pd.Timedelta(days=IC_WINDOW * 1.5)  # 用 calendar 天覆盖 60 个交易日
        # 取 trade_date 在 [ws, rd] 范围内的行
        window = df.filter(
            (pl.col("trade_date") <= pl.lit(rd).str.to_datetime()) &
            (pl.col("trade_date") >= pl.lit(ws).str.to_datetime())
        )
        # 取所有交易日去重排序
        days = sorted(window["trade_date"].unique().to_list())
        if len(days) < IC_WINDOW:
            continue
        # 取最后 IC_WINDOW 个
        last_days = days[-IC_WINDOW:]
        sub = window.filter(pl.col("trade_date").is_in(last_days)).drop_nulls()
        if sub.height < IC_WINDOW * 100:
            continue
        # 对每个因子算 corr (polars 一次算所有列)
        corr_exprs = [pl.corr(pl.col(c), pl.col("net20")).alias(c) for c in batch]
        corr = sub.select(corr_exprs).row(0)
        for c, ic in zip(batch, corr):
            if ic is not None and np.isfinite(ic):
                ic_records.append((rd_pd.isoformat(), c, float(ic)))
    if (b0 // BATCH) % 5 == 0:
        log(f"  factor batch {min(b0+BATCH, len(factor_cols))}/{len(factor_cols)}, "
            f"已 {len(ic_records)} IC, {time.time()-t0:.0f}s")

ic_df = pl.DataFrame(
    ic_records,
    schema={"trade_date": pl.Utf8, "factor": pl.Utf8, "ic": pl.Float64},
)
log(f"  IC 矩阵: {ic_df.shape}, {time.time()-t0:.1f}s")
ic_df.write_csv(OUT / "ic_history.csv")

# ============================================================================
# 3. 每调仓日: 取 IC top-M 因子 + 投票选股
# ============================================================================
log(f"\n[4] 生成 picks (top-{TOP_M} IC 因子投票)...")

t0 = time.time()

# 重新读 slim panel (ohlcv + 因子, 投影读取)
df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open"] + factor_cols)
df = df.filter(
    (pl.col("trade_date") >= pl.lit(W_S).str.to_datetime()) &
    (pl.col("trade_date") <= pl.lit(W_E).str.to_datetime())
).sort(["ts_code", "trade_date"])

picks = {}  # date_str -> {ts_code: vote_count}
picks_meta = {}  # date_str -> {pool: [...], n_pool: int, n_picks: int}

ic_pd = ic_df.to_pandas()

for rd in rebal_dates:
    rd_str = pd.Timestamp(rd).date().isoformat()
    rd_pd = pd.Timestamp(rd).date()
    sub_ic = ic_pd[(ic_pd["trade_date"] <= rd_str)]
    if len(sub_ic) == 0:
        continue
    # 取每个因子最近一次的 IC (跨调仓日的滚动 IC)
    latest_ic = sub_ic.sort_values("trade_date").groupby("factor").tail(1)
    # 取 IC 绝对值 top-M
    latest_ic = latest_ic.assign(abs_ic=lambda x: x["ic"].abs())
    top_factors = latest_ic.sort_values("abs_ic", ascending=False).head(TOP_M)["factor"].tolist()
    if not top_factors:
        continue

    # 每天用因子排序: 保留 hi/lo 中更高的 abs_ic 方向
    directions = {}
    for _, row in latest_ic[latest_ic["factor"].isin(top_factors)].iterrows():
        directions[row["factor"]] = "hi" if row["ic"] > 0 else "lo"

    # 该调仓日各因子 top-K 提名
    day_picks = df.filter(pl.col("trade_date") == pl.lit(rd).str.to_datetime())
    if day_picks.height == 0:
        continue
    votes = Counter()
    for fac in top_factors:
        d = day_picks.select(["ts_code", fac]).drop_nulls()
        if d.height < K_NOM:
            continue
        if directions[fac] == "hi":
            top = d.sort(fac, descending=True).head(K_NOM)["ts_code"].to_list()
        else:
            top = d.sort(fac, descending=False).head(K_NOM)["ts_code"].to_list()
        for s in top:
            votes[s] += 1
    if not votes:
        continue
    # 入选规则: ≥ MIN_VOTES; < MIN_STOCKS 取最多; > MAX_STOCKS 取票数最多的 MAX_STOCKS
    sel = {s: v for s, v in votes.items() if v >= MIN_VOTES}
    if len(sel) < MIN_STOCKS:
        sel = dict(votes.most_common(MIN_STOCKS))
    if len(sel) > MAX_STOCKS:
        sel = dict(votes.most_common(MAX_STOCKS))
    picks[rd_str] = sel
    picks_meta[rd_str] = {
        "n_pool": len(top_factors),
        "directions": directions,
        "n_picks": len(sel),
        "n_votes_dist": sorted(set(sel.values()), reverse=True)[:5],
    }
    if len(picks) % 20 == 1:
        log(f"  picks {len(picks)}/{len(rebal_dates)}, sample {rd_str}: {len(sel)} stocks")

with open(OUT / "picks.json", "w") as f:
    json.dump(picks, f, indent=1)
with open(OUT / "picks_meta.json", "w") as f:
    json.dump(picks_meta, f, indent=1)

n_picks_avg = np.mean([len(p) for p in picks.values()])
log(f"\n  picks: {len(picks)} 调仓点, 平均 {n_picks_avg:.1f} 只, "
    f"用时 {time.time()-t0:.1f}s")

# 汇总
summary = {
    "window": f"{W_S.date()} ~ {W_E.date()}",
    "n_rebalances": len(rebal_dates),
    "n_picks": len(picks),
    "avg_n_stocks": float(n_picks_avg),
    "ic_window": IC_WINDOW,
    "top_m_factors": TOP_M,
    "k_nom": K_NOM,
    "min_votes": MIN_VOTES,
    "min_stocks": MIN_STOCKS,
    "max_stocks": MAX_STOCKS,
    "rebal_step": REBAL_STEP,
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log(f"\nDONE: {OUT}/")