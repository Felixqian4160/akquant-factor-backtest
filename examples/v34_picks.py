"""V34 picks 生成: 全部 412 因子 + 滚动 IC + 投票 (参数化)

合同:
  - 窗口: 2010-2025
  - 调仓: 每 20 个交易日
  - IC: 滚动 60 天 Spearman corr (factor vs net20)
  - 选因子: ALL 412 因子 (不筛) + 方向 hi/lo 按当前 IC
  - 投票: MIN_VOTES 阈值, MIN_STOCKS/MAX_STOCKS (CLI)
  - 输出: picks.json, picks_meta.json, ic_history.csv, summary.json

CLI: --min-votes N --min-stocks N --max-stocks N --out-tag TAG
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUT_BASE = Path("evidence/v34_ic_voting_2010_2025")

W_S, W_E = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31")
REBAL_STEP = 20
IC_WINDOW = 60
K_NOM = 10

EXCLUDE_PREFIXES = ("v10_1_", "idx_")
EXCLUDE_EXACT = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "volume", "amount",
    "adj_factor", "pct_chg", "cap", "circ_cap", "turnover_rate",
    "pe", "pb", "bps", "roe", "roe_waa", "roa", "grossprofit_margin", "netprofit_margin",
    "current_ratio", "quick_ratio", "debt_to_assets", "assets_turn",
    "fcff", "cfps", "ocfps", "netprofit_yoy", "or_yoy", "ocf_yoy",
    "adj_close", "idx_close",
    "talib_MAX2", "talib_MIN2",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--min-stocks", type=int, default=10)
    ap.add_argument("--max-stocks", type=int, default=30)
    ap.add_argument("--out-tag", default="A")
    args = ap.parse_args()

    out = OUT_BASE / args.out_tag
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"V34 picks [{args.out_tag}]: 全 412 因子 + 滚动 IC + 投票")
    log(f"MIN_VOTES={args.min_votes}, {args.min_stocks}~{args.max_stocks} 股, "
        f"IC 窗口={IC_WINDOW}d, rebal={REBAL_STEP}d")
    log("=" * 80)

    # 1. net20
    log("\n[1] net20...")
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
    )
    log(f"  net20 {net20.shape}, {time.time()-t0:.1f}s")

    # 2. 因子清单
    log("\n[2] 因子清单...")
    schema = pl.scan_parquet(PANEL).collect_schema().names()
    factor_cols = [
        c for c in schema
        if c not in EXCLUDE_EXACT and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
    ]
    factor_cols.sort()
    log(f"  因子候选: {len(factor_cols)}")

    # 3. 调仓日
    all_dates = sorted(net20["trade_date"].unique().to_list())
    rebal_dates = all_dates[::REBAL_STEP]
    log(f"  调仓点: {len(rebal_dates)} ({pd.Timestamp(rebal_dates[0]).date()} → "
        f"{pd.Timestamp(rebal_dates[-1]).date()})")

    # 4. 每日 IC (滚动 60 天)
    log(f"\n[3] 每日 IC 矩阵 ({len(factor_cols)} 因子 × {len(rebal_dates)} 调仓日)...")
    t0 = time.time()

    BATCH = 100
    ic_records = []  # (date_str, factor, ic)

    # 预算每个调仓日的 IC_WINDOW calendar 范围
    win_days = int(IC_WINDOW * 1.6)  # ~96 calendar days

    for b0 in range(0, len(factor_cols), BATCH):
        batch = factor_cols[b0:b0+BATCH]
        df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + batch)
        df = df.join(net20, on=["trade_date", "ts_code"], how="left")

        for rd in rebal_dates:
            rd_ts = pd.Timestamp(rd).to_pydatetime()
            ws_dt = pd.Timestamp(rd).date() - pd.Timedelta(days=win_days)
            ws_ts = pd.Timestamp(ws_dt).to_pydatetime()
            sub = df.filter(
                (pl.col("trade_date") <= rd_ts) &
                (pl.col("trade_date") >= ws_ts) &
                pl.col("net20").is_not_null()
            ).select(["trade_date"] + batch + ["net20"])
            # 取最后 IC_WINDOW 个不同日期
            days = sub["trade_date"].unique().to_list()
            if len(days) < IC_WINDOW:
                continue
            last_days = days[-IC_WINDOW:]
            # cast str 给 is_in 避开 datetime 精度差
            last_days_str = [d.strftime("%Y-%m-%d") for d in last_days]
            sub2 = sub.filter(
                pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(last_days_str)
            ).drop_nulls()
            if sub2.height < IC_WINDOW * 100:
                continue
            # 每个因子 vs net20 的 Spearman 近似: rank corr
            corr_exprs = [pl.corr(pl.col(c), pl.col("net20")).alias(c) for c in batch]
            corr = sub2.select(corr_exprs).row(0)
            rd_str = pd.Timestamp(rd).date().isoformat()
            for c, ic in zip(batch, corr):
                if ic is not None and np.isfinite(ic):
                    ic_records.append((rd_str, c, float(ic)))
        if (b0 // BATCH) % 10 == 0:
            log(f"  batch {min(b0+BATCH, len(factor_cols))}/{len(factor_cols)}, "
                f"{len(ic_records)} IC, {time.time()-t0:.0f}s")

    ic_df = pl.DataFrame(
        ic_records,
        schema={"trade_date": pl.Utf8, "factor": pl.Utf8, "ic": pl.Float64},
    )
    log(f"  IC 矩阵: {ic_df.shape}, {time.time()-t0:.1f}s")
    ic_df.write_csv(out / "ic_history.csv")

    # 5. 投票 picks
    log(f"\n[4] 生成 picks (全部因子投票, 票数 ≥ {args.min_votes})...")
    t0 = time.time()

    df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + factor_cols)
    df = df.filter(
        (pl.col("trade_date") >= pl.lit(W_S.to_pydatetime())) &
        (pl.col("trade_date") <= pl.lit(W_E.to_pydatetime()))
    ).sort(["ts_code", "trade_date"])

    picks = {}        # date_str -> {ts_code: vote_count}
    picks_meta = {}   # date_str -> {n_votes_max, n_picks, n_above_threshold}

    ic_pd = ic_df.to_pandas()
    ic_idx = ic_pd.set_index(["trade_date", "factor"])["ic"]  # MultiIndex

    for rd in rebal_dates:
        rd_str = pd.Timestamp(rd).date().isoformat()
        # 取 ≤ rd 的最后一次 (跨调仓日的滚动 IC)
        sub_ic = ic_pd[ic_pd["trade_date"] <= rd_str]
        if sub_ic.length == 0:
            continue
        latest = sub_ic.sort_values("trade_date").groupby("factor").tail(1)
        latest = latest.assign(abs_ic=lambda x: x["ic"].abs())
        # 全部因子参与 (top_M=None)
        all_fac = latest.sort_values("abs_ic", ascending=False)["factor"].tolist()
        # 方向按当前 IC 符号
        directions = dict(zip(latest["factor"], latest["ic"].apply(lambda x: "hi" if x > 0 else "lo")))

        day_picks = df.filter(pl.col("trade_date") == pl.lit(rd))
        if day_picks.height == 0:
            continue
        votes = Counter()
        # 用 lazy filter 减少内存压力
        day_pd = day_picks.select(["ts_code"] + all_fac).to_pandas()
        for fac in all_fac:
            d = day_pd[["ts_code", fac]].dropna()
            if len(d) < K_NOM:
                continue
            if directions[fac] == "hi":
                top = d.nlargest(K_NOM, fac)["ts_code"].tolist()
            else:
                top = d.nsmallest(K_NOM, fac)["ts_code"].tolist()
            for s in top:
                votes[s] += 1
        if not votes:
            continue
        sel = {s: v for s, v in votes.items() if v >= args.min_votes}
        if len(sel) < args.min_stocks:
            sel = dict(votes.most_common(args.min_stocks))
        if len(sel) > args.max_stocks:
            sel = dict(votes.most_common(args.max_stocks))
        picks[rd_str] = sel
        picks_meta[rd_str] = {
            "n_factors": len(all_fac),
            "n_picks": len(sel),
            "max_votes": max(votes.values()),
            "n_at_or_above_threshold": sum(1 for v in votes.values() if v >= args.min_votes),
            "n_total_votes_observed": len(votes),
        }
        if len(picks) % 20 == 1:
            log(f"  picks {len(picks)}/{len(rebal_dates)}: {rd_str} "
                f"n={len(sel)}, max_votes={picks_meta[rd_str]['max_votes']}")

    with open(out / "picks.json", "w") as f:
        json.dump(picks, f, indent=1)
    with open(out / "picks_meta.json", "w") as f:
        json.dump(picks_meta, f, indent=1)

    summary = {
        "window": f"{W_S.date()} ~ {W_E.date()}",
        "n_rebalances": len(rebal_dates),
        "n_picks": len(picks),
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0,
        "ic_window": IC_WINDOW,
        "all_factors": True,
        "n_factors": len(factor_cols),
        "k_nom": K_NOM,
        "min_votes": args.min_votes,
        "min_stocks": args.min_stocks,
        "max_stocks": args.max_stocks,
        "rebal_step": REBAL_STEP,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\n  picks: {len(picks)} 调仓点, "
        f"平均 {summary['avg_n_stocks']:.1f} 只, "
        f"用时 {time.time()-t0:.1f}s")
    log(f"DONE → {out}/")


if __name__ == "__main__":
    main()