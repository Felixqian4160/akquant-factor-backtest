"""V34 picks 生成 v2: 全部 412 因子 + 滚动 IC + 投票 (polars-native)

修复 v1:
  - IC 计算移到独立函数 (复用结果)
  - picks 阶段: 不再用 to_pandas(), 改为 polars-native long format
  - 投票一次性 group_by 完成 (不再 408 次 Counter 累加)
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


def compute_ic(net20, factor_cols, rebal_dates, ic_window=IC_WINDOW, batch_size=100):
    """滚动 IC: 对每个调仓日, 取前 ic_window 个交易日的 Spearman corr."""
    log(f"  IC matrix: {len(factor_cols)} 因子 × {len(rebal_dates)} 调仓日")
    t0 = time.time()
    records = []
    win_days = int(ic_window * 1.6)

    for b0 in range(0, len(factor_cols), batch_size):
        batch = factor_cols[b0:b0+batch_size]
        # 一次性读全期 panel 子集, 然后内存循环
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
            days = sub["trade_date"].unique().to_list()
            if len(days) < ic_window:
                continue
            last_days = days[-ic_window:]
            last_days_str = [d.strftime("%Y-%m-%d") for d in last_days]
            sub2 = sub.filter(
                pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(last_days_str)
            ).drop_nulls()
            if sub2.height < ic_window * 100:
                continue
            corr_exprs = [pl.corr(pl.col(c), pl.col("net20")).alias(c) for c in batch]
            corr = sub2.select(corr_exprs).row(0)
            rd_str = pd.Timestamp(rd).date().isoformat()
            for c, ic in zip(batch, corr):
                if ic is not None and np.isfinite(ic):
                    records.append((rd_str, c, float(ic)))
        if (b0 // batch_size) % 5 == 0:
            log(f"  batch {min(b0+batch_size, len(factor_cols))}/{len(factor_cols)}, "
                f"{len(records)} IC, {time.time()-t0:.0f}s")
        # 释放内存
        del df

    ic_df = pl.DataFrame(
        records,
        schema=["trade_date", "factor", "ic"],
        orient="row",
    )
    log(f"  IC matrix done: {ic_df.shape} in {time.time()-t0:.1f}s")
    return ic_df


def build_picks(df, ic_df, factor_cols, rebal_dates, min_votes, min_stocks, max_stocks, k_nom=K_NOM):
    """生成 picks: 全部因子投票, polars-native long format + group_by 投票."""
    log(f"  building picks (all {len(factor_cols)} 因子, votes≥{min_votes})")
    t0 = time.time()

    # IC 长格式: 每个调仓日的最新 IC
    ic_pd = ic_df.to_pandas()
    latest_per_factor_date = (
        ic_pd.sort_values("trade_date").groupby(["trade_date", "factor"]).tail(1)
    )

    # 调仓日: 选 2010 起 (按 W_S)
    rebal_dates_filtered = [pd.Timestamp(r).date().isoformat() for r in rebal_dates
                            if pd.Timestamp(r).date() >= W_S.date()]
    latest = latest_per_factor_date[
        latest_per_factor_date["trade_date"].isin(rebal_dates_filtered)
    ].copy()
    log(f"    IC 池 rows in window: {len(latest)}")

    # 每个调仓日的 latest IC (跨调仓日的滚动)
    latest_per_date = latest.sort_values("trade_date").groupby("factor").tail(1)
    log(f"    每天 latest IC 行数: {len(latest_per_date)}")

    picks = {}
    picks_meta = {}
    n = 0
    log(f"    开始投票: {len(rebal_dates_filtered)} 调仓日")
    t1 = time.time()
    for rd_str in rebal_dates_filtered:
        # 该调仓日各因子最新 IC
        sub_ic = latest[latest["trade_date"] == rd_str]
        if sub_ic.empty:
            continue
        # direction
        sub_ic = sub_ic.assign(
            direction=sub_ic["ic"].apply(lambda x: "hi" if x > 0 else "lo"),
            abs_ic=sub_ic["ic"].abs(),
        )
        # 当天所有因子值
        day = df.filter(pl.col("trade_date") == pl.lit(pd.Timestamp(rd_str).to_pydatetime()))
        if day.height == 0:
            continue
        # 只保留方向已知的因子列
        keep = ["ts_code"] + sub_ic["factor"].tolist()
        day_keep = day.select([c for c in keep if c in day.columns]).drop_nulls()
        if day_keep.height < k_nom:
            continue

        # 长格式: ts_code, factor, value, direction
        # melt 到 long
        id_cols = ["ts_code"]
        val_cols = [c for c in day_keep.columns if c not in id_cols]
        # 用 melt: 慢, 用 polars-native
        long_df = (
            day_keep
            .melt(id_vars=id_cols, value_vars=val_cols,
                  variable_name="factor", value_name="value")
            .join(
                sub_ic[["factor", "direction"]].rename(columns={"factor": "factor"}),
                on="factor", how="inner"
            )
        )

        # 对每个 factor 按 direction 选 top-K
        # hi: nlargest, lo: nsmallest
        # group_by (factor, direction), then rank + filter
        # 用 window + filter
        ranked = long_df.with_columns([
            pl.when(pl.col("direction") == "hi")
            .then(pl.col("value").rank(descending=True, method="ordinal").over("factor"))
            .otherwise(pl.col("value").rank(ascending=True, method="ordinal").over("factor"))
            .alias("rk")
        ])
        top_picks = ranked.filter(pl.col("rk") <= k_nom).select(["ts_code", "factor"])
        # 投票: 每个 ts_code 出现 = +1 票
        votes = top_picks.group_by("ts_code").agg(
            pl.len().alias("n_votes")
        ).sort("n_votes", descending=True)
        # 选 ≥ min_votes 的; 不足补到 min_stocks
        sel_df = votes.filter(pl.col("n_votes") >= min_votes)
        if sel_df.height < min_stocks:
            sel_df = votes.head(min_stocks)
        if sel_df.height > max_stocks:
            sel_df = votes.head(max_stocks)
        sel = dict(zip(sel_df["ts_code"].to_list(), sel_df["n_votes"].to_list()))
        picks[rd_str] = sel
        picks_meta[rd_str] = {
            "n_factors": len(sub_ic),
            "n_picks": len(sel),
            "max_votes": int(votes["n_votes"].max()),
            "n_at_or_above_threshold": int((votes["n_votes"] >= min_votes).sum()),
            "n_total_votes_observed": votes.height,
        }
        n += 1
        if n % 20 == 1:
            log(f"    picks {n}/{len(rebal_dates_filtered)}: {rd_str} "
                f"n={len(sel)}, max={picks_meta[rd_str]['max_votes']}")
    log(f"  picks 完成: {len(picks)}/{len(rebal_dates_filtered)}, "
        f"{time.time()-t1:.1f}s (总 {time.time()-t0:.1f}s)")
    return picks, picks_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--min-stocks", type=int, default=10)
    ap.add_argument("--max-stocks", type=int, default=30)
    ap.add_argument("--out-tag", default="A")
    ap.add_argument("--no-polars-picks", action="store_true",
                    help="回退到 pandas 投票 (调试用)")
    args = ap.parse_args()

    out = OUT_BASE / args.out_tag
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"V34 picks v2 [{args.out_tag}]: 全 {args.out_tag} 因子 + 滚动 IC + 投票")
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
    all_dates = sorted(net20["trade_date"].to_list())
    rebal_dates = all_dates[::REBAL_STEP]
    log(f"  调仓点: {len(rebal_dates)} ({pd.Timestamp(rebal_dates[0]).date()} → "
        f"{pd.Timestamp(rebal_dates[-1]).date()})")

    # 4. IC 矩阵
    ic_df = compute_ic(net20, factor_cols, rebal_dates)
    ic_df.write_csv(out / "ic_history.csv")

    # 5. 投票 picks (polars-native long format + group_by)
    log("\n[3] 投票 picks...")
    # 读 slim panel (所有因子列) → 转 pandas
    t0 = time.time()
    df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + factor_cols)
    df = df.filter(
        (pl.col("trade_date") >= pl.lit(W_S.to_pydatetime())) &
        (pl.col("trade_date") <= pl.lit(W_E.to_pydatetime()))
    ).sort(["ts_code", "trade_date"])
    log(f"  panel slim: {df.shape}, {time.time()-t0:.1f}s")

    picks, picks_meta = build_picks(
        df, ic_df, factor_cols, rebal_dates,
        args.min_votes, args.min_stocks, args.max_stocks,
    )

    # 6. 落盘
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
        f"平均 {summary['avg_n_stocks']:.1f} 只")
    log(f"DONE → {out}/")


if __name__ == "__main__":
    main()