"""对指定 picks 应用双均线条件过滤 (用户假设 2026-10-12).

条件 (全部因果, 只用 <= T 的数据):
  D: 日线 MA20 > MA120 (adj_close, T 日收盘)
  W: 周线 wMA5 > wMA10 > wMA20 多头排列 (最后完成周, 三线开花)

fstyle:
  entry: 保留原 survivors (picks[k]∩picks[k-1]), 只过滤新增买入  (纯入场择时)
  full : 每个调仓日全部重新过滤 (不满足即剔除 = 加"破位卖出")

输出: picks/<SRC>_MA_<MODE>_<FSTYLE>/ (picks.json + picks_meta.json)
用法: filter_picks_ma.py --src V5_P413_RET_RANK_o0 --mode {d,w,dw} --fstyle {entry,full}
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
PICKS = AKQ / "evidence" / "v3_resonance_20261010" / "picks"


def build_conditions():
    df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close"])
    dates = [str(d)[:10] for d in df["trade_date"].unique().sort().to_list()]
    codes = df["ts_code"].unique().sort().to_list()
    tvals = df["trade_date"].to_numpy().astype("datetime64[ms]")
    date_ms = np.array([np.datetime64(d) for d in dates]).astype("datetime64[ms]")
    ti = np.searchsorted(date_ms, tvals)
    si = pd.Categorical(df["ts_code"].to_numpy(), categories=codes).codes
    CL = np.full((len(dates), len(codes)), np.nan)
    CL[ti, si] = df["adj_close"].to_numpy()
    dfp = pd.DataFrame(CL, index=pd.to_datetime(dates), columns=codes)
    # --- daily: MA20 > MA120 ---
    ma20 = dfp.rolling(20).mean()
    ma120 = dfp.rolling(120).mean()
    cond_d = np.where(np.isfinite(ma20.to_numpy()) & np.isfinite(ma120.to_numpy()),
                      (ma20 > ma120).to_numpy(), False)
    # 刚上穿 (金叉): 状态为真 且 最近 10 日内曾为假
    st = cond_d
    crossed = np.zeros_like(st)
    for k in range(1, 11):
        prev = np.zeros_like(st)
        prev[k:] = st[:-k]
        crossed |= np.logical_and(st, np.logical_not(prev))
    # 前 k 行预热不足时 prev=False 会误判: 前 120 行状态均为 False, 只在 k>=1 时检查即可 (预热段无真值)
    cond_dc = crossed
    # --- weekly: wMA5 > wMA10 > wMA20 (最后完成周) ---
    iso = dfp.index.isocalendar()
    wk = np.array([f"{y}-{w:02d}" for y, w in zip(iso["year"], iso["week"])])
    last_idx = []
    wcl = []
    cur = None
    buf = None
    for i, w in enumerate(wk):
        if w != cur:
            if cur is not None:
                last_idx.append(i - 1)
                wcl.append(buf)
            cur = w
            buf = CL[i]
        else:
            buf = CL[i]
    last_idx.append(len(wk) - 1)
    wcl.append(buf)
    W = pd.DataFrame(np.array(wcl), columns=codes)
    wma5 = W.rolling(5).mean()
    wma10 = W.rolling(10).mean()
    wma20 = W.rolling(20).mean()
    cond_w_week = np.where(np.isfinite(wma5.to_numpy()) & np.isfinite(wma20.to_numpy()),
                           ((wma5 > wma10) & (wma10 > wma20)).to_numpy(), False)
    last_idx = np.array(last_idx)
    wi = np.searchsorted(last_idx, np.arange(len(dates)), side="right") - 1
    cond_w = np.zeros_like(cond_d)
    ok = wi >= 0
    cond_w[ok] = cond_w_week[wi[ok]]
    return dates, codes, cond_d, cond_w, (cond_d & cond_w), cond_dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="V5_P413_RET_RANK_o0")
    ap.add_argument("--mode", choices=["d", "w", "dw", "dc", "dcw", "rand"], default="dw")
    ap.add_argument("--fstyle", choices=["entry", "full"], default="entry")
    ap.add_argument("--count-src", default=None, help="rand 模式: 从该 tag 的每日持仓数取样")
    args = ap.parse_args()
    dates, codes, cond_d, cond_w, cond_dw, cond_dc = build_conditions()
    cidx = {c: i for i, c in enumerate(codes)}
    didx = {d: i for i, d in enumerate(dates)}
    picks = json.loads((PICKS / args.src / "picks.json").read_text())
    if args.mode == "rand":
        cnt_src = json.loads((PICKS / (args.count_src or "") / "picks.json").read_text())
        rng = np.random.default_rng(20261012)
        out = {}
        per_day = {}
        for d in sorted(picks):
            orig = picks[d]
            items = list(orig.items())
            n = min(len(cnt_src.get(d, {})), len(items))
            if n >= len(items):
                keep = dict(items)
            else:
                idx = rng.choice(len(items), size=n, replace=False)
                keep = {items[i][0]: items[i][1] for i in idx}
            out[d] = keep
            per_day[d] = {"n_elig": len(orig), "n_picks": len(keep)}
        tag = f"{args.src}_RAND_CTRL"
    else:
        cond = {"d": cond_d, "w": cond_w, "dw": cond_dw, "dc": cond_dc,
                "dcw": (cond_dc & cond_w)}[args.mode]
        out = {}
        prev = set()
        per_day = {}
        for d in sorted(picks):
            orig = picks[d]
            di = didx[d]
            if args.fstyle == "full":
                keep = {s: v for s, v in orig.items() if cond[di, cidx[s]]}
            else:
                surv = {s: v for s, v in orig.items() if s in prev}
                new = {s: v for s, v in orig.items() if s not in prev}
                keep = {**surv, **{s: v for s, v in new.items() if cond[di, cidx[s]]}}
            out[d] = keep
            per_day[d] = {"n_elig": len(orig), "n_picks": len(keep)}
            prev = set(keep)
        tag = f"{args.src}_MA_{args.mode.upper()}_{args.fstyle.upper()}"
    od = PICKS / tag
    od.mkdir(parents=True, exist_ok=True)
    (od / "picks.json").write_text(json.dumps(out))
    (od / "picks_meta.json").write_text(json.dumps({
        "contract": {"src": args.src, "filter_mode": args.mode, "fstyle": args.fstyle,
                     "daily_cond": "MA20>MA120 (adj_close)",
                     "weekly_cond": "wMA5>wMA10>wMA20 (last completed week)"},
        "per_day": per_day}, ensure_ascii=False))
    n0 = sum(len(v) for v in picks.values())
    n1 = sum(len(v) for v in out.values())
    print(f"{tag}: {len(out)} days | holdings {n0} -> {n1} ({n1 / n0:.1%}) | "
          f"avg/day {n1 / len(out):.1f} (orig {n0 / len(picks):.1f})")


if __name__ == "__main__":
    main()
