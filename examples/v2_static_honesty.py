"""v2 static 诚实化 + 慢动态（年度重选）— v2 快筛后续。

arms:
  static_2017    : 因子集合 = 2010-2017 已结算行均值 top-20（只用 ≤2017 数据），
                   全期固定使用（2010-2017 段为 in-sample，判分看 2018-2025）
  static_roll250 : 每 250 个交易日重选一次集合（expanding，只用已结算行 t ≤ i-21）

其余合同与 v2 快筛完全一致：top-20 因子、score≥0→top-10 / <0→bottom-10 提名、
min_votes bull2/bear4、股票 5-10、结算合规、v41 执行。

用法:
  --build           构建两路 picks (offset 0)
  --run N           跑最多 N 个缺失 sim
  --segments        计算 2018-2025 分段指标（对快筛全部臂 + 本两路）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = AKQ / "evidence" / "v2_screen_20261010"
PICKS = OUT / "picks"
SIMS = OUT / "sims"

START, END, STEP, OFFSET = "2010-01-01", "2025-12-31", 20, 0
SETTLE = 21
TOP_K_FACTORS = 20
K_NOM = 10
MV_BULL, MS_BULL, MX_BULL = 2, 5, 10
MV_BEAR, MS_BEAR, MX_BEAR = 4, 5, 10
RESEL = 250
NEW_ARMS = ["static_2017", "static_roll250"]
SEG_ARMS = ["static", "k20", "k60", "k120", "k250", "ic", "static_2017", "static_roll250"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_matrix():
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy().astype(np.float64)
    dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    return factors, M, dates


def votes_block(day, set_names, dirs, mv, ms, mx, fpos):
    codes = day["ts_code"].to_list()
    values = day.select(set_names).to_numpy()
    votes = np.zeros(day.height, dtype=np.int32)
    for j, fname in enumerate(set_names):
        col = values[:, j]
        valid = np.flatnonzero(np.isfinite(col))
        if len(valid) < K_NOM:
            continue
        if dirs[fname] >= 0:
            chosen = valid[np.argpartition(col[valid], -K_NOM)[-K_NOM:]]
        else:
            chosen = valid[np.argpartition(col[valid], K_NOM - 1)[:K_NOM]]
        votes[chosen] += 1
    order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
    sel = [k for k in order if int(votes[k]) >= mv]
    if len(sel) < ms:
        sel = order[:ms]
    sel = sel[:mx]
    return [{str(codes[k]): int(votes[k])} for k in sel]


def build():
    factors, M, dates = load_matrix()
    dpos = {d: i for i, d in enumerate(dates)}
    fpos = {f: j for j, f in enumerate(factors)}
    rebal = [d for d in dates[OFFSET::STEP] if START <= d <= END]
    log(f"matrix {M.shape}; rebal {len(rebal)}")

    # ---- sets ----
    i2018 = next(i for i, d in enumerate(dates) if d >= "2018-01-01")
    i2010 = next(i for i, d in enumerate(dates) if d >= "2010-01-01")
    rows17 = M[i2010:max(i2010, i2018 - SETTLE)]  # settled-by-2018 & >=2010
    with np.errstate(all="ignore"):
        means17 = np.nanmean(rows17, axis=0)
    order17 = np.argsort(-means17)
    set17 = [factors[k] for k in order17[:TOP_K_FACTORS]]
    dirs17 = {f: (1 if means17[fpos[f]] >= 0 else -1) for f in set17}
    log(f"[static_2017] set top6: {set17[:6]}")

    roll_cache = {}

    def roll_set(i):
        k = max(1, (i - 20) // RESEL)
        if k not in roll_cache:
            cut = k * RESEL
            with np.errstate(all="ignore"):
                mm = np.nanmean(M[:cut], axis=0)
            s = [factors[j] for j in np.argsort(-mm)[:TOP_K_FACTORS]]
            roll_cache[k] = (s, {f: (1 if mm[fpos[f]] >= 0 else -1) for f in s})
        return roll_cache[k]

    rdt = pl.Series("trade_date", [pd.Timestamp(d).to_pydatetime() for d in rebal]).dt.cast_time_unit("ms")
    day_panels = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + factors).filter(
        pl.col("trade_date").is_in(rdt)
    )
    router = json.loads(ROUTER_MAP.read_text())

    for arm in NEW_ARMS:
        picks, meta = {}, {}
        for rd in rebal:
            i = dpos[rd]
            if arm.startswith("static_2017"):
                s, dr = set17, dirs17
            else:
                s, dr = roll_set(i)
            regime = router.get(rd, "bull_neutral")
            if regime == "bull_neutral":
                mv, ms, mx = MV_BULL, MS_BULL, MX_BULL
            else:
                mv, ms, mx = MV_BEAR, MS_BEAR, MX_BEAR
            day = day_panels.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == rd)
            if day.height == 0:
                continue
            sel = votes_block(day, s, dr, mv, ms, mx, fpos)
            basket = {}
            for d_ in sel:
                basket.update(d_)
            if not basket:
                continue
            picks[rd] = basket
            meta[rd] = {"regime": regime, "selector": f"v2_voting_{arm}", "offset": OFFSET,
                        "n_active_factors": len(s), "n_picks": len(basket)}
        outdir = PICKS / arm / "V14_0"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
        (outdir / "picks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        if arm.startswith("static_roll250"):
            log(f"[static_roll250] n_reselections={len(roll_cache)}")
        log(f"picks[{arm}]: {len(picks)} dates")


def result_path(arm):
    return SIMS / arm / "V14_0" / "result.json"


def run_sims(max_n):
    todo = [a for a in NEW_ARMS if not result_path(a).exists()]
    log(f"missing sims: {todo}")
    ran = 0
    for arm in todo:
        if ran >= max_n:
            break
        t0 = time.time()
        cmd = [sys.executable, "-u", str(RUNNER), "--tag", "V14_0",
               "--picks-base", str(PICKS / arm), "--actions", "on", "--ca-mode", "all",
               "--prices", "raw", "--end", "2025-12-31", "--out-base", str(SIMS / arm)]
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
        ok = result_path(arm).exists()
        log(f"  sim[{arm}]: {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
        if not ok:
            return
        ran += 1


def seg_metrics(nav_path: pathlib.Path, start="2018-01-01", end="2025-12-31"):
    df = pd.read_csv(nav_path)
    df["d"] = pd.to_datetime(df["date"])
    df = df.sort_values("d")
    df = df[df["d"] <= pd.Timestamp(end)]
    pre = df[df["d"] < pd.Timestamp(start)]["value"]
    if len(pre) == 0:
        return None
    base = float(pre.iloc[-1])
    seg = df[df["d"] >= pd.Timestamp(start)]
    if len(seg) < 50:
        return None
    vals = seg["value"].to_numpy(dtype=float) / base
    curve = np.concatenate(([1.0], vals))
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1.0))
    ret = float(vals[-1] - 1.0)
    years = (seg["d"].iloc[-1] - seg["d"].iloc[0]).days / 365.25
    ann = float((1.0 + ret) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    daily = np.diff(vals) / vals[:-1]
    sharpe = float(daily.mean() / daily.std() * np.sqrt(244)) if daily.std() > 0 else 0.0
    return {"seg_ret": ret, "seg_ann": ann, "seg_mdd": mdd, "seg_sharpe": sharpe, "n_days": len(seg)}


def segments():
    rows = []
    for arm in SEG_ARMS:
        nav = SIMS / arm / "V14_0" / "nav.csv"
        if not nav.exists():
            rows.append({"arm": arm, "status": "missing"})
            continue
        m = seg_metrics(nav)
        if m is None:
            rows.append({"arm": arm, "status": "no-segment"})
            continue
        m["arm"] = arm
        rows.append(m)
    print(f"{'arm':<16s} {'seg_ret':>9s} {'seg_ann':>9s} {'seg_mdd':>9s} {'seg_sharpe':>10s}")
    for r in rows:
        if "status" in r:
            print(f"{r['arm']:<16s}  ({r['status']})")
        else:
            print(f"{r['arm']:<16s} {r['seg_ret']:>8.2%} {r['seg_ann']:>8.2%} "
                  f"{r['seg_mdd']:>8.1%} {r['seg_sharpe']:>10.3f}")
    (OUT / "segment_2018_2025.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    log(f"saved {OUT/'segment_2018_2025.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--segments", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--only-static", action="store_true")
    args = ap.parse_args()
    global OFFSET, NEW_ARMS
    OFFSET = int(args.offset)
    arms = ["static_2017"] if args.only_static else ["static_2017", "static_roll250"]
    NEW_ARMS = [a if OFFSET == 0 else f"{a}_o{OFFSET}" for a in arms]
    if args.build:
        build()
    elif args.run is not None:
        run_sims(args.run)
    elif args.segments:
        segments()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
