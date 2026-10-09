"""v2 快筛 — 去重矩阵 (413 factors) 上的「动态打分 + top-20 投票」六路对照。

合同 (全部结算合规, 滞后 21 天):
  - 因子池: 去重后 413 个唯一代表 (data/wavehunter_hs300_v34_dedup_20261010.parquet)
  - 打分 (只用结算行 t <= i-21):
      static : 固定一组 20 个因子 (按全期均值 top-20, 前端参考臂——含 in-sample 说明)
      k20/60/120/250 : score[f,i] = nanmean(diff[f, i-20-K : i-20])   (K 个已结算行)
      ic     : score[f,i] = mean of last 60 usable rank-IC rows (每行 IC[d] 结算于 d+21)
  - 选 top-20 因子; 每因子提名: score>=0 → 当日 top-10 股; score<0 → bottom-10 股
  - 投票: votes>=min_votes (bull 2 / bear 4), 股票 5-10 只 (与 causal v1 完全一致)
  - 执行: v41 runner (T+1 NextOpen, 成本, CA), 单 offset=0, 2010-2025

用法:
  --build          构建六路 picks (offset 0)
  --run N          跑最多 N 个缺失 sim (v41)
  --report         汇总对照表
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = AKQ / "evidence" / "v2_screen_20261010"
PICKS = OUT / "picks"
SIMS = OUT / "sims"

START, END, STEP = "2010-01-01", "2025-12-31", 20
OFFSET = 0
SETTLE = 21
TOP_K_FACTORS = 20
K_NOM = 10
MV_BULL, MS_BULL, MX_BULL = 2, 5, 10
MV_BEAR, MS_BEAR, MX_BEAR = 4, 5, 10
ARMS = ["static", "k20", "k60", "k120", "k250", "ic"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_matrix():
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy().astype(np.float64)
    dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    return factors, M, dates


def compute_ic(factors: list[str]) -> np.ndarray:
    """rank-IC per (date, factor): Spearman corr(factor value, net20), settle at d+21."""
    frames = []
    B = 50
    for s in range(0, len(factors), B):
        batch = factors[s:s + B]
        pf = (
            pl.scan_parquet(PANEL)
            .select(["trade_date", "ts_code", "adj_open", "adj_close"] + batch)
            .sort(["ts_code", "trade_date"])
            .with_columns([
                pl.col("adj_open").shift(-1).over("ts_code").alias("_o1"),
                pl.col("adj_close").shift(-SETTLE).over("ts_code").alias("_c21"),
            ])
            .with_columns((pl.col("_c21") / pl.col("_o1") - 1.0 - 0.005).alias("net20"))
            .select(["trade_date"] + batch + ["net20"])
            .collect()
        )
        icw = (
            pf.group_by("trade_date")
            .agg([pl.corr(pl.col(c), pl.col("net20"), method="spearman").alias(c) for c in batch])
            .sort("trade_date")
        )
        frames.append(icw)
        log(f"  IC batch {min(s+B, len(factors))}/{len(factors)}")
    base = frames[0]
    for f in frames[1:]:
        assert f.select("trade_date").equals(base.select("trade_date")), "IC date mismatch"
        base = base.hstack(f.drop("trade_date"))
    return base.select(factors).to_numpy().astype(np.float64)


def derive_picks(arm, factors, M, IC, dates, day_panels, router, static_factors):
    picks, meta = {}, {}
    rebal = [d for d in dates[OFFSET::STEP] if START <= d <= END]
    dpos = {d: i for i, d in enumerate(dates)}
    for rd in rebal:
        i = dpos[rd]
        if arm == "static":
            active_names = static_factors
            dirs = [1] * len(active_names)
        else:
            hi = i - 20  # exclusive -> last included row = i-21 (settled)
            if arm == "ic":
                lo = max(0, hi - 60)
                win = IC[lo:hi]
                need = 30
            else:
                K = int(arm[1:])
                lo = max(0, hi - K)
                win = M[lo:hi]
                need = max(10, K // 2)
            with np.errstate(all="ignore"):
                counts = np.isfinite(win).sum(axis=0)
                scores = np.where(counts >= need,
                                  np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
            scored = [(fi, scores[fi]) for fi in range(len(factors)) if np.isfinite(scores[fi])]
            if not scored:
                continue
            ranked = sorted(scored, key=lambda kv: -kv[1])
            active = ranked[:TOP_K_FACTORS]
            active_names = [factors[fi] for fi, _ in active]
            dirs = [1 if s >= 0 else -1 for _, s in active]
        regime = router.get(rd, "bull_neutral")
        if regime == "bull_neutral":
            mv, ms, mx = MV_BULL, MS_BULL, MX_BULL
        else:
            mv, ms, mx = MV_BEAR, MS_BEAR, MX_BEAR
        day = day_panels.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == rd)
        if day.height == 0:
            continue
        codes = day["ts_code"].to_list()
        values = day.select(active_names).to_numpy()
        votes = np.zeros(day.height, dtype=np.int32)
        for j, fname in enumerate(active_names):
            col = values[:, j]
            valid = np.flatnonzero(np.isfinite(col))
            if len(valid) < K_NOM:
                continue
            if dirs[j] >= 0:
                chosen = valid[np.argpartition(col[valid], -K_NOM)[-K_NOM:]]
            else:
                chosen = valid[np.argpartition(col[valid], K_NOM - 1)[:K_NOM]]
            votes[chosen] += 1
        order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
        sel = [k for k in order if int(votes[k]) >= mv]
        if len(sel) < ms:
            sel = order[:ms]
        sel = sel[:mx]
        if not sel:
            continue
        picks[rd] = {str(codes[k]): int(votes[k]) for k in sel}
        meta[rd] = {"regime": regime, "selector": f"v2_voting_{arm}", "offset": OFFSET,
                    "n_active_factors": len(active_names), "n_picks": len(sel)}
    return picks, meta


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    factors, M, dates = load_matrix()
    log(f"matrix: {M.shape}, factors={len(factors)}")
    # static set: top-20 by full-period mean (reference arm; in-sample note in meta)
    sel_rows = np.array([("2010-01-01" <= d <= "2025-12-31") for d in dates])
    with np.errstate(all="ignore"):
        means = np.nanmean(M[sel_rows], axis=0)
    order = np.argsort(-means)
    static_factors = [factors[k] for k in order[:TOP_K_FACTORS]]
    log(f"static-20: {static_factors[:6]}...")

    IC = compute_ic(factors)
    log(f"IC series: {IC.shape}")

    import pandas as pd
    rebal = [d for d in dates[OFFSET::STEP] if START <= d <= END]
    rdt = pl.Series("trade_date", [pd.Timestamp(d).to_pydatetime() for d in rebal]).dt.cast_time_unit("ms")
    day_panels = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + factors).filter(
        pl.col("trade_date").is_in(rdt)
    )
    router = json.loads(ROUTER_MAP.read_text())

    for arm in ARMS:
        pk, mt = derive_picks(arm, factors, M, IC, dates, day_panels, router, static_factors)
        outdir = PICKS / arm / "V14_0"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "picks.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1))
        (outdir / "picks_meta.json").write_text(json.dumps(mt, ensure_ascii=False, indent=1))
        log(f"picks[{arm}]: {len(pk)} dates")


def result_path(arm):
    return SIMS / arm / "V14_0" / "result.json"


def run_sims(max_n):
    todo = [a for a in ARMS if not result_path(a).exists()]
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


def report():
    rows = []
    for arm in ARMS:
        p = result_path(arm)
        if not p.exists():
            rows.append({"arm": arm, "status": "missing"})
            continue
        m = json.loads(p.read_text())["metrics"]
        rows.append({"arm": arm, "ann": m["annualized_return"], "sharpe": m["sharpe_ratio"],
                     "mdd": m["max_drawdown_pct"], "win": m["win_rate"],
                     "trades": m["closed_trade_count"], "total": m["total_return_pct"]})
    REF = {"causal_v1(offset0)": {"ann": 0.0432, "sharpe": 0.298, "mdd": 69.5, "win": 50.2},
           "archived_leaky(offset0)": {"ann": 0.6507, "sharpe": 1.911, "mdd": 29.9, "win": 64.0}}
    print(f"{'arm':<10s} {'ann':>8s} {'sharpe':>7s} {'mdd':>7s} {'win':>7s} {'trades':>7s}")
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['arm']:<10s}  (missing)")
            continue
        print(f"{r['arm']:<10s} {r['ann']:>7.2%} {r['sharpe']:>7.3f} {r['mdd']:>6.1f}% "
              f"{r['win']:>6.1f}% {int(r['trades']):>7d}")
    for k, v in REF.items():
        print(f"{k:<24s} {v['ann']:>6.2%} {v['sharpe']:>7.3f} {v['mdd']:>6.1f}% {v['win']:>6.1f}%")
    (OUT / "screen_results.json").write_text(json.dumps({"arms": rows, "ref": REF},
                                                        ensure_ascii=False, indent=1))
    log(f"saved {OUT/'screen_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.run is not None:
        run_sims(args.run)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
