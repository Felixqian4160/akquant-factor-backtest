"""v3 共振选股框架 — 多窗口共振买入 + 条件失效卖出轮动 (用户合同 2026-10-10)

概念 (用户原话):
  第一: 利用股票过去 10/20/40/60/120 天中一个或几个共振的因子, 预测股票未来
        5/10/20 天能涨 -> 买入, 持有对应时间
  第二: 同一批因子预测到持仓股有跌的风险 (原买入条件不成立) -> 卖出, 换入
        其他满足条件的股票

实现 (v1 配置):
  信号层 (每因子 f, 每股 s, 每日 t, 全部因果):
    - 因子横截面百分位 r(f,t,s) = rank_pct(值, 同日全部股票)   [方向修正: 用已结算
      业绩滚动方向 dir(f,t) = sign(mean(结算收益 diff[f, t-270 : t-20]))]
    - 多窗口持续性: sw(f,t,s,W) = 过去 W 日 r 的均值, W ∈ {10,20,40,60,120}
    - 共振判定: factor 看多 = (>=2 个窗口 sw >= 0.65); 看空 = (>=2 窗口 sw <= 0.35)
    - 投票: BULL(s) = 看多因子数; BEAR(s) = 看空因子数
  选股层 (5 交易日检查网格):
    - 买入条件: BULL >= ENT_MIN 且 BEAR <= BEAR_MAX (阈值在 2010-2017 网格上校准)
    - 持有条件 (滞回): BULL >= HOLD_MIN 且 BEAR <= BEAR_MAX
    - 卖出: 持有条件失效 或 持有达 4 个网格 (~20 bars) -> 下一网格剔除
    - 冷却: 卖出后 1 个网格才可再买入; 目标持仓 N=10, 不足不强行补
  执行层: v41 runner (T+1 NextOpen, 等权 90% 仓, 佣金 0.25% + 滑点 0.10%, CA, 21bar 上限)

用法:
  --signals K   计算 K 个缺失因子批次 (20 因子/批, 断点续传)
  --status      显示信号进度
  --build       校准阈值 + 构建 picks (5 日网格, 2010-2025)
  --run N       跑最多 N 个缺失 sim
  --report      指标 + 2018-2025 分段 + 对照表
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
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
OUT = AKQ / "evidence" / "v3_resonance_20261010"
CACHE = OUT / "_cache"
PICKS = OUT / "picks"
SIMS = OUT / "sims"

START, END = "2010-01-01", "2025-12-31"
GRID_STEP = 5
WS = (10, 20, 40, 60, 120)
UP_TH, DOWN_TH, NEED_WIN = 0.65, 0.35, 2
DIR_WIN, LAG = 250, 21
N_POS, MAX_GRID_AGE, COOLDOWN = 10, 4, 1
FACTOR_BATCH = 20
TAG = "V14_0"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_matrix():
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy().astype(np.float64)
    dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    return factors, M, dates


def compute_dirs(M):
    """dir(f,t) = sign(mean settled diff rows [t-20-250, t-20)) ; +1 default."""
    T, F = M.shape
    valid = np.isfinite(M)
    Mc = np.where(valid, M, 0.0)
    P = np.concatenate([np.zeros((1, F)), np.cumsum(Mc, axis=0)], axis=0)   # (T+1,F)
    NP = np.concatenate([np.zeros((1, F)), np.cumsum(valid.astype(np.float64), axis=0)], axis=0)
    hi = np.clip(np.arange(T) - 20, 0, T)     # settled rows [.., hi): rows <= t-21
    lo = np.clip(hi - DIR_WIN, 0, T)
    num = P[hi] - P[lo]
    den = NP[hi] - NP[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den >= 30, num / np.maximum(den, 1), 0.0)
    D = np.where(m >= 0, 1, -1).astype(np.int8)
    return D


def signals(max_batches, res_mode="any2"):
    factors, M, dates = load_matrix()
    T, F = M.shape
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    D = compute_dirs(M)
    n_batches = (F + FACTOR_BATCH - 1) // FACTOR_BATCH
    ck = CACHE / ("votes_batches.npz" if res_mode == "any2" else f"votes_{res_mode}.npz")

    bull = np.zeros((T, 0), dtype=np.int16)
    bear = np.zeros((T, 0), dtype=np.int16)
    start_batch = 0
    # global stock list & dates mapping
    codes = pl.read_parquet(PANEL, columns=["ts_code"]).get_column("ts_code").unique().sort().to_list()
    S = len(codes)
    mat_ms = np.array([np.datetime64(d) for d in dates]).astype("datetime64[ms]")
    pan_ms = (pl.read_parquet(PANEL, columns=["trade_date"]).get_column("trade_date")
              .unique().sort().to_numpy().astype("datetime64[ms]"))
    assert len(mat_ms) == len(pan_ms) and np.array_equal(mat_ms, pan_ms), "panel/matrix date mismatch"
    date_ms_global = mat_ms

    if ck.exists():
        z = np.load(ck)
        bull = z["bull"]; bear = z["bear"]; start_batch = int(z["next_batch"])
        log(f"resume: batches done={start_batch}/{n_batches}")
    else:
        bull = np.zeros((T, S), dtype=np.int16)
        bear = np.zeros((T, S), dtype=np.int16)

    done_now = 0
    for bi in range(start_batch, n_batches):
        if done_now >= max_batches:
            break
        t0 = time.time()
        batch = factors[bi * FACTOR_BATCH:(bi + 1) * FACTOR_BATCH]
        df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + batch)
        tvals = df.get_column("trade_date").to_numpy().astype("datetime64[ms]")
        ti = np.searchsorted(date_ms_global, tvals)
        cvals = df.get_column("ts_code").to_numpy()
        si = pd.Categorical(cvals, categories=codes).codes
        vals = df.select(batch).to_numpy()   # (n, B)
        cube = np.full((T, S, len(batch)), np.nan)
        cube[ti, si, :] = vals
        for j in range(len(batch)):
            V = cube[:, :, j]
            ok = np.isfinite(V)
            Vadj = np.where(D[:, j][:, None] > 0, V, -V)
            Vf = np.where(ok, Vadj, np.inf)
            order = np.argsort(Vf, axis=1, kind="stable")
            ranks = np.argsort(order, axis=1, kind="stable")
            counts = ok.sum(axis=1, keepdims=True)
            R = (ranks + 1) / np.maximum(counts, 1)
            R = np.where(ok, R, np.nan)
            Rc = np.where(np.isfinite(R), R, 0.0)
            Nc = np.isfinite(R).astype(np.float64)
            P = np.concatenate([np.zeros((1, S)), np.cumsum(Rc, axis=0)], axis=0)
            NP = np.concatenate([np.zeros((1, S)), np.cumsum(Nc, axis=0)], axis=0)
            if res_mode == "any2":
                upcnt = np.zeros((T, S), dtype=np.int8)
                dncnt = np.zeros((T, S), dtype=np.int8)
                for W in WS:
                    num = P[W:] - P[:-W]
                    den = NP[W:] - NP[:-W]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sw = np.where(den >= max(5, int(W * 0.4)), num / np.maximum(den, 1), np.nan)
                    upcnt[W - 1:] += (sw >= UP_TH).astype(np.int8)
                    dncnt[W - 1:] += (sw <= DOWN_TH).astype(np.int8)
                bull += (upcnt >= NEED_WIN).astype(np.int16)
                bear += (dncnt >= NEED_WIN).astype(np.int16)
            else:  # streak: >=1 adjacent window pair both up (10&20|20&40|40&60|60&120)
                ups, downs = [], []
                for W in WS:
                    num = P[W:] - P[:-W]
                    den = NP[W:] - NP[:-W]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sw = np.where(den >= max(5, int(W * 0.4)), num / np.maximum(den, 1), np.nan)
                    u = np.zeros((T, S), dtype=bool)
                    u[W - 1:] = (sw >= UP_TH)
                    d = np.zeros((T, S), dtype=bool)
                    d[W - 1:] = (sw <= DOWN_TH)
                    ups.append(u)
                    downs.append(d)
                su = np.zeros((T, S), dtype=bool)
                sd = np.zeros((T, S), dtype=bool)
                for i in range(len(WS) - 1):
                    su |= ups[i] & ups[i + 1]
                    sd |= downs[i] & downs[i + 1]
                bull += su.astype(np.int16)
                bear += sd.astype(np.int16)
        np.savez_compressed(ck, bull=bull, bear=bear, next_batch=bi + 1)
        done_now += 1
        log(f"batch {bi+1}/{n_batches} ({len(batch)} factors) {time.time()-t0:.0f}s  "
            f"[bull mean {bull.sum(1).mean()/S:.1f}, bear mean {bear.sum(1).mean()/S:.1f}]")
    log(f"signals: {min(start_batch+done_now, n_batches)}/{n_batches} batches")


def status():
    ck = CACHE / "votes_batches.npz"
    if ck.exists():
        z = np.load(ck)
        log(f"batches done: {int(z['next_batch'])} | bull shape {z['bull'].shape}")
    else:
        log("no signals yet")


def build(hold_pctl=0.90, cooldown=1, tag=TAG, grid_step=GRID_STEP,
          bear_pctl=0.90, res_mode="any2", net_min_pctl=None, rank_by="bull", offset=0):
    cool = int(cooldown)
    gstep = int(grid_step)
    off = int(offset)
    age_cap = max(1, 20 // gstep)
    ck = CACHE / ("votes_batches.npz" if res_mode == "any2" else f"votes_{res_mode}.npz")
    if not ck.exists():
        raise SystemExit("signals incomplete")
    z = np.load(ck)
    bull, bear = z["bull"], z["bear"]
    factors, M, dates = load_matrix()
    n_batches = (len(factors) + FACTOR_BATCH - 1) // FACTOR_BATCH
    assert int(z["next_batch"]) == n_batches, f"signals not complete: {int(z['next_batch'])}/{n_batches}"
    codes = pl.read_parquet(PANEL, columns=["ts_code"]).get_column("ts_code").unique().sort().to_list()
    T, S = bull.shape
    grid = [d for d in dates[off::gstep] if START <= d <= END]
    gpos = {d: i for i, d in enumerate(dates)}
    gi_all = np.array([gpos[d] for d in grid])

    # calibrate on 2010-2017 grids
    cal = gi_all[np.array([d <= "2017-12-31" for d in grid])]
    ENT_MIN = int(np.ceil(np.quantile(bull[cal], 0.95)))
    HOLD_MIN = int(np.ceil(np.quantile(bull[cal], hold_pctl)))
    BEAR_MAX = int(np.ceil(np.quantile(bear[cal], bear_pctl)))
    NET_MIN = None
    if net_min_pctl is not None:
        net_cal = bull[cal].astype(np.int32) - bear[cal].astype(np.int32)
        NET_MIN = int(np.ceil(np.quantile(net_cal, net_min_pctl)))
    log(f"calibrated (2010-2017 grids n={len(cal)}): ENT_MIN(P95)={ENT_MIN} "
        f"HOLD_MIN(P{int(hold_pctl*100)})={HOLD_MIN} BEAR_MAX(P{int(bear_pctl*100)})={BEAR_MAX} "
        f"NET_MIN({int(net_min_pctl*100) if net_min_pctl else '-'})={NET_MIN} res={res_mode}")
    log(f"bull dist: mean {bull[cal].mean():.1f} p90 {np.quantile(bull[cal],0.90):.0f} "
        f"p95 {np.quantile(bull[cal],0.95):.0f} p99 {np.quantile(bull[cal],0.99):.0f} | "
        f"bear mean {bear[cal].mean():.1f}")

    picks, meta = {}, {}
    hold, cooldown = {}, {}
    stats = {"n_elig": [], "n_picks": [], "n_surv": [], "n_new": []}
    for gi, d in enumerate(grid):
        i = gpos[d]
        b_row = bull[i].astype(np.int32)
        r_row = bear[i].astype(np.int32)
        net_row = b_row - r_row
        elig = (b_row >= ENT_MIN) & (r_row <= BEAR_MAX)
        if NET_MIN is not None:
            elig &= (net_row >= NET_MIN)
        keep = (b_row >= HOLD_MIN) & (r_row <= BEAR_MAX)
        n_elig = int(elig.sum())
        survivors = [c for c in hold if keep[codes.index(c)] and (gi - hold[c]) < age_cap]
        # drops (condition break or age) -> cooldown
        for c in list(hold):
            if c not in survivors:
                cooldown[c] = gi
                del hold[c]
        n_surv = len(survivors)
        slots = N_POS - n_surv
        n_new = 0
        if slots > 0:
            cand = [j for j in range(S) if elig[j] and codes[j] not in hold
                    and (gi - cooldown.get(codes[j], -999)) >= cool]
            cand.sort(key=lambda j: (-net_row[j], r_row[j], codes[j]) if rank_by == "net"
                      else (-b_row[j], r_row[j], codes[j]))
            for j in cand[:slots]:
                hold[codes[j]] = gi
                survivors.append(codes[j])
                n_new += 1
        basket = {c: int(b_row[codes.index(c)]) for c in survivors}
        if basket:
            picks[d] = basket
            meta[d] = {"n_picks": len(basket), "n_elig": n_elig,
                       "n_survivors": n_surv, "n_new": n_new}
        stats["n_elig"].append(n_elig)
        stats["n_picks"].append(len(basket))
    # recompute survivor/new split per date for meta (light pass)
    contract = {
        "framework": "v3_resonance", "windows": list(WS), "up_th": UP_TH, "down_th": DOWN_TH,
        "need_win": NEED_WIN, "dir_window": DIR_WIN, "lag": LAG, "grid_step": gstep,
        "age_cap_grids": age_cap, "offset": off,
        "n_pos": N_POS, "max_grid_age": MAX_GRID_AGE, "cooldown_grids": cool,
        "hold_pctl": hold_pctl, "bear_pctl": bear_pctl, "res_mode": res_mode,
        "net_min": NET_MIN, "rank_by": rank_by,
        "ent_min": ENT_MIN, "hold_min": HOLD_MIN, "bear_max": BEAR_MAX,
        "calibration": "2010-2017 grid rows", "factor_pool": f"{len(factors)} dedup factors",
        "execution": "v41 (T+1 NextOpen, equal weight 90%, comm 0.25% + slip 0.10%, CA, 21bar cap)",
    }
    outdir = PICKS / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (outdir / "picks_meta.json").write_text(json.dumps({"contract": contract, "per_day": meta},
                                                        ensure_ascii=False, indent=1))
    log(f"picks: {len(picks)} dates; avg picks/day {np.mean(stats['n_picks']):.1f}; "
        f"avg eligible/day {np.mean(stats['n_elig']):.1f}")
    log(f"contract -> {outdir/'picks_meta.json'}")


def result_path(tag=TAG):
    return SIMS / tag / "result.json"


def run_sims(max_n, tag=TAG):
    if result_path(tag).exists():
        log("sim already exists")
        return
    t0 = time.time()
    cmd = [sys.executable, "-u", str(RUNNER), "--tag", tag,
           "--picks-base", str(PICKS), "--actions", "on", "--ca-mode", "all",
           "--prices", "raw", "--end", "2025-12-31", "--out-base", str(SIMS)]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
    ok = result_path(tag).exists()
    log(f"sim: {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")


def seg_metrics(nav_path, start="2018-01-01", end="2025-12-31"):
    df = pd.read_csv(nav_path)
    df["d"] = pd.to_datetime(df["date"])
    df = df.sort_values("d")
    df = df[df["d"] <= pd.Timestamp(end)]
    pre = df[df["d"] < pd.Timestamp(start)]["value"]
    if len(pre) == 0:
        return None
    base = float(pre.iloc[-1])
    seg = df[df["d"] >= pd.Timestamp(start)]
    vals = seg["value"].to_numpy(dtype=float) / base
    curve = np.concatenate(([1.0], vals))
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1.0))
    ret = float(vals[-1] - 1.0)
    years = (seg["d"].iloc[-1] - seg["d"].iloc[0]).days / 365.25
    ann = float((1.0 + ret) ** (1.0 / years) - 1.0)
    daily = np.diff(vals) / vals[:-1]
    sh = float(daily.mean() / daily.std() * np.sqrt(244)) if daily.std() > 0 else 0.0
    return {"ret": ret, "ann": ann, "mdd": mdd, "sharpe": sh}


def report(tag=TAG):
    p = result_path(tag)
    if not p.exists():
        log("no result yet")
        return
    m = json.loads(p.read_text())["metrics"]
    log(f"v3 resonance: ann={m['annualized_return']:.2%} sharpe={m['sharpe_ratio']:.3f} "
        f"mdd={m['max_drawdown_pct']:.1f}% win={m['win_rate']:.1f}% trades={m['closed_trade_count']}")
    nav = SIMS / tag / "nav.csv"
    s = seg_metrics(nav)
    if s:
        log(f"segment 2018-2025: ann={s['ann']:.2%} mdd={s['mdd']:.1%} sharpe={s['sharpe']:.3f}")
    ref = {
        "variant": tag,
        "v3_resonance": {"ann": m["annualized_return"], "sharpe": m["sharpe_ratio"],
                         "mdd": m["max_drawdown_pct"], "win": m["win_rate"],
                         "seg": s},
        "refs": {
            "static_2017": {"full_ann": 0.1787, "seg_ann": 0.2123, "seg_mdd": -0.4489},
            "k120": {"full_ann": 0.1216, "seg_ann": 0.0918, "seg_mdd": -0.3285},
            "causal_v1": {"full_ann": 0.0432, "seg_ann": 0.1008, "seg_mdd": -0.5610},
            "EW_portfolio": {"seg_ann": 0.1669, "seg_mdd": -0.2613},
            "HS300_index": {"seg_ann": 0.0175, "seg_mdd": -0.4560},
        },
    }
    (OUT / "report.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1, default=str))
    log(f"saved {OUT/'report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=int, metavar="K")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--hold-pctl", type=float, default=0.90)
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--grid-step", type=int, default=GRID_STEP)
    ap.add_argument("--bear-pctl", type=float, default=0.90)
    ap.add_argument("--res-mode", choices=["any2", "streak"], default="any2")
    ap.add_argument("--net-min-pctl", type=float, default=None)
    ap.add_argument("--rank-by", choices=["bull", "net"], default="bull")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()
    if args.signals is not None:
        signals(args.signals, args.res_mode)
    elif args.status:
        status()
    elif args.build:
        build(args.hold_pctl, args.cooldown, args.tag, args.grid_step,
              args.bear_pctl, args.res_mode, args.net_min_pctl, args.rank_by, args.offset)
    elif args.run is not None:
        run_sims(args.run, args.tag)
    elif args.report:
        report(args.tag)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
