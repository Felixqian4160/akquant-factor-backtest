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
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = pathlib.Path(os.environ.get("AKQ_PANEL", str(AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet")))
MAT = pathlib.Path(os.environ.get("AKQ_MATRIX", str(AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet")))
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
RUNNERS = {"v41": RUNNER, "v42": AKQ / "examples" / "v42_run_akquant_v34.py"}
OUT = AKQ / "evidence" / "v3_resonance_20261010"
CACHE = OUT / "_cache"
PICKS = OUT / "picks"
SIMS = OUT / "sims"

START, END = "2010-01-01", os.environ.get("AKQ_END", "2025-12-31")
GRID_STEP = 5
WS = (10, 20, 40, 60, 120)
UP_TH, DOWN_TH, NEED_WIN = 0.65, 0.35, 2
DIR_WIN, LAG = 250, 21
N_POS, MAX_GRID_AGE, COOLDOWN = 10, 4, 1
FACTOR_BATCH = 20
TAG = "V14_0"


def _resolve_ws(windows=None):
    if not windows:
        return WS
    return tuple(int(x) for x in str(windows).replace(" ", "").split(","))


def _th_suffix(up, dn, nw):
    if up == UP_TH and dn == DOWN_TH and nw == NEED_WIN:
        return ""
    return f"__u{int(round(up * 100))}d{int(round(dn * 100))}n{nw}"


def _cache_path(res_mode, ws, fact=None, th_suffix=""):
    if fact:
        return CACHE / f"votes__{fact}__{res_mode}_{'_'.join(map(str, ws))}{th_suffix}.npz"
    if res_mode == "any2" and ws == WS and not th_suffix:
        return CACHE / "votes_batches.npz"
    if res_mode == "streak" and ws == WS and not th_suffix:
        return CACHE / "votes_streak.npz"
    return CACHE / f"votes_{res_mode}_{'_'.join(map(str, ws))}{th_suffix}.npz"


def _read_factset(path):
    d = json.loads(pathlib.Path(path).read_text())
    return str(d["tag"]), [str(f) for f in d["factors"]]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_matrix(fact_names=None):
    mat = pl.read_parquet(MAT)
    if fact_names is not None:
        missing = [f for f in fact_names if f not in mat.columns]
        if missing:
            raise SystemExit(f"factset missing columns: {missing[:8]}")
        mat = mat.select(["trade_date"] + list(fact_names))
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


def signals(max_batches, res_mode="any2", windows=None, fact=None, fact_names=None,
            up_th=None, down_th=None, need_win=None):
    factors, M, dates = load_matrix(fact_names)
    T, F = M.shape
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    D = compute_dirs(M)
    ws = _resolve_ws(windows)
    up = UP_TH if up_th is None else float(up_th)
    dn = DOWN_TH if down_th is None else float(down_th)
    nw = NEED_WIN if need_win is None else int(need_win)
    ths = _th_suffix(up, dn, nw)
    n_batches = (F + FACTOR_BATCH - 1) // FACTOR_BATCH
    ck = _cache_path(res_mode, ws, fact, ths)

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

    score = None
    fs = fb = fbe = None
    if ck.exists():
        z = np.load(ck)
        bull = z["bull"]; bear = z["bear"]; start_batch = int(z["next_batch"])
        if "score" in z.files:
            score = z["score"]
        if "fs" in z.files:
            fs, fb, fbe = z["fs"], z["fbull"], z["fbear"]
        log(f"resume: batches done={start_batch}/{n_batches}")
    else:
        bull = np.zeros((T, S), dtype=np.int16)
        bear = np.zeros((T, S), dtype=np.int16)
        if fact is not None:
            score = np.zeros((T, S), dtype=np.float32)
    if fact is not None and fs is None:
        fs = np.zeros((F, T, S), dtype=np.float32)
        fb = np.zeros((F, T, S), dtype=np.int8)
        fbe = np.zeros((F, T, S), dtype=np.int8)

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
                facc = np.zeros((T, S)) if score is not None else None
                fcnt = np.zeros((T, S)) if score is not None else None
                for W in ws:
                    num = P[W:] - P[:-W]
                    den = NP[W:] - NP[:-W]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sw = np.where(den >= max(5, int(W * 0.4)), num / np.maximum(den, 1), np.nan)
                    upcnt[W - 1:] += (sw >= up).astype(np.int8)
                    dncnt[W - 1:] += (sw <= dn).astype(np.int8)
                    if facc is not None:
                        okw = np.isfinite(sw)
                        facc[W - 1:][okw] += sw[okw]
                        fcnt[W - 1:][okw] += 1.0
                if facc is not None:
                    fsc = np.where(fcnt > 0, facc / np.maximum(fcnt, 1.0), 0.0).astype(np.float32)
                    score += fsc
                    if fs is not None:
                        jj = bi * FACTOR_BATCH + j
                        fs[jj] = fsc
                        fb[jj] = (upcnt >= nw).astype(np.int8)
                        fbe[jj] = (dncnt >= nw).astype(np.int8)
                bull += (upcnt >= nw).astype(np.int16)
                bear += (dncnt >= nw).astype(np.int16)
            else:  # streak: adjacent pair 共振; sl: 短端(20) & 长端(120) 同时确认
                ups, downs = [], []
                facc = np.zeros((T, S)) if score is not None else None
                fcnt = np.zeros((T, S)) if score is not None else None
                for W in ws:
                    num = P[W:] - P[:-W]
                    den = NP[W:] - NP[:-W]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sw = np.where(den >= max(5, int(W * 0.4)), num / np.maximum(den, 1), np.nan)
                    u = np.zeros((T, S), dtype=bool)
                    u[W - 1:] = (sw >= up)
                    d = np.zeros((T, S), dtype=bool)
                    d[W - 1:] = (sw <= dn)
                    ups.append(u)
                    downs.append(d)
                    if facc is not None:
                        okw = np.isfinite(sw)
                        facc[W - 1:][okw] += sw[okw]
                        fcnt[W - 1:][okw] += 1.0
                if res_mode == "sl":
                    ish = ws.index(20) if 20 in ws else 0
                    ilg = len(ws) - 1
                    su = ups[ish] & ups[ilg]
                    sd = downs[ish] & downs[ilg]
                else:
                    su = np.zeros((T, S), dtype=bool)
                    sd = np.zeros((T, S), dtype=bool)
                    for i in range(len(ws) - 1):
                        su |= ups[i] & ups[i + 1]
                        sd |= downs[i] & downs[i + 1]
                if facc is not None:
                    fsc = np.where(fcnt > 0, facc / np.maximum(fcnt, 1.0), 0.0).astype(np.float32)
                    score += fsc
                    if fs is not None:
                        jj = bi * FACTOR_BATCH + j
                        fs[jj] = fsc
                        fb[jj] = su.astype(np.int8)
                        fbe[jj] = sd.astype(np.int8)
                bull += su.astype(np.int16)
                bear += sd.astype(np.int16)
        extra = {"score": score} if score is not None else {}
        if fs is not None:
            extra.update({"fs": fs, "fbull": fb, "fbear": fbe})
        np.savez_compressed(ck, bull=bull, bear=bear, next_batch=bi + 1, **extra)
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
          bear_pctl=0.90, res_mode="any2", net_min_pctl=None, rank_by="bull", offset=0,
          max_hold_bars=20, windows=None, n_pos=N_POS, fact=None, fact_names=None,
          ent_pctl=0.95, up_th=None, down_th=None, need_win=None,
          decide="count", gate_idx=0, select_trail=0, select_metric="mean", select_topn=5,
          select_mode="top", select_matrix="m", select_tau=0.7, select_lag=0):
    cool = int(cooldown)
    gstep = int(grid_step)
    off = int(offset)
    ws = _resolve_ws(windows)
    up = UP_TH if up_th is None else float(up_th)
    dn = DOWN_TH if down_th is None else float(down_th)
    nw = NEED_WIN if need_win is None else int(need_win)
    age_cap = max(1, int(max_hold_bars) // gstep)
    ck = _cache_path(res_mode, ws, fact, _th_suffix(up, dn, nw))
    if not ck.exists():
        raise SystemExit("signals incomplete")
    z = np.load(ck)
    bull, bear = z["bull"], z["bear"]
    score = z["score"] if "score" in z.files else None
    if rank_by == "score" and score is None:
        raise SystemExit("rank_by=score requires a factset cache with score")
    fs = z["fs"] if "fs" in z.files else None
    fbull = z["fbull"] if "fbull" in z.files else None
    fbear = z["fbear"] if "fbear" in z.files else None
    if decide in ("gate", "disp") and fs is None:
        raise SystemExit(f"decide={decide} requires per-factor cache fields (re-run signals)")
    factors, M, dates = load_matrix(fact_names)
    # 选择矩阵: m=收益矩阵(默认) / ic=IC矩阵; borda 需双矩阵
    MIC = None
    if select_matrix == "ic" or select_mode == "borda":
        icp = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_ic_ADJ.parquet"
        if not icp.exists():
            raise SystemExit("IC matrix missing: build matrix_ic_ADJ.parquet first")
        imm = pl.read_parquet(icp)
        missing = [f for f in factors if f not in imm.columns]
        if missing:
            raise SystemExit(f"ic matrix missing factors: {missing[:5]}")
        MIC = imm.select(list(factors)).to_numpy().astype(np.float64)
    MX = MIC if select_matrix == "ic" else M
    n_batches = (len(factors) + FACTOR_BATCH - 1) // FACTOR_BATCH
    assert int(z["next_batch"]) == n_batches, f"signals not complete: {int(z['next_batch'])}/{n_batches}"
    codes = pl.read_parquet(PANEL, columns=["ts_code"]).get_column("ts_code").unique().sort().to_list()
    T, S = bull.shape
    grid = [d for d in dates[off::gstep] if START <= d <= END]
    gpos = {d: i for i, d in enumerate(dates)}
    gi_all = np.array([gpos[d] for d in grid])

    # calibrate on 2010-2017 grids
    cal = gi_all[np.array([d <= "2017-12-31" for d in grid])]
    sel_trail = int(select_trail) if select_trail else 0
    sel_lag = int(select_lag) if select_lag else 0

    def _dyn_rows(rows):
        """动态选因子: 每行日期按近期效力 (settled) 打分 (矩阵 MX = m/ic).
        mode: top=top-N计数; wsum=全池加权软票; dedup=相关性去重; cluster=种子+一致性近邻; borda=收益+IC双榜."""
        bb = np.zeros((len(rows), S), dtype=np.float32)
        rr = np.zeros((len(rows), S), dtype=np.float32)
        ss = np.zeros((len(rows), S), dtype=np.float32)
        for k, i in enumerate(rows):
            hi = max(i - 20, 0)
            lo = max(hi - sel_trail, 0)
            llo = max(hi - DIR_WIN, 0)
            sc = None
            dual_top2 = None
            if hi - lo >= 20 and fbull is not None:
                def _metric(mat, trail=None):
                    tr = sel_trail if trail is None else trail
                    hi2 = max(hi - sel_lag, 0)          # 跳过期 (清华论文: skip 最近1月)
                    lo2 = max(hi2 - tr, 0)
                    with np.errstate(all="ignore"):
                        lmean = np.nanmean(mat[llo:hi, :], axis=0)
                        rmean = np.nanmean(mat[lo2:hi2, :], axis=0)
                        rstd = np.nanstd(mat[lo2:hi2, :], axis=0)
                        rhit = np.nanmean(mat[lo2:hi2, :] > 0, axis=0)
                    d = np.where(np.nan_to_num(lmean, nan=0.0) >= 0, 1.0, -1.0)
                    if select_metric == "ir":
                        return d * np.where(rstd > 0, rmean / np.maximum(rstd, 1e-12), -1e18)
                    if select_metric == "hit":
                        return np.where(d > 0, rhit, 1.0 - rhit)
                    if select_metric == "mo":  # 效力动量: 本期 vs 上一窗口的改善
                        return d * (rmean - np.nanmean(mat[max(lo2 - tr, 0):lo2, :], axis=0))
                    return d * rmean
                if select_mode == "borda":
                    a1 = np.nan_to_num(_metric(M), nan=-1e18)
                    a2 = np.nan_to_num(_metric(MIC), nan=-1e18)
                    sc = (np.argsort(np.argsort(a1)) + np.argsort(np.argsort(a2))).astype(np.float64)
                elif select_mode == "dual":
                    # 双回看共振: 120日排名 + 20日排名 的秩和 (收益 mean 口径)
                    a1 = np.nan_to_num(_metric(MX, 120), nan=-1e18)
                    a2 = np.nan_to_num(_metric(MX, 20), nan=-1e18)
                    sc = (np.argsort(np.argsort(a1)) + np.argsort(np.argsort(a2))).astype(np.float64)
                elif select_mode == "dualx":
                    # 硬交集: 20日榜 top15 内, 按 120日分排序
                    sc = _metric(MX, 120)
                    dual_top2 = set(np.argsort(-np.nan_to_num(_metric(MX, 20), nan=-1e18))[:15].tolist())
                else:
                    sc = _metric(MX)
            if sc is not None:
                order = np.argsort(-np.nan_to_num(sc, nan=-1e18))
            if select_mode == "wsum" and sc is not None:
                w = np.maximum(np.nan_to_num(sc, nan=0.0), 0.0)
                tot = w.sum()
                w = (w / tot if tot > 0 else np.ones(len(factors)) / len(factors)) * select_topn
                bb[k] = (w[:, None] * fbull[:, i, :]).sum(axis=0)
                rr[k] = (w[:, None] * fbear[:, i, :]).sum(axis=0)
                ss[k] = (w[:, None] * fs[:, i, :]).sum(axis=0)
            else:
                if sc is not None and select_mode == "dedup":
                    sel = []
                    for c in order[:100]:
                        if len(sel) >= int(select_topn):
                            break
                        okc = True
                        for s0 in sel:
                            a = M[lo:hi, c]
                            b2 = M[lo:hi, s0]
                            msk = np.isfinite(a) & np.isfinite(b2)
                            if msk.sum() >= 30 and abs(np.corrcoef(a[msk], b2[msk])[0, 1]) > select_tau:
                                okc = False
                                break
                        if okc:
                            sel.append(int(c))
                    for c in order:
                        if len(sel) >= int(select_topn):
                            break
                        if int(c) not in sel:
                            sel.append(int(c))
                    sel = np.array(sel[:int(select_topn)])
                elif sc is not None and select_mode == "cluster":
                    seed = int(order[0])
                    cands = [int(c) for c in order[1:1 + 30]]
                    smask = fbull[seed, lo:hi, :] > 0
                    inter = np.array([np.logical_and(smask, fbull[c, lo:hi, :] > 0).sum() for c in cands],
                                     dtype=np.float64)
                    union = np.array([np.logical_or(smask, fbull[c, lo:hi, :] > 0).sum() for c in cands],
                                     dtype=np.float64)
                    agr = inter / np.maximum(union, 1)
                    peers = [cands[j] for j in np.argsort(-agr)[: int(select_topn) - 1]]
                    sel = np.array([seed] + peers)
                elif sc is not None and select_mode == "dualx" and dual_top2 is not None:
                    sel = [int(c) for c in order if int(c) in dual_top2]
                    for c in order:
                        if len(sel) >= int(select_topn):
                            break
                        if int(c) not in sel:
                            sel.append(int(c))
                    sel = np.array(sel[:int(select_topn)])
                elif sc is not None:
                    sel = order[:int(select_topn)]
                else:
                    sel = np.arange(len(factors))
                bb[k] = fbull[sel, i, :].sum(axis=0)
                rr[k] = fbear[sel, i, :].sum(axis=0)
                ss[k] = fs[sel, i, :].sum(axis=0)
        return bb, rr, ss

    if sel_trail:
        if fs is None or fbull is None or fbear is None:
            raise SystemExit("select layer requires fs/fbull/fbear cache (re-run signals)")
        DB, DR, DS = _dyn_rows(gi_all)
        mcal = np.isin(gi_all, cal)
        ENT_MIN = int(np.ceil(np.quantile(DB[mcal], ent_pctl)))
        HOLD_MIN = int(np.ceil(np.quantile(DB[mcal], hold_pctl)))
        BEAR_MAX = int(np.ceil(np.quantile(DR[mcal], bear_pctl)))
        log(f"dynamic-select: trail={sel_trail} metric={select_metric} topn={select_topn} "
            f"-> ENT_MIN={ENT_MIN} HOLD_MIN={HOLD_MIN} BEAR_MAX={BEAR_MAX} (2010-17 dynamic votes)")
    else:
        ENT_MIN = int(np.ceil(np.quantile(bull[cal], ent_pctl)))
        HOLD_MIN = int(np.ceil(np.quantile(bull[cal], hold_pctl)))
        BEAR_MAX = int(np.ceil(np.quantile(bear[cal], bear_pctl)))
    NET_MIN = None
    if net_min_pctl is not None:
        net_cal = bull[cal].astype(np.int32) - bear[cal].astype(np.int32)
        NET_MIN = int(np.ceil(np.quantile(net_cal, net_min_pctl)))
    S_ENT = S_HOLD = G1 = G2 = D1 = D2 = None
    if decide == "score":
        S_ENT = float(np.quantile(score[cal], ent_pctl))
        S_HOLD = float(np.quantile(score[cal], hold_pctl))
    elif decide == "gate":
        G1 = float(np.quantile(fs[gate_idx][cal], 0.75))
        G2 = float(np.quantile(fs[gate_idx][cal], 0.50))
    elif decide == "disp":
        dcal = fs[:, cal, :].std(axis=0)
        D1 = float(np.quantile(dcal, 0.75))
        D2 = float(np.quantile(dcal, 0.90))
    if decide != "count":
        log(f"decide={decide}: S_ENT={S_ENT} S_HOLD={S_HOLD} G1={G1} G2={G2} D1={D1} D2={D2} gate_idx={gate_idx}")
    log(f"calibrated (2010-2017 grids n={len(cal)}): ENT_MIN(P{int(ent_pctl*100)})={ENT_MIN} "
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
        if sel_trail:
            b_row = DB[gi]
            r_row = DR[gi]
            s_dyn = DS[gi]
        else:
            b_row = bull[i].astype(np.int32)
            r_row = bear[i].astype(np.int32)
            s_dyn = None
        net_row = b_row - r_row
        if decide == "score":
            srow = score[i].astype(np.float32)
            elig = (srow >= S_ENT) & (r_row <= BEAR_MAX)
            keep = (srow >= S_HOLD) & (r_row <= BEAR_MAX)
        elif decide == "gate":
            grow = fs[gate_idx][i]
            elig = (grow >= G1) & (b_row >= ENT_MIN) & (r_row <= BEAR_MAX)
            keep = (grow >= G2) & (b_row >= HOLD_MIN) & (r_row <= BEAR_MAX)
        elif decide == "disp":
            drow = fs[:, i, :].std(axis=0)
            elig = (b_row >= ENT_MIN) & (r_row <= BEAR_MAX) & (drow <= D1)
            keep = (b_row >= HOLD_MIN) & (r_row <= BEAR_MAX) & (drow <= D2)
        else:
            elig = (b_row >= ENT_MIN) & (r_row <= BEAR_MAX)
            keep = (b_row >= HOLD_MIN) & (r_row <= BEAR_MAX)
        if NET_MIN is not None:
            elig &= (net_row >= NET_MIN)
        n_elig = int(elig.sum())
        survivors = [c for c in hold if keep[codes.index(c)] and (gi - hold[c]) < age_cap]
        # drops (condition break or age) -> cooldown
        for c in list(hold):
            if c not in survivors:
                cooldown[c] = gi
                del hold[c]
        n_surv = len(survivors)
        slots = n_pos - n_surv
        n_new = 0
        if slots > 0:
            cand = [j for j in range(S) if elig[j] and codes[j] not in hold
                    and (gi - cooldown.get(codes[j], -999)) >= cool]
            if rank_by == "net":
                cand.sort(key=lambda j: (-net_row[j], r_row[j], codes[j]))
            elif rank_by == "score":
                s_row = (s_dyn if s_dyn is not None else score[i]).astype(np.float64)
                cand.sort(key=lambda j: (-s_row[j], r_row[j], codes[j]))
            else:
                cand.sort(key=lambda j: (-b_row[j], r_row[j], codes[j]))
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
        "framework": "v3_resonance", "up_th": up, "down_th": dn,
        "need_win": nw, "dir_window": DIR_WIN, "lag": LAG, "grid_step": gstep,
        "age_cap_grids": age_cap, "offset": off,
        "max_hold_bars": int(max_hold_bars), "windows": list(ws),
        "n_pos": int(n_pos), "max_grid_age": MAX_GRID_AGE, "cooldown_grids": cool,
        "hold_pctl": hold_pctl, "bear_pctl": bear_pctl, "res_mode": res_mode,
        "net_min": NET_MIN, "rank_by": rank_by,
        "ent_min": ENT_MIN, "hold_min": HOLD_MIN, "bear_max": BEAR_MAX, "ent_pctl": ent_pctl,
        "decide": decide, "gate_factor": (factors[gate_idx] if decide == "gate" else None),
        "score_cuts": [S_ENT, S_HOLD], "gate_cuts": [G1, G2], "disp_cuts": [D1, D2],
        "dynamic_select": ({"trail": sel_trail, "metric": select_metric, "topn": int(select_topn),
                            "mode": select_mode, "matrix": select_matrix, "lag": sel_lag,
                            "tau": (select_tau if select_mode == "dedup" else None)}
                           if sel_trail else None),
        "calibration": "2010-2017 grid rows",
        "factor_pool": (f"{len(factors)} factors factset={fact}" if fact else f"{len(factors)} dedup factors"),
        "factset": fact,
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


def run_sims(max_n, tag=TAG, runner=None, holding_bars=0):
    if result_path(tag).exists():
        log("sim already exists")
        return
    t0 = time.time()
    cmd = [sys.executable, "-u", str(runner or RUNNER), "--tag", tag,
           "--picks-base", str(PICKS), "--actions", "on", "--ca-mode", "all",
           "--prices", "raw", "--end", END, "--out-base", str(SIMS)]
    if holding_bars:
        cmd += ["--holding-bars", str(int(holding_bars))]
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
    ap.add_argument("--res-mode", choices=["any2", "streak", "sl"], default="any2")
    ap.add_argument("--net-min-pctl", type=float, default=None)
    ap.add_argument("--rank-by", choices=["bull", "net", "score"], default="bull")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-hold-bars", type=int, default=20)
    ap.add_argument("--n-pos", type=int, default=N_POS)
    ap.add_argument("--windows", default="")
    ap.add_argument("--runner", choices=["v41", "v42"], default="v41")
    ap.add_argument("--holding-bars", type=int, default=0)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--factset", default=None)
    ap.add_argument("--ent-pctl", type=float, default=0.95)
    ap.add_argument("--up-th", type=float, default=None)
    ap.add_argument("--down-th", type=float, default=None)
    ap.add_argument("--need-win", type=int, default=None)
    ap.add_argument("--decide", choices=["count", "score", "gate", "disp"], default="count")
    ap.add_argument("--gate-idx", type=int, default=0)
    ap.add_argument("--select-trail", type=int, default=0)
    ap.add_argument("--select-metric", choices=["mean", "ir", "hit", "mo"], default="mean")
    ap.add_argument("--select-topn", type=int, default=5)
    ap.add_argument("--select-lag", type=int, default=0)
    ap.add_argument("--select-mode", choices=["top", "wsum", "dedup", "cluster", "borda", "dual", "dualx"], default="top")
    ap.add_argument("--select-matrix", choices=["m", "ic"], default="m")
    ap.add_argument("--dedup-tau", type=float, default=0.7)
    args = ap.parse_args()
    fact, fact_names = (None, None)
    if args.factset:
        fact, fact_names = _read_factset(args.factset)
    if args.signals is not None:
        signals(args.signals, args.res_mode, args.windows, fact, fact_names,
                args.up_th, args.down_th, args.need_win)
    elif args.status:
        status()
    elif args.build:
        build(args.hold_pctl, args.cooldown, args.tag, args.grid_step,
              args.bear_pctl, args.res_mode, args.net_min_pctl, args.rank_by, args.offset,
              args.max_hold_bars, args.windows, args.n_pos, fact, fact_names, args.ent_pctl,
              args.up_th, args.down_th, args.need_win, args.decide, args.gate_idx,
              args.select_trail, args.select_metric, args.select_topn, args.select_mode,
              args.select_matrix, args.dedup_tau, args.select_lag)
    elif args.run is not None:
        run_sims(args.run, args.tag, RUNNERS[args.runner], args.holding_bars)
    elif args.report:
        report(args.tag)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
