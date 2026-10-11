"""IC 矩阵构建 (T×413): IC_f(t) = spearman(Vadj_f[t], fwd[t]) — 与收益矩阵同结算合同.

合同:
  fwd[t]   = adj_close[t+21] / adj_open[t+1] - 1 - 0.005          (settled t+21)
  Vadj[t]  = V[t] * sign(D[t,f])                                  (D = compute_dirs(M), 逐日方向)
  IC_f(t)  = pearson(rank(Vadj_f[t]), rank(fwd[t]))                (共同有效股票 n >= 30)
输出: evidence/v34_dedup_20261010/matrix_ic_ADJ.parquet
分片: evidence/v34_dedup_20261010/_cache_ic/batch_{bi:02d}.npz
用法: --all / --assemble / --verify
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

AKQ = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(AKQ / "examples"))
from v3_resonance_framework import compute_dirs, load_matrix  # noqa: E402

PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
OUTDIR = AKQ / "evidence" / "v34_dedup_20261010"
ICMAT = OUTDIR / "matrix_ic_ADJ.parquet"
CACHE = OUTDIR / "_cache_ic"
BATCH = 20
MIN_N = 30
SHIFT = 21


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_globals():
    factors, M, dates = load_matrix()
    T, F = M.shape
    D = compute_dirs(M)
    codes = pl.read_parquet(PANEL, columns=["ts_code"]).get_column("ts_code").unique().sort().to_list()
    S = len(codes)
    date_ms = np.array([np.datetime64(d) for d in dates]).astype("datetime64[ms]")
    return factors, M, dates, D, codes, S, date_ms, T, F


def build_fwd(dates, codes, S, T, date_ms):
    df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_close", "adj_open"])
    tvals = df.get_column("trade_date").to_numpy().astype("datetime64[ms]")
    ti = np.searchsorted(date_ms, tvals)
    si = pd.Categorical(df.get_column("ts_code").to_numpy(), categories=codes).codes
    CL = np.full((T, S), np.nan)
    OP = np.full((T, S), np.nan)
    CL[ti, si] = df.get_column("adj_close").to_numpy()
    OP[ti, si] = df.get_column("adj_open").to_numpy()
    FWD = np.full((T, S), np.nan)
    n = T - SHIFT
    with np.errstate(all="ignore"):
        FWD[:n] = CL[SHIFT:] / OP[1:1 + n] - 1.0 - 0.005
    return FWD


def run_all():
    t_all = time.time()
    factors, M, dates, D, codes, S, date_ms, T, F = load_globals()
    FWD = build_fwd(dates, codes, S, T, date_ms)
    fwd_ok = np.isfinite(FWD)
    log(f"globals ready: T={T} F={F} S={S} fwd_valid={(fwd_ok).mean():.2%}")
    nb = (F + BATCH - 1) // BATCH
    for bi in range(nb):
        ck = CACHE / f"batch_{bi:02d}.npz"
        if ck.exists():
            log(f"batch {bi}/{nb}: exists, skip")
            continue
        t0 = time.time()
        batch = factors[bi * BATCH:(bi + 1) * BATCH]
        B = len(batch)
        df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code"] + batch)
        tvals = df.get_column("trade_date").to_numpy().astype("datetime64[ms]")
        ti = np.searchsorted(date_ms, tvals)
        si = pd.Categorical(df.get_column("ts_code").to_numpy(), categories=codes).codes
        vals = df.select(batch).to_numpy()
        cube = np.full((T, S, B), np.nan)
        cube[ti, si, :] = vals
        ics = np.full((T, B), np.nan)
        for j in range(B):
            g = bi * BATCH + j
            V = cube[:, :, j]
            m = np.isfinite(V) & fwd_ok
            nrow = m.sum(axis=1)
            if not m.any():
                continue
            Vadj = np.where(D[:, g][:, None] > 0, V, -V)
            xf = np.where(m, Vadj, np.inf)
            yf = np.where(m, FWD, np.inf)
            ox = np.argsort(xf, axis=1, kind="stable")
            rx = (np.argsort(ox, axis=1, kind="stable") + 1.0)
            oy = np.argsort(yf, axis=1, kind="stable")
            ry = (np.argsort(oy, axis=1, kind="stable") + 1.0)
            rx = np.where(m, rx, np.nan)
            ry = np.where(m, ry, np.nan)
            nn = np.maximum(nrow, 1).astype(np.float64)
            mx = np.nansum(rx, axis=1) / nn
            my = np.nansum(ry, axis=1) / nn
            cov = np.nansum(rx * ry, axis=1) / nn - mx * my
            vx = np.nansum(rx * rx, axis=1) / nn - mx * mx
            vy = np.nansum(ry * ry, axis=1) / nn - my * my
            with np.errstate(invalid="ignore", divide="ignore"):
                ic = cov / np.sqrt(vx * vy)
            ic[nrow < MIN_N] = np.nan
            ics[:, j] = ic
        np.savez(CACHE / f"batch_{bi:02d}.npz", ic=ics)
        log(f"batch {bi + 1}/{nb} ({B} factors) {time.time()-t0:.0f}s"
            f"  [ic mean {np.nanmean(ics):.4f}]")
    log(f"all batches done in {time.time()-t_all:.0f}s")


def assemble():
    factors, M, dates = load_matrix()
    T, F = M.shape
    nb = (F + BATCH - 1) // BATCH
    IC = np.full((T, F), np.nan)
    for bi in range(nb):
        z = np.load(CACHE / f"batch_{bi:02d}.npz")
        a = z["ic"]
        b0 = bi * BATCH
        IC[:, b0:b0 + a.shape[1]] = a
    tser = pl.read_parquet(AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet",
                           columns=["trade_date"]).get_column("trade_date")
    out = pl.DataFrame({"trade_date": tser}).hstack(
        pl.DataFrame({f: IC[:, i] for i, f in enumerate(factors)}))
    out.write_parquet(ICMAT)
    log(f"assembled -> {ICMAT} ({out.shape})")


def verify():
    imm = pl.read_parquet(ICMAT)
    arr = imm.select([c for c in imm.columns if c != "trade_date"]).to_numpy()
    print("shape:", arr.shape)
    print("nan pct overall:", float(np.isnan(arr).mean()))
    m = np.nanmean(arr, axis=0)
    names = [c for c in imm.columns if c != "trade_date"]
    top = np.argsort(-np.where(np.isfinite(m), m, -1e9))[:10]
    print("top-10 mean IC:")
    for i in top:
        print(f"  {names[i]:24s} {m[i]:.4f}")
    print("ic range:", float(np.nanmin(arr)), float(np.nanmax(arr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.all:
        run_all()
    if args.assemble:
        assemble()
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
