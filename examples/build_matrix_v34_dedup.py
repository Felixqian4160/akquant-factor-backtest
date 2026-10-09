"""在去重面板上重建因子矩阵 (413 factors) — 合同与 stage_a_1.compute_matrix 完全一致。

合同（复刻 stage_a_1 的 ADJ 口径）:
  fwd_adj[t] = adj_close[t+21] / adj_open[t+1] - 1 - 0.005
  diff[f,d]  = mean(top10 by f@d 的 fwd) - mean(bot10 by f@d 的 fwd)
               (rank ordinal desc, 需 f 与 fwd 均 finite, sum/10 口径)
输入: data/wavehunter_hs300_v34_dedup_20261010.parquet (446 cols, 413 voting factors)
输出: evidence/v34_dedup_20261010/matrix_v34_dedup_ADJ.parquet (dates x 413)
分片: evidence/v34_dedup_20261010/_cache/batch_{s}_{e}.parquet (可断点续跑)

用法:
  python3.12 -u examples/build_matrix_v34_dedup.py --batch S N
  python3.12 -u examples/build_matrix_v34_dedup.py --assemble
  python3.12 -u examples/build_matrix_v34_dedup.py --verify
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

AKQ = Path("/media/felix/f/quant/akquant-factor-backtest")
SRC = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
OUTDIR = AKQ / "evidence" / "v34_dedup_20261010"
CACHE = OUTDIR / "_cache"
MATRIX = OUTDIR / "matrix_v34_dedup_ADJ.parquet"
OLD_MATRIX = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"

BASE_COLS = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount",
    "pct_chg", "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
    "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60", "turnover_rate",
    "circ_cap", "cap", "symbol", "volume",
    "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
}
ZIGZAG = {
    "v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
    "v10_1_b1_start", "v10_1_b1_interval",
    "v10_1_down_start", "v10_1_down_interval",
    "v10_1_peak_zone", "v10_1_valley_zone",
    "v10_1_zig_peak", "v10_1_zig_valley",
}
ROUND_TRIP_COST = 0.005
SHIFT = 21


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def factor_list() -> list[str]:
    cols = list(pl.read_parquet_schema(SRC).keys())
    return [c for c in cols if c not in (BASE_COLS | ZIGZAG)]


def run_batch(start: int, count: int) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    factors = factor_list()
    end = min(start + count, len(factors))
    todo = factors[start:end]
    out_path = CACHE / f"batch_{start:04d}_{end:04d}.parquet"
    if out_path.exists():
        log(f"batch {start}:{end} exists, skip")
        return
    log(f"batch {start}:{end} ({len(todo)} factors)")
    t0 = time.time()
    pf = (
        pl.scan_parquet(SRC)
        .select(["trade_date", "ts_code", "adj_open", "adj_close"] + todo)
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("adj_open").shift(-1).over("ts_code").alias("_o1"),
            pl.col("adj_close").shift(-SHIFT).over("ts_code").alias("_c21"),
        ])
        .with_columns((pl.col("_c21") / pl.col("_o1") - 1.0 - ROUND_TRIP_COST).alias("_fwd"))
        .select(["trade_date"] + todo + ["_fwd"])
        .collect()
    )
    log(f"  frame loaded {pf.shape} in {time.time()-t0:.0f}s")
    base = pf.select("trade_date").unique().sort("trade_date")
    cols = []
    t1 = time.time()
    for fi, f in enumerate(todo, 1):
        d = pf.select(["trade_date", "_fwd", f]).filter(pl.col(f).is_finite())
        d = d.with_columns(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk")
        )
        agg = d.group_by("trade_date").agg([
            (pl.col("_fwd").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top"),
            (pl.col("_fwd").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot"),
        ]).select(["trade_date", (pl.col("_top") - pl.col("_bot")).alias(f)])
        j = base.join(agg, on="trade_date", how="left")
        cols.append(j[f].alias(f))
        if fi % 25 == 0:
            log(f"  {fi}/{len(todo)} ({time.time()-t1:.0f}s)")
    m = base.with_columns(cols)
    m.write_parquet(out_path)
    log(f"  saved {out_path.name}: {m.shape} in {time.time()-t0:.0f}s")


def assemble() -> None:
    batches = sorted(CACHE.glob("batch_*.parquet"))
    if not batches:
        raise SystemExit("no batches")
    base = None
    frames = []
    for p in batches:
        b = pl.read_parquet(p)
        if base is None:
            base = b.select("trade_date")
            frames.append(b)
        else:
            assert b.select("trade_date").equals(base), f"date order mismatch: {p}"
            frames.append(b.drop("trade_date"))
    m = pl.concat(frames, how="horizontal")
    m.write_parquet(MATRIX)
    log(f"assembled: {m.shape} -> {MATRIX} ({MATRIX.stat().st_size/1e6:.1f} MB)")


def verify() -> None:
    mat = pl.read_parquet(MATRIX)
    old = pl.read_parquet(OLD_MATRIX)
    factors = [c for c in mat.columns if c != "trade_date"]
    removed = [c for c in old.columns if c != "trade_date" and c not in factors]
    log(f"new: {mat.shape}; old: {old.shape}; removed absent: {len(removed)}")

    assert mat.height == old.height, "row count mismatch"
    assert mat.select("trade_date").equals(old.select("trade_date")), "date mismatch"
    assert len(factors) == 413, f"expected 413 factors, got {len(factors)}"

    stats = []
    max_all = 0.0
    for f in factors:
        a = mat[f].cast(pl.Float64).to_numpy()
        b = old[f].cast(pl.Float64).to_numpy()
        nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
        both = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        max_all = max(max_all, d)
        stats.append({"factor": f, "max_abs_diff": d, "nan_pattern_equal": bool(nan_ok),
                      "n_finite": int(both.sum())})
    n_exact = sum(1 for s in stats if s["max_abs_diff"] == 0.0 and s["nan_pattern_equal"])
    bad = [s for s in stats if not (s["max_abs_diff"] == 0.0 and s["nan_pattern_equal"])]
    log(f"exact match vs old matrix: {n_exact}/{len(factors)}; global max|diff|={max_all}")
    for s in bad[:10]:
        log(f"  DIFF {s['factor']}: max={s['max_abs_diff']:.2e} nan_ok={s['nan_pattern_equal']}")

    # quick "look at data": factor means over 2010-2025 (settled rows)
    dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    sel = np.array([("2010-01-01" <= d <= "2025-12-31") for d in dates])
    M = mat.select(factors).to_numpy().astype(np.float64)[sel]
    with np.errstate(all="ignore"):
        means = np.nanmean(M, axis=0)
    order = np.argsort(-means)
    log("top-10 factors by full-period mean (2010-2025):")
    for k in order[:10]:
        log(f"  {factors[k]:<30s} {means[k]:+.4f}")

    out = {"new_matrix": str(MATRIX), "old_matrix": str(OLD_MATRIX),
           "n_factors": len(factors), "n_dates": mat.height,
           "removed_insent": len(removed),
           "n_exact_match": n_exact, "global_max_abs_diff": max_all,
           "per_factor": stats,
           "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUTDIR / "matrix_verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"verification saved: {OUTDIR/'matrix_verification.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", nargs=2, type=int, metavar=("S", "N"))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.batch:
        run_batch(args.batch[0], args.batch[1])
    elif args.assemble:
        assemble()
    elif args.verify:
        verify()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
