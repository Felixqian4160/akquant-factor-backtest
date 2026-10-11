"""v5 动态选择池: 2010-17 t top-N 作为每期选因子的候选池."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
OUT = AKQ / "evidence" / "v5_dynamic_select_20261011"
FS = OUT / "run_factsets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()
    FS.mkdir(parents=True, exist_ok=True)
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    dates = np.array([str(d)[:10] for d in mat["trade_date"].to_list()])
    sel = np.logical_and(dates >= "2010-01-01", dates <= "2017-12-31")
    X = mat.select(factors).to_numpy().astype(np.float64)[sel]
    n = np.isfinite(X).sum(0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(X, 0)
        std = np.nanstd(X, 0)
    t = np.where(n > 2, mean / (std / np.sqrt(np.maximum(n, 1)) + 1e-12), np.nan)
    valid = n > 500
    order = sorted([f for f in factors if valid[factors.index(f)]],
                   key=lambda f: -(t[factors.index(f)] if np.isfinite(t[factors.index(f)]) else -1e9))
    pool = order[: args.topn]
    (FS / f"v5_pool{args.topn}.json").write_text(json.dumps(
        {"tag": f"v5_pool{args.topn}", "factors": pool,
         "reason": f"2010-17 t top-{args.topn} (动态选择池)"},
        ensure_ascii=False, indent=1))
    print(f"pool{args.topn}:", pool[:25], "..." if len(pool) > 25 else "")


if __name__ == "__main__":
    main()
