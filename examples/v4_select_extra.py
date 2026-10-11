"""v4 扩展: 从 2010-2017 t 排名生成 3 因子 / 8 因子集合 (与 s1 同口径, 验证"5"是否甜点)."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "v34_dedup_20261010" / "matrix_v34_dedup_ADJ.parquet"
OUT = AKQ / "evidence" / "v4_five_factor_20261011"
FS = OUT / "run_factsets"
W0, W1 = "2010-01-01", "2017-12-31"


def main():
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    dates = np.array([str(d)[:10] for d in mat["trade_date"].to_list()])
    sel = (dates >= W0) & (dates <= W1)
    X = mat.select(factors).to_numpy().astype(np.float64)[sel]
    n = np.isfinite(X).sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    t = np.where(n > 2, mean / (std / np.sqrt(np.maximum(n, 1)) + 1e-12), np.nan)
    valid = n > 500
    order_t = sorted([f for f in factors if valid[factors.index(f)]],
                     key=lambda f: -(t[factors.index(f)] if np.isfinite(t[factors.index(f)]) else -1e9))
    n3 = order_t[:3]
    n8 = order_t[:8]
    for tag, fl in (("s1_n3", n3), ("s1_n8", n8)):
        (FS / f"{tag}.json").write_text(json.dumps(
            {"tag": tag, "factors": fl, "reason": f"2010-2017 t top-{len(fl)}"}, ensure_ascii=False, indent=1))
        print(tag, fl)
    # 记录到 factorsets.json
    p = json.loads((OUT / "factorsets.json").read_text())
    p["sets"]["s1_n3"] = {"reason": "t top-3 (个数验证)", "factors": n3,
                          "stats": {"per_factor": {f: round(float(t[factors.index(f)]), 2) for f in n3}}}
    p["sets"]["s1_n8"] = {"reason": "t top-8 (个数验证)", "factors": n8,
                          "stats": {"per_factor": {f: round(float(t[factors.index(f)]), 2) for f in n8}}}
    (OUT / "factorsets.json").write_text(json.dumps(p, ensure_ascii=False, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
