"""Replay the last unattributed low-rho factors on the v34 input contract.

Check that talib_MININDEX / talib_MAXINDEX / v_ep match the v34 stored columns
EXACTLY when computed on the v34 input contract (frame built by
fidelity_check.build_panel). Exact match => the v4-vs-v34 rank-corr gap comes
from input data source differences only, not from the formula implementation.

Run:
    PYTHONPATH=... /usr/bin/python3.12 replay_remaining.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE / "factorlib_build"))
import fidelity_check as fc  # noqa: E402

TARGETS = [
    ("talib", "talib_MININDEX"),
    ("talib", "talib_MAXINDEX"),
    ("academic", "v_ep"),
]

panel, _ = fc.build_panel()
print("contract frame:", panel.shape)

v34 = pl.read_parquet(fc.V34, columns=["ts_code", "trade_date"] + [t[1] for t in TARGETS])

print()
print(f"{'factor':<22} {'n_both':>9} {'n_diff':>8} {'max_abs':>12}  verdict")
for family, col in TARGETS:
    mod = importlib.import_module(f"factorlib.{family}.{col}")
    vals = fc.sanitize_registry(mod.compute(panel)).alias("v")
    mine = panel.select(["stock_code", "trade_date"]).with_columns(vals)
    j = mine.rename({"stock_code": "ts_code"}).join(
        v34.select(["ts_code", "trade_date", col]).rename({col: "ref"}),
        on=["ts_code", "trade_date"], how="inner",
    )
    a, b = j["v"], j["ref"]
    both = a.is_not_null() & b.is_not_null()
    d = (a - b).abs().filter(both)
    fin = d.is_finite()
    d_fin = d.filter(fin)
    n_diff = int((d_fin > 1e-9).sum())
    mx = float(d_fin.max()) if d_fin.len() else 0.0
    verdict = "EXACT" if n_diff == 0 else "DIFF"
    print(f"{col:<22} {int(both.sum()):>9} {n_diff:>8} {mx:>12.3e}  {verdict}")
