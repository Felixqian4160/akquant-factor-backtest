"""Diagnose gtja_191's 137 boundary-diff rows (max_abs=0.0 but n_gt=137)."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "factorlib_build"))

import fidelity_check as fc

panel, outs = fc.build_panel()
mod = __import__("factorlib.gtja.gtja_191", fromlist=["compute"])
vals = fc.sanitize_registry(mod.compute(panel))
mine = panel.select(["stock_code", "trade_date"]).with_columns(vals.alias("mine"))
ref = (outs.rename({"ts_code": "stock_code"})
       .select(["stock_code", "trade_date", "gtja_gtja_191"]).rename({"gtja_gtja_191": "ref"}))
joined = mine.join(ref, on=["stock_code", "trade_date"], how="inner")
a, b = joined["mine"], joined["ref"]

print("total rows:", joined.height)
print("mine nulls:", a.null_count(), " ref nulls:", b.null_count())
print("mine nan:", a.is_nan().sum(), " ref nan:", b.is_nan().sum())

both = a.is_not_null() & b.is_not_null()
diff = (a.filter(both) - b.filter(both)).abs()
print("n_both:", diff.len())
print("nan in diff:", diff.is_nan().sum())
print("inf in diff:", diff.is_infinite().sum())
finite = diff.is_finite()
print("gt 1e-6 (all):", (diff > 1e-6).sum())
print("gt 1e-6 (finite only):", (diff.filter(finite) > 1e-6).sum(), "of", finite.sum())

d = joined.with_columns((pl.col("mine") - pl.col("ref")).abs().alias("d"))
flagged = d.filter(pl.col("d").is_nan() | (pl.col("d") > 1e-6).fill_null(False))
print("flagged rows:", flagged.height)
print("mine nan & ref nan:", flagged.filter(pl.col("mine").is_nan() & pl.col("ref").is_nan()).height)
print("mine nan & ref num:", flagged.filter(pl.col("mine").is_nan() & pl.col("ref").is_not_nan() & pl.col("ref").is_not_null()).height)
print("mine num & ref nan:", flagged.filter(pl.col("mine").is_not_nan() & pl.col("mine").is_not_null() & pl.col("ref").is_nan()).height)
print("both num & d>1e-6:", flagged.filter(pl.col("mine").is_not_nan() & pl.col("ref").is_not_nan() & (pl.col("d") > 1e-6)).height)
print(flagged.head(8).select(["stock_code", "trade_date", "mine", "ref", "d"]))
