"""Fidelity extension for low-rho factors + raw-data diff quantification.

Part 1: run the low-rho factors on the v34 input contract (frame built by
fidelity_check.build_panel). If they match v34 EXACTLY here, their v4-vs-v34
divergence is attributable to input data source differences, not to the
formula implementation.

Part 2: quantify raw data differences between stockdb (fsdb) and Tushare (v34)
on matched rows: close/volume/amount relative diffs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE / "factorlib_build"))
import fidelity_check as fc  # noqa: E402

LOW_FACTORS = [
    # alpha
    "alpha_alpha015", "alpha_alpha096", "alpha_alpha050", "alpha_alpha027",
    "alpha_alpha004", "alpha_alpha016", "alpha_alpha068", "alpha_alpha081",
    "alpha_alpha093", "alpha_alpha013", "alpha_alpha003", "alpha_alpha092",
    "alpha_alpha074", "alpha_alpha075", "alpha_alpha091", "alpha_alpha076",
    "alpha_alpha082", "alpha_alpha062", "alpha_alpha098", "alpha_alpha069",
    "alpha_alpha078", "alpha_alpha097", "alpha_alpha088", "alpha_alpha089",
    # gtja
    "gtja_gtja_032", "gtja_gtja_064", "gtja_gtja_143", "gtja_gtja_090",
    # NOTE (2026-10-10): gtja_016 / gtja_083 / gtja_099 / gtja_105 removed —
    # bit-exact duplicates of alpha_003 / alpha_016 / alpha_013 / alpha_011
    # (see evidence/audit_lookahead_20261010/v34_exact_dups.json).
    "gtja_gtja_036", "gtja_gtja_140", "gtja_gtja_101", "gtja_gtja_141",
    "gtja_gtja_074", "gtja_gtja_130", "gtja_gtja_119", "gtja_gtja_138",
    "gtja_gtja_179",
    # academic
    "p_pr", "v_ep",
]

panel, _ = fc.build_panel()
print("contract frame:", panel.shape)

# add industry (v34 build used stock_basic_industry.parquet; alias to sub_industry)
ind = pl.read_parquet(
    "/media/felix/f/quant/aurumq-rl/data_cache/stock_basic_industry.parquet"
).rename({"ts_code": "stock_code"})
panel = panel.join(ind, on="stock_code", how="left")
panel = panel.with_columns(pl.col("industry").alias("sub_industry"))
print("industry nulls:", panel["industry"].null_count())

# v34 stored columns for these factors
cols34 = [c for c in LOW_FACTORS if c not in ("p_pr", "v_ep")]
v34s = pl.read_parquet(fc.V34, columns=["ts_code", "trade_date"] + cols34 + ["p_pr", "v_ep"])

import importlib

MODULE = {
    "alpha_alpha015": ("alpha", "alpha015"),
    "alpha_alpha096": ("alpha", "alpha096"),
    "alpha_alpha050": ("alpha", "alpha050"),
    "alpha_alpha027": ("alpha", "alpha027"),
    "alpha_alpha004": ("alpha", "alpha004"),
    "alpha_alpha016": ("alpha", "alpha016"),
    "alpha_alpha068": ("alpha", "alpha068"),
    "alpha_alpha081": ("alpha", "alpha081"),
    "alpha_alpha093": ("alpha", "alpha093"),
    "alpha_alpha013": ("alpha", "alpha013"),
    "alpha_alpha003": ("alpha", "alpha003"),
    "alpha_alpha092": ("alpha", "alpha092"),
    "alpha_alpha074": ("alpha", "alpha074"),
    "alpha_alpha075": ("alpha", "alpha075"),
    "alpha_alpha091": ("alpha", "alpha091"),
    "alpha_alpha076": ("alpha", "alpha076"),
    "alpha_alpha082": ("alpha", "alpha082"),
    "alpha_alpha062": ("alpha", "alpha062"),
    "alpha_alpha098": ("alpha", "alpha098"),
    "alpha_alpha069": ("alpha", "alpha069"),
    "alpha_alpha078": ("alpha", "alpha078"),
    "alpha_alpha097": ("alpha", "alpha097"),
    "alpha_alpha088": ("alpha", "alpha088"),
    "alpha_alpha089": ("alpha", "alpha089"),
    "gtja_gtja_032": ("gtja", "gtja_032"),
    "gtja_gtja_064": ("gtja", "gtja_064"),
    "gtja_gtja_143": ("gtja", "gtja_143"),
    "gtja_gtja_090": ("gtja", "gtja_090"),
    "gtja_gtja_036": ("gtja", "gtja_036"),
    "gtja_gtja_140": ("gtja", "gtja_140"),
    "gtja_gtja_101": ("gtja", "gtja_101"),
    "gtja_gtja_141": ("gtja", "gtja_141"),
    "gtja_gtja_074": ("gtja", "gtja_074"),
    "gtja_gtja_130": ("gtja", "gtja_130"),
    "gtja_gtja_119": ("gtja", "gtja_119"),
    "gtja_gtja_138": ("gtja", "gtja_138"),
    "gtja_gtja_179": ("gtja", "gtja_179"),
    "p_pr": ("academic", "p_pr"),
    "v_ep": ("academic", "v_ep"),
}
# 18 industry-neutral alphas (verified to call ind_neutralize)
IND_ALPHAS = ["alpha048", "alpha058", "alpha059", "alpha063", "alpha067", "alpha069",
              "alpha070", "alpha076", "alpha079", "alpha080", "alpha082", "alpha087",
              "alpha089", "alpha090", "alpha091", "alpha093", "alpha097", "alpha100"]

print()
print("=== Part 1: replay on v34 contract (exact-match test) ===")
print(f"{'factor':<22} {'n_both':>9} {'n_diff>1e-6':>12} {'max_abs':>12}  verdict")
for col in LOW_FACTORS:
    family, module = MODULE[col]
    mod = importlib.import_module(f"factorlib.{family}.{module}")
    vals = fc.sanitize_registry(mod.compute(panel)).alias("v")
    mine = panel.select(["stock_code", "trade_date"]).with_columns(vals)
    j = mine.rename({"stock_code": "ts_code"}).join(
        v34s.select(["ts_code", "trade_date", col]).rename({col: "ref"}),
        on=["ts_code", "trade_date"], how="inner",
    )
    a, b = j["v"], j["ref"]
    both = a.is_not_null() & b.is_not_null()
    d = (a.filter(both) - b.filter(both)).abs()
    fin = d.is_finite()
    n_gt = int((d.filter(fin) > 1e-6).sum())
    mx = float(d.filter(fin).max()) if fin.any() else None
    verdict = "EXACT" if n_gt == 0 else f"{n_gt} diff"
    print(f"{col:<22} {int(both.sum()):>9} {n_gt:>12} {str(mx)[:12]:>12}  {verdict}")

print()
print("=== Part 1b: industry-neutral alphas on v34 contract ===")
for stem in IND_ALPHAS:
    col = f"alpha_{stem}"
    mod = importlib.import_module(f"factorlib.alpha.{stem}")
    vals = fc.sanitize_registry(mod.compute(panel)).alias("v")
    mine = panel.select(["stock_code", "trade_date"]).with_columns(vals)
    ref = pl.read_parquet(fc.V34, columns=["ts_code", "trade_date", col])
    j = mine.rename({"stock_code": "ts_code"}).join(
        ref.rename({col: "ref"}), on=["ts_code", "trade_date"], how="inner"
    )
    a, b = j["v"], j["ref"]
    both = a.is_not_null() & b.is_not_null()
    d = (a.filter(both) - b.filter(both)).abs()
    fin = d.is_finite()
    n_gt = int((d.filter(fin) > 1e-6).sum())
    print(f"{col:<22} {int(both.sum()):>9} {n_gt:>12} n_diff>1e-6")

print()
print("=== Part 2: raw data diff (stockdb vs Tushare on matched rows) ===")
f34 = pl.read_parquet(fc.V34, columns=["ts_code", "trade_date", "close", "vol", "amount"])
f34 = f34.with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")
ff = pl.read_parquet(fc.V34.parent.parent / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet",
                    columns=["ts_code", "trade_date", "close", "volume", "amount"])
ff = ff.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d")).drop("trade_date")
jj = f34.join(ff, on=["ts_code", "d"], how="inner", suffix="_f")
# unit conversions: v34 vol=手 vs fsdb shares (/100); v34 amount=千元 vs fsdb 元 (/1000)
PAIRS = (
    ("close", "close_f", 1.0),
    ("vol", "volume", 1.0 / 100),
    ("amount", "amount_f", 1.0 / 1000),
)
for col, other, scale in PAIRS:
    a = jj[col].to_numpy().astype(float)
    b = jj[other].to_numpy().astype(float) * scale
    ok = (a != 0) & np.isfinite(a) & np.isfinite(b)
    rel = np.abs(a[ok] - b[ok]) / np.abs(a[ok])
    print(f"{col:<8} matched={int(ok.sum())}  median_rel_diff={np.median(rel):.2e}  "
          f"p99={np.percentile(rel, 99):.2e}  frac>1%={float((rel > 0.01).mean()):.4f}")
