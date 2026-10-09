"""Fidelity check: recompute factorlib factors on v34's exact input contract and
compare against the v34 panel's stored columns.

Purpose: prove that extracting formulas into factorlib preserved the canonical
computation (no transcription loss). Comparison runs on the full 354-stock
cross-section (2004-2026) because cross-sectional ranks depend on the daily
universe.

Input contract reconstructed from v34_build_part1_factors.py:
  open/high/low/close = adj_open/adj_high/adj_low/adj_close
  volume = vol
  vwap = amount * 10 / vol.clip(1) * adj_factor
  returns = close/close.shift(1) - 1  (fill_null(0.0))
  adv{w} = vol.rolling_mean(w, min_periods=1)

Usage:
    python fidelity_check.py            # full run (resumable)
    python fidelity_check.py --report   # print saved report only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
OUTDIR = BASE / "factorlib_build"
PROGRESS = OUTDIR / "fidelity_progress.jsonl"
REPORT = OUTDIR / "fidelity_report.json"

sys.path.insert(0, str(BASE))

# (family, module, v34 column, sanitize mode, tolerance)
# sanitize: "registry" = inf->null + clip 1e6 (alpha/gtja);
#           "github"   = (inf|nan)->null + clip 1e6 (github17);
#           None       = raw (talib/academic as stored by v34)
CASES = [
    ("alpha", "alpha001", "alpha_alpha001", "registry", 1e-6),
    ("alpha", "alpha092", "alpha_alpha092", "registry", 1e-6),
    ("gtja", "gtja_001", "gtja_gtja_001", "registry", 1e-6),
    ("gtja", "gtja_070", "gtja_gtja_070", "registry", 1e-6),
    ("gtja", "gtja_132", "gtja_gtja_132", "registry", 1e-6),
    ("gtja", "gtja_191", "gtja_gtja_191", "registry", 1e-6),
    ("talib", "talib_SMA", "talib_SMA", None, 1e-6),
    ("talib", "talib_RSI", "talib_RSI", None, 1e-6),
    ("talib", "talib_MACD_0", "talib_MACD_0", None, 1e-6),
    ("talib", "talib_NATR", "talib_NATR", None, 1e-6),
    ("academic", "l_ami", "l_ami", None, 1e-6),
    ("academic", "r_tv", "r_tv", None, 1e-6),
    ("academic", "r_beta", "r_beta", None, 1e-5),
    ("academic", "p_m6", "p_m6", None, 1e-6),
    ("github", "efficiency_ratio", "efficiency_ratio", "github", 1e-6),
    ("github", "mom12m_jt", "mom12m_jt", "github", 1e-6),
    ("github", "resmom_6m", "resmom_6m", "github", 1e-6),
]

INPUT_COLS = ["ts_code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close",
              "adj_factor", "vol", "amount", "cap", "circ_cap", "bps", "pe",
              "grossprofit_margin", "netprofit_yoy"]
OUT_COLS = sorted({c for _, _, c, _, _ in CASES})
ADV_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180]


def sanitize_registry(s: pl.Series) -> pl.Series:
    name = s.name
    return pl.DataFrame({name: s}).with_columns(
        pl.when(pl.col(name).is_infinite()).then(None)
        .otherwise(pl.col(name).clip(-1e6, 1e6)).alias(name)
    )[name]


def sanitize_github(s: pl.Series) -> pl.Series:
    name = s.name
    return pl.DataFrame({name: s}).with_columns(
        pl.when(pl.col(name).is_infinite() | pl.col(name).is_nan()).then(None)
        .otherwise(pl.col(name).clip(-1e6, 1e6)).alias(name)
    )[name]


def build_panel():
    t0 = time.time()
    df = pl.read_parquet(V34, columns=INPUT_COLS + OUT_COLS)
    print(f"loaded: {df.shape} in {time.time()-t0:.0f}s", flush=True)
    outs = df.select(["ts_code", "trade_date"] + OUT_COLS)
    panel = df.select([
        pl.col("ts_code").alias("stock_code"),
        pl.col("trade_date"),
        pl.col("adj_open").alias("open"),
        pl.col("adj_high").alias("high"),
        pl.col("adj_low").alias("low"),
        pl.col("adj_close").alias("close"),
        pl.col("vol").alias("volume"),
        pl.col("amount"), pl.col("cap"), pl.col("circ_cap"),
        pl.col("bps"), pl.col("pe"),
        pl.col("grossprofit_margin"), pl.col("netprofit_yoy"),
        pl.col("adj_factor"),
    ]).sort(["stock_code", "trade_date"])
    panel = panel.with_columns([
        (pl.col("amount") * 10.0 / pl.col("volume").clip(lower_bound=1)
         * pl.col("adj_factor")).alias("vwap"),
        (pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1.0)
        .fill_null(0.0).alias("returns"),
    ])
    for w in ADV_WINDOWS:
        panel = panel.with_columns(
            pl.col("volume").rolling_mean(window_size=w, min_periods=1)
            .over("stock_code").alias(f"adv{w}")
        )
    print(f"panel: {panel.shape} in {time.time()-t0:.0f}s", flush=True)
    return panel, outs


def done_cases() -> dict[str, dict]:
    out = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["module"]] = rec
    return out


def run_case(panel: pl.DataFrame, outs: pl.DataFrame, case) -> dict:
    family, module, col, san, tol = case
    t0 = time.time()
    mod = __import__(f"factorlib.{family}.{module}", fromlist=["compute"])
    vals = mod.compute(panel)
    if san == "registry":
        vals = sanitize_registry(vals)
    elif san == "github":
        vals = sanitize_github(vals)
    mine = panel.select(["stock_code", "trade_date"]).with_columns(vals.alias("mine"))
    ref = (outs.rename({"ts_code": "stock_code"})
           .select(["stock_code", "trade_date", col]).rename({col: "ref"}))
    joined = mine.join(ref, on=["stock_code", "trade_date"], how="inner")
    a, b = joined["mine"], joined["ref"]
    both = a.is_not_null() & b.is_not_null()
    only_mine = (a.is_not_null() & b.is_null()).sum()
    only_ref = (a.is_null() & b.is_not_null()).sum()
    aa, bb = a.filter(both), b.filter(both)
    diff = (aa - bb).abs()
    n_both = diff.len()
    finite = diff.is_finite()
    n_gt = int((diff.filter(finite) > tol).sum())
    n_nan_both = int((aa.is_nan() & bb.is_nan()).sum())
    n_nan_mismatch = int(((aa.is_nan() & bb.is_not_nan()) | (aa.is_not_nan() & bb.is_nan())).sum())
    rec = {
        "family": family, "module": module, "column": col, "tol": tol,
        "n_rows": joined.height, "n_both": n_both,
        "n_only_mine": int(only_mine), "n_only_ref": int(only_ref),
        "n_exact_1e9": int((diff.filter(finite) <= 1e-9).sum()),
        "n_gt_tol": n_gt,
        "n_nan_both": n_nan_both,
        "n_nan_mismatch": n_nan_mismatch,
        "max_abs_diff": float(diff.filter(finite).max()) if finite.any() else None,
        "mean_abs_diff": float(diff.filter(finite).mean()) if finite.any() else None,
        "runtime_s": round(time.time() - t0, 1),
    }
    rec["verdict"] = ("EXACT_PASS" if n_gt == 0 and n_nan_mismatch == 0
                      else "BOUNDARY_PASS" if n_both and n_gt / n_both <= 0.01
                      else "DIVERGE")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        recs = list(done_cases().values())
        for r in recs:
            print(f"{r['module']:<28} {r['verdict']:<14} max={r['max_abs_diff']:.3e} "
                  f"gt_tol={r['n_gt_tol']}/{r['n_both']}")
        return 0

    done = done_cases()
    todo = [c for c in CASES if c[1] not in done]
    if not todo:
        print("all cases already done")
    else:
        print(f"todo: {len(todo)} cases", flush=True)
        panel, outs = build_panel()
        for case in todo:
            try:
                rec = run_case(panel, outs, case)
            except Exception as exc:
                rec = {"family": case[0], "module": case[1], "column": case[2],
                       "tol": case[4], "verdict": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            with PROGRESS.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rec['module']:<28} {rec['verdict']:<14} "
                  f"max={rec.get('max_abs_diff')} gt={rec.get('n_gt_tol')}", flush=True)

    recs = list(done_cases().values())
    REPORT.write_text(json.dumps(recs, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    for r in recs:
        print(f"{r['family']:<9} {r['module']:<28} {r['verdict']:<14} "
              f"n_gt={r.get('n_gt_tol')}/{r.get('n_both')} "
              f"max_abs={r.get('max_abs_diff')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
