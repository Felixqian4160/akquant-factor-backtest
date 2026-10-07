"""V33 Comprehensive Audit — Module 4: Factor value correctness spot checks

Independent verification of factor values:
  A. Period-free TA-Lib factors (TYPPRICE, MEDPRICE, WCLPRICE, AVGPRICE, BOP)
  B. Period-based TA-Lib factors (RSI, NATR, MOM) — try standard periods
  C. GTJA formula check (gtja_gtja_002)
  D. Academic factor check (l_size vs log(circ_cap))
"""
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import talib

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
OUT = ROOT / "evidence" / "audit_v33_20261007"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

results = {}
S = "600519.SH"  # Moutai: full history, liquid

# Load needed columns for Moutai + a few stocks
need = ["trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount",
        "talib_TYPPRICE", "talib_MEDPRICE", "talib_WCLPRICE", "talib_AVGPRICE",
        "talib_BOP", "talib_RSI", "talib_NATR", "talib_MOM",
        "gtja_gtja_002", "l_size", "circ_cap"]
df = pl.read_parquet(PANEL, columns=need).filter(pl.col("ts_code") == S).sort("trade_date")
log(f"Loaded {S}: {df.height} rows, {df['trade_date'].min()} ~ {df['trade_date'].max()}")

pdf = df.to_pandas()

# ── A. Period-free TA-Lib factors ──
log("=== A. Period-free TA-Lib factors ===")
checks = {}

expected_typ = (pdf["high"] + pdf["low"] + pdf["close"]) / 3
diff = (pdf["talib_TYPPRICE"] - expected_typ).abs().max()
checks["TYPPRICE"] = diff

expected_med = (pdf["high"] + pdf["low"]) / 2
checks["MEDPRICE"] = (pdf["talib_MEDPRICE"] - expected_med).abs().max()

expected_wcl = (pdf["high"] + pdf["low"] + 2 * pdf["close"]) / 4
checks["WCLPRICE"] = (pdf["talib_WCLPRICE"] - expected_wcl).abs().max()

expected_avg = (pdf["open"] + pdf["high"] + pdf["low"] + pdf["close"]) / 4
checks["AVGPRICE"] = (pdf["talib_AVGPRICE"] - expected_avg).abs().max()

expected_bop = (pdf["close"] - pdf["open"]) / (pdf["high"] - pdf["low"])
checks["BOP"] = (pdf["talib_BOP"] - expected_bop).abs().max()

for k, v in checks.items():
    flag = "PASS" if v < 1e-8 else ("CHECK" if v < 1e-4 else "FAIL")
    log(f"  [{flag}] {k}: max_abs_diff={v:.2e}")
results["period_free"] = {k: float(v) for k, v in checks.items()}

# ── B. Period-based TA-Lib factors — find matching period ──
log("=== B. Period-based TA-Lib factors ===")

def best_period(predicate, periods, name, factor_col):
    best = None
    for p in periods:
        expected = predicate(p)
        valid = ~np.isnan(expected.values) & ~np.isnan(pdf[factor_col].values)
        if valid.sum() < 100:
            continue
        diff = np.abs(expected.values[valid] - pdf[factor_col].values[valid]).max()
        if best is None or diff < best[1]:
            best = (p, diff)
    if best:
        flag = "PASS" if best[1] < 1e-8 else ("CHECK" if best[1] < 1e-4 else "FAIL")
        log(f"  [{flag}] {name}: best_period={best[0]}, max_diff={best[1]:.2e}")
    else:
        log(f"  [SKIP] {name}: no valid comparison")
    return best

rsi_best = best_period(
    lambda p: pd.Series(talib.RSI(pdf["close"].values, timeperiod=p), index=pdf.index),
    [6, 12, 14, 24], "talib_RSI", "talib_RSI")
natr_best = best_period(
    lambda p: pd.Series(talib.NATR(pdf["high"].values, pdf["low"].values, pdf["close"].values, timeperiod=p), index=pdf.index),
    [5, 10, 14, 20, 30], "talib_NATR", "talib_NATR")
mom_best = best_period(
    lambda p: pd.Series(talib.MOM(pdf["close"].values, timeperiod=p), index=pdf.index),
    [5, 10, 12, 20, 30], "talib_MOM", "talib_MOM")

results["period_based"] = {
    "RSI": {"best_period": rsi_best[0], "max_diff": float(rsi_best[1])} if rsi_best else None,
    "NATR": {"best_period": natr_best[0], "max_diff": float(natr_best[1])} if natr_best else None,
    "MOM": {"best_period": mom_best[0], "max_diff": float(mom_best[1])} if mom_best else None,
}

# ── C. GTJA formula check: gtja_gtja_002 ──
# GTJA alpha#002 (per standard 191 list):
#   -1 * DELTA( (((CLOSE-LOW) - (HIGH-CLOSE)) / (HIGH-LOW)), 1 )
log("=== C. GTJA gtja_gtja_002 formula ===")
inner = ((pdf["close"] - pdf["low"]) - (pdf["high"] - pdf["close"])) / (pdf["high"] - pdf["low"])
expected_gtja2 = -1 * inner.diff(1)  # DELTA with period 1
valid = ~np.isnan(expected_gtja2.values) & ~np.isnan(pdf["gtja_gtja_002"].values) & np.isfinite(pdf["gtja_gtja_002"].values)
diff2 = np.abs(expected_gtja2.values[valid] - pdf["gtja_gtja_002"].values[valid]).max()
flag = "PASS" if diff2 < 1e-6 else ("CHECK" if diff2 < 1e-2 else "FAIL")
log(f"  [{flag}] gtja_gtja_002 (-1 × DELTA(inner, 1)): max_diff={diff2:.2e}, n_valid={valid.sum()}")
results["gtja_002"] = {"max_diff": float(diff2), "n_valid": int(valid.sum())}

# ── D. Academic: l_size vs log(circ_cap) ──
log("=== D. l_size vs log(circ_cap) ===")
for transform in ["log_circ", "log_circ_div_1e4", "log_circ_div_1e8"]:
    if transform == "log_circ":
        expected = np.log(pdf["circ_cap"])
    elif transform == "log_circ_div_1e4":
        expected = np.log(pdf["circ_cap"] / 1e4)
    else:
        expected = np.log(pdf["circ_cap"] / 1e8)
    valid = np.isfinite(expected.values) & np.isfinite(pdf["l_size"].values)
    if valid.sum() == 0:
        continue
    # correlation + mean diff (level shift possible)
    corr = np.corrcoef(expected.values[valid], pdf["l_size"].values[valid])[0, 1]
    meandiff = (expected.values[valid] - pdf["l_size"].values[valid]).mean()
    log(f"  {transform}: corr={corr:.6f}, mean_diff={meandiff:.4f}")
    if corr > 0.9999:
        results["l_size"] = {"transform": transform, "corr": float(corr), "mean_diff": float(meandiff)}

# ── E. N-factor sample check on l_ami (top factor) — just null rate + range ──
log("=== E. Top factor l_ami range check ===")
ami = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "l_ami"])
log(f"  l_ami: null={ami['l_ami'].null_count()}, min={ami['l_ami'].min()}, max={ami['l_ami'].max()}, mean={ami['l_ami'].mean()}")
results["l_ami"] = {
    "null": int(ami["l_ami"].null_count()),
    "min": float(ami["l_ami"].min()), "max": float(ami["l_ami"].max()),
    "mean": float(ami["l_ami"].mean()),
}

with (OUT / "audit_04_factor_values.json").open("w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved {OUT/'audit_04_factor_values.json'}")
log("DONE module 4")
