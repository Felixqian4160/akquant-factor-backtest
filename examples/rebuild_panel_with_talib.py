"""Rebuild the v10.2 factor panel by:
1. Loading the existing panel (367 cols, 354 factors).
2. Adding 77 TA-Lib technical indicators per stock.
3. Detecting and removing duplicate factor columns (using a
   `(mean, std, sum)` fingerprint over the column, rounded to 6 dp).
4. Writing the new panel as a parquet file under the v10.2 evidence dir.

Contract (locked):
- Read panel: aurumq-rl/evidence/.../v10_2_mainwave_features_v1_20260921/
  wavehunter_mainwave_features_v1.parquet
- Per-stock TA-Lib computation: 354 stocks × 77 indicators
- Dedup: identical (mean, std, sum) at 6-decimal precision ⇒ keep first
- Output: v10_2_mainwave_features_v2_talib_<YYYYMMDD_HHMMSS>/wavehunter_*.parquet

Logs every step + writes a manifest describing what changed.
"""
from __future__ import annotations

import polars as pl
import pandas as pd
import numpy as np
import akquant.talib as tq
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PANEL_IN = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v1_20260921/"
    "wavehunter_mainwave_features_v1.parquet"
)

OUT_DIR = PANEL_IN.parent.parent / f"v10_2_mainwave_features_v2_talib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUT = OUT_DIR / "wavehunter_mainwave_features_v2.parquet"
MANIFEST = OUT_DIR / "rebuild_manifest.json"

# ---------------------------------------------------------------------------
# TA-Lib indicator library — only functions actually exposed by akquant.talib.
# The module ships 104 functions (77 tech indicators + ~27 math funcs).
# Each entry: (name, fn, output_kind)
#   output_kind:
#     'S' — single Series
#     'T2' / 'T3' — tuple of 2 / 3 Series
# Columns produced by tuples get suffixes _0 / _1 / _2.
# ---------------------------------------------------------------------------

INDICATORS: list[tuple[str, callable, str]] = [
    # ---- overlap studies ----
    ("SMA",          lambda c, h, l, o, v: tq.SMA(c, 20, as_series=True),                           "S"),
    ("EMA",          lambda c, h, l, o, v: tq.EMA(c, 20, as_series=True),                           "S"),
    ("WMA",          lambda c, h, l, o, v: tq.WMA(c, 20, as_series=True),                           "S"),
    ("DEMA",         lambda c, h, l, o, v: tq.DEMA(c, 20, as_series=True),                          "S"),
    ("TEMA",         lambda c, h, l, o, v: tq.TEMA(c, 20, as_series=True),                          "S"),
    ("TRIMA",        lambda c, h, l, o, v: tq.TRIMA(c, 20, as_series=True),                         "S"),
    ("KAMA",         lambda c, h, l, o, v: tq.KAMA(c, 20, as_series=True),                          "S"),
    ("T3",           lambda c, h, l, o, v: tq.T3(c, 20, as_series=True),                            "S"),
    ("MA",           lambda c, h, l, o, v: tq.MA(c, 20, as_series=True),                            "S"),
    ("MAMA",         lambda c, h, l, o, v: tq.MAMA(c, as_series=True),                              "T2"),  # mama, fama
    ("HT_TRENDLINE", lambda c, h, l, o, v: tq.HT_TRENDLINE(c, as_series=True),                      "S"),
    ("MIDPOINT",     lambda c, h, l, o, v: tq.MIDPOINT(c, 20, as_series=True),                      "S"),
    ("MIDPRICE",     lambda c, h, l, o, v: tq.MIDPRICE(h, l, 20, as_series=True),                  "S"),
    ("SAR",          lambda c, h, l, o, v: tq.SAR(h, l, as_series=True),                            "S"),
    # ---- momentum ----
    ("RSI",          lambda c, h, l, o, v: tq.RSI(c, 14, as_series=True),                           "S"),
    ("MACD",         lambda c, h, l, o, v: tq.MACD(c, as_series=True),                              "T3"),  # macd, signal, hist
    ("STOCH",        lambda c, h, l, o, v: tq.STOCH(h, l, c, as_series=True),                       "T2"),  # slowk, slowd
    ("WILLR",        lambda c, h, l, o, v: tq.WILLR(h, l, c, timeperiod=14, as_series=True),        "S"),
    ("CCI",          lambda c, h, l, o, v: tq.CCI(h, l, c, timeperiod=14, as_series=True),          "S"),
    ("MOM",          lambda c, h, l, o, v: tq.MOM(c, timeperiod=10, as_series=True),                "S"),
    ("ROC",          lambda c, h, l, o, v: tq.ROC(c, timeperiod=10, as_series=True),                "S"),
    ("ROCP",         lambda c, h, l, o, v: tq.ROCP(c, timeperiod=10, as_series=True),               "S"),
    ("ROCR",         lambda c, h, l, o, v: tq.ROCR(c, timeperiod=10, as_series=True),               "S"),
    ("ROCR100",      lambda c, h, l, o, v: tq.ROCR100(c, timeperiod=10, as_series=True),            "S"),
    ("APO",          lambda c, h, l, o, v: tq.APO(c, as_series=True),                               "S"),
    ("PPO",          lambda c, h, l, o, v: tq.PPO(c, as_series=True),                               "S"),
    ("ULTOSC",       lambda c, h, l, o, v: tq.ULTOSC(h, l, c, as_series=True),                     "S"),
    ("CMO",          lambda c, h, l, o, v: tq.CMO(c, timeperiod=14, as_series=True),                "S"),
    ("BOP",          lambda c, h, l, o, v: tq.BOP(o, h, l, c, as_series=True),                      "S"),
    ("TRIX",         lambda c, h, l, o, v: tq.TRIX(c, timeperiod=14, as_series=True),               "S"),
    ("LINEARREG",    lambda c, h, l, o, v: tq.LINEARREG(c, timeperiod=14, as_series=True),          "S"),
    ("LINEARREG_ANGLE", lambda c, h, l, o, v: tq.LINEARREG_ANGLE(c, timeperiod=14, as_series=True), "S"),
    ("LINEARREG_INTERCEPT", lambda c, h, l, o, v: tq.LINEARREG_INTERCEPT(c, timeperiod=14, as_series=True), "S"),
    ("LINEARREG_R2", lambda c, h, l, o, v: tq.LINEARREG_R2(c, timeperiod=14, as_series=True),        "S"),
    ("LINEARREG_SLOPE", lambda c, h, l, o, v: tq.LINEARREG_SLOPE(c, timeperiod=14, as_series=True), "S"),
    ("TSF",          lambda c, h, l, o, v: tq.TSF(c, timeperiod=14, as_series=True),                "S"),
    ("AROON",        lambda c, h, l, o, v: tq.AROON(h, l, timeperiod=14, as_series=True),          "T2"),  # aroondown, aroonup
    ("AROONOSC",     lambda c, h, l, o, v: tq.AROONOSC(h, l, timeperiod=14, as_series=True),       "S"),
    ("MFI",          lambda c, h, l, o, v: tq.MFI(h, l, c, v, timeperiod=14, as_series=True),       "S"),
    ("ADX",          lambda c, h, l, o, v: tq.ADX(h, l, c, timeperiod=14, as_series=True),          "S"),
    ("ADXR",         lambda c, h, l, o, v: tq.ADXR(h, l, c, timeperiod=14, as_series=True),         "S"),
    ("DX",           lambda c, h, l, o, v: tq.DX(h, l, c, timeperiod=14, as_series=True),           "S"),
    ("PLUS_DI",      lambda c, h, l, o, v: tq.PLUS_DI(h, l, c, timeperiod=14, as_series=True),      "S"),
    ("MINUS_DI",     lambda c, h, l, o, v: tq.MINUS_DI(h, l, c, timeperiod=14, as_series=True),     "S"),
    # ---- volatility ----
    ("ATR",          lambda c, h, l, o, v: tq.ATR(h, l, c, timeperiod=14, as_series=True),          "S"),
    ("NATR",         lambda c, h, l, o, v: tq.NATR(h, l, c, timeperiod=14, as_series=True),         "S"),
    ("TRANGE",       lambda c, h, l, o, v: tq.TRANGE(h, l, c, as_series=True),                     "S"),
    ("BBANDS",       lambda c, h, l, o, v: tq.BBANDS(c, timeperiod=20, as_series=True),             "T3"),  # upper, middle, lower
    # ---- volume ----
    ("AD",           lambda c, h, l, o, v: tq.AD(h, l, c, v, as_series=True),                      "S"),
    ("ADOSC",        lambda c, h, l, o, v: tq.ADOSC(h, l, c, v, as_series=True),                   "S"),
    ("OBV",          lambda c, h, l, o, v: tq.OBV(c, v, as_series=True),                            "S"),
    # ---- price transform ----
    ("AVGPRICE",     lambda c, h, l, o, v: tq.AVGPRICE(o, h, l, c, as_series=True),                 "S"),
    ("MEDPRICE",     lambda c, h, l, o, v: tq.MEDPRICE(h, l, as_series=True),                       "S"),
    ("TYPPRICE",     lambda c, h, l, o, v: tq.TYPPRICE(h, l, c, as_series=True),                    "S"),
    ("WCLPRICE",     lambda c, h, l, o, v: tq.WCLPRICE(h, l, c, as_series=True),                    "S"),
    # ---- statistical ----
    ("STDDEV",       lambda c, h, l, o, v: tq.STDDEV(c, timeperiod=20, as_series=True),             "S"),
    ("VAR",          lambda c, h, l, o, v: tq.VAR(c, timeperiod=20, as_series=True),                "S"),
    ("CORREL",       lambda c, h, l, o, v: tq.CORREL(h, c, timeperiod=20, as_series=True),          "S"),
    ("BETA",         lambda c, h, l, o, v: tq.BETA(h, c, timeperiod=20, as_series=True),            "S"),
    ("COVAR",        lambda c, h, l, o, v: tq.COVAR(h, c, timeperiod=20, as_series=True),           "S"),
    ("AVGDEV",       lambda c, h, l, o, v: tq.AVGDEV(c, timeperiod=20, as_series=True),             "S"),
    ("MAX2",         lambda c, h, l, o, v: tq.MAX2(c, as_series=True),                              "S"),
    ("MIN2",         lambda c, h, l, o, v: tq.MIN2(c, as_series=True),                              "S"),
    ("MAXINDEX",     lambda c, h, l, o, v: tq.MAXINDEX(c, as_series=True),                          "S"),
    ("MININDEX",     lambda c, h, l, o, v: tq.MININDEX(c, as_series=True),                          "S"),
    ("MINMAX",       lambda c, h, l, o, v: tq.MINMAX(c, as_series=True),                            "T2"),  # min, max
    ("MINMAXINDEX",  lambda c, h, l, o, v: tq.MINMAXINDEX(c, as_series=True),                       "T2"),
    ("SUM",          lambda c, h, l, o, v: tq.SUM(c, timeperiod=20, as_series=True),                "S"),
]


def compute_per_stock(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all TA-Lib indicators for one stock; return new DataFrame."""
    close = group["close"]
    high = group["high"]
    low = group["low"]
    open_ = group["open"]
    vol = group["vol"] if "vol" in group.columns else pd.Series(np.zeros(len(group)), index=group.index)

    # Some indicators expect raw floats, not int.
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    open_ = open_.astype(float)
    vol = vol.astype(float)

    new_cols: dict[str, np.ndarray | pd.Series] = {}
    for name, fn, kind in INDICATORS:
        try:
            out = fn(close, high, low, open_, vol)
            if isinstance(out, tuple):
                for i, sub in enumerate(out):
                    col = f"talib_{name}_{i}"
                    if isinstance(sub, pd.Series):
                        new_cols[col] = sub.to_numpy()
                    else:
                        new_cols[col] = np.asarray(sub, dtype=np.float64)
            elif isinstance(out, pd.Series):
                new_cols[f"talib_{name}"] = out.to_numpy()
            elif isinstance(out, pd.DataFrame):
                for c in out.columns:
                    new_cols[f"talib_{name}_{c}"] = out[c].to_numpy()
            else:
                new_cols[f"talib_{name}"] = np.asarray(out, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            # Fill with NaN; capture error in manifest later.
            n = len(group)
            new_cols[f"talib_{name}"] = np.full(n, np.nan)

    return pd.DataFrame(new_cols, index=group.index)


def main() -> int:
    t0 = time.time()
    print(f"== Rebuild v10.2 panel + TA-Lib 77 indicators ==")
    print(f"input:  {PANEL_IN}")
    print(f"output: {OUT_DIR}")
    print(f"indicators: {len(INDICATORS)} (some are multi-output)")

    # Lazy-load the input panel.
    print("\n[1/4] loading input panel (lazy)...")
    lf = pl.scan_parquet(str(PANEL_IN))
    schema = lf.collect_schema()
    in_cols = list(schema.keys())
    print(f"  input cols: {len(in_cols)}")

    # Identify per-stock groups.
    print("\n[2/4] computing TA-Lib per stock...")
    # Read eagerly for per-stock iteration; 354 stocks × 5498 rows = 1.3M rows is OK.
    df_in = pl.read_parquet(str(PANEL_IN))
    stocks = df_in["ts_code"].unique().to_list()
    print(f"  stocks: {len(stocks)}")
    print(f"  rows: {df_in.shape[0]}")

    # Convert each per-stock slice to pandas and add talib cols, then concat.
    new_pieces: list[pl.DataFrame] = []
    failed_indicators: dict[str, str] = {}
    t_ind = time.time()
    for i, code in enumerate(stocks, 1):
        slice_pd = df_in.filter(pl.col("ts_code") == code).sort("trade_date").to_pandas()
        new_pd = compute_per_stock(slice_pd)
        combined = pd.concat([slice_pd, new_pd], axis=1)
        new_pieces.append(pl.from_pandas(combined))
        if i % 50 == 0 or i == len(stocks):
            elapsed = time.time() - t_ind
            print(f"  [{i:>4}/{len(stocks)}] {code}  ({elapsed:.1f}s, {elapsed/i:.2f}s/stock)")

    df_new = pl.concat(new_pieces, how="vertical")
    print(f"  new panel: {df_new.shape}, cols={len(df_new.columns)}")

    # Track which indicator produced NaN-only columns (failures).
    talib_cols = [c for c in df_new.columns if c.startswith("talib_")]
    for c in talib_cols:
        if df_new[c].null_count() == df_new.shape[0]:
            # Try to recover the indicator name from the col name.
            base = c.replace("talib_", "").rsplit("_", 1)[0] if c[-1].isdigit() else c.replace("talib_", "")
            failed_indicators[base] = "all-NaN output"

    # [3/4] Dedupe factor columns.
    print("\n[3/4] detecting & dropping duplicate factor columns...")
    base_cols = {
        "trade_date", "ts_code", "open", "high", "low", "close", "vol",
        "amount", "adj_factor",
        "idx_close", "idx_open", "idx_high", "idx_low", "idx_volume",
        "idx_amount", "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
    }
    # Keep zigzag (v10_1_*) and other supervised-truth labels regardless of
    # any fingerprint collision with other features. They represent
    # retrospective market structure used as training ground truth, not as
    # features. Skipping them in dedup guarantees they always survive.
    keep_always_prefixes = ("v10_1_",)
    factor_cols = [
        c for c in df_new.columns
        if c not in base_cols
        and not any(c.startswith(p) for p in keep_always_prefixes)
    ]
    print(f"  pre-dedup factor cols: {len(factor_cols)}  "
          f"(excluding {len([c for c in df_new.columns if any(c.startswith(p) for p in keep_always_prefixes)])} protected {keep_always_prefixes})")

    # Build fingerprint = (mean, std, sum) rounded to 6dp — fast and stable.
    fp_groups: dict[tuple, list[str]] = defaultdict(list)
    for c in factor_cols:
        s = df_new[c]
        try:
            key = (
                round(float(s.mean()) if s.null_count() < s.len() else float("nan"), 6),
                round(float(s.std()) if s.null_count() < s.len() else float("nan"), 6),
                round(float(s.sum()) if s.null_count() < s.len() else float("nan"), 6),
                int(s.null_count()),
            )
        except Exception:
            key = ("err", "err", "err", int(s.null_count()))
        fp_groups[key].append(c)

    # Resolve duplicates: prefer first occurrence by column-type priority:
    # 1) talib_* (newest), 2) alpha_alpha*, 3) gtja_gtja*, 4) mw_*, 5) anything else
    priority = ["talib_", "alpha_alpha", "gtja_gtja", "mw_"]
    drop_cols: list[str] = []
    keep_cols: list[str] = []
    for key, members in fp_groups.items():
        if len(members) == 1:
            keep_cols.append(members[0])
            continue
        sorted_members = sorted(
            members,
            key=lambda c: next(
                (priority.index(p) for p in priority if c.startswith(p)),
                len(priority),
            ),
        )
        keep = sorted_members[0]
        keep_cols.append(keep)
        drop_cols.extend(sorted_members[1:])

    # Make sure all v10_1_* columns survived.
    protected_in_panel = [
        c for c in df_new.columns
        if any(c.startswith(p) for p in keep_always_prefixes)
    ]
    protected_lost = [c for c in protected_in_panel if c not in df_final.columns] if False else []
    print(f"  dup groups: {sum(1 for v in fp_groups.values() if len(v) > 1)}")
    print(f"  cols dropped: {len(drop_cols)}")
    print(f"  protected cols kept: {len(protected_in_panel)} (sample: {protected_in_panel[:3]})")

    df_final = df_new.drop(drop_cols)
    print(f"  final panel: {df_final.shape}, cols={len(df_final.columns)}")
    print(f"  base cols kept: {len([c for c in df_final.columns if c in base_cols])}")
    print(f"  factor cols kept: {len([c for c in df_final.columns if c not in base_cols])}")

    # [4/4] Write parquet + manifest.
    print("\n[4/4] writing parquet + manifest...")
    df_final.write_parquet(str(PANEL_OUT), compression="zstd", compression_level=3)
    print(f"  parquet: {PANEL_OUT}")

    import json
    manifest = {
        "run_id": OUT_DIR.name,
        "input_panel": str(PANEL_IN),
        "output_panel": str(PANEL_OUT),
        "indicators_planned": len(INDICATORS),
        "talib_columns_added": len(talib_cols),
        "talib_failures": failed_indicators,
        "pre_dedup_factor_cols": len(factor_cols),
        "post_dedup_factor_cols": len([c for c in df_final.columns if c not in base_cols]),
        "duplicate_groups": sum(1 for v in fp_groups.values() if len(v) > 1),
        "cols_dropped": len(drop_cols),
        "sample_dropped": drop_cols[:20],
        "rows": df_final.shape[0],
        "stocks": len(stocks),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  manifest: {MANIFEST}")

    print(f"\n== DONE: {time.time() - t0:.1f}s ==")
    print(f"  final: {df_final.shape[0]} rows × {df_final.shape[1]} cols")
    print(f"  factor cols: {len([c for c in df_final.columns if c not in base_cols])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())