"""Adjustment Impact Study — Data Assembly

Build adjusted (hfq) price series for the 354 panel stocks from
aurumq-rl/data_cache/*_full.parquet, and save a compact long-form frame.

Output: evidence/audit_v33_20261007/adj_study/adj_prices_354.parquet
  columns: ts_code, trade_date, open, close, adj_factor, adj_open, adj_close
"""
import json
import pathlib
import time

import numpy as np
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
CACHE = pathlib.Path("/media/felix/f/quant/aurumq-rl/data_cache")
OUT = ROOT / "evidence" / "audit_v33_20261007" / "adj_study"
OUT.mkdir(parents=True, exist_ok=True)

PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# Panel universe
syms = pl.read_parquet(PANEL, columns=["ts_code"])["ts_code"].unique().to_list()
log(f"panel symbols: {len(syms)}")

# Map to cache files: 000001.SZ -> 000001_SZ_full.parquet
frames = []
missing = []
t0 = time.time()
for i, s in enumerate(syms, 1):
    fname = s.replace(".", "_") + "_full.parquet"
    fp = CACHE / fname
    if not fp.exists():
        missing.append(s)
        continue
    d = pl.read_parquet(fp, columns=["ts_code", "trade_date", "open", "close", "adj_factor"])
    frames.append(d)
    if i % 90 == 0:
        log(f"  loaded {i}/{len(syms)} ({time.time()-t0:.0f}s)")

log(f"loaded {len(frames)} frames; missing: {len(missing)}")
if missing:
    log(f"  missing: {missing[:10]}")

df = pl.concat(frames)
# Normalize date
df = df.with_columns(
    pl.col("trade_date").str.to_datetime("%Y%m%d").alias("trade_date")
).sort(["ts_code", "trade_date"])

# Adjusted series: hfq = raw × adj_factor
df = df.with_columns([
    (pl.col("open") * pl.col("adj_factor")).alias("adj_open"),
    (pl.col("close") * pl.col("adj_factor")).alias("adj_close"),
])

log(f"total rows: {df.height:,}")
log(f"date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

# Save
out_path = OUT / "adj_prices_354.parquet"
df.write_parquet(out_path)
log(f"saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

# Quick sanity: factor change counts per stock
chk = df.group_by("ts_code").agg([
    pl.col("adj_factor").n_unique().alias("n_factor_changes"),
])
arr = chk["n_factor_changes"].to_numpy()
log(f"adj_factor unique values per stock: min={arr.min()}, median={np.median(arr):.0f}, max={arr.max()}")

summary = {
    "n_symbols_loaded": len(frames),
    "n_missing": len(missing),
    "missing_list": missing,
    "n_rows": df.height,
    "date_min": str(df["trade_date"].min()),
    "date_max": str(df["trade_date"].max()),
    "factor_unique_median": float(np.median(arr)),
}
(OUT / "assembly_summary.json").write_text(json.dumps(summary, indent=2))
log("DONE")
