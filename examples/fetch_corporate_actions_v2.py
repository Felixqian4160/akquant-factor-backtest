"""Fetch Tushare dividend/split data (RESUMABLE, incremental).

Appends raw rows to a growing CSV after each batch, so kills lose nothing.
Re-running skips stocks already present in the CSV.

Usage: python fetch_corporate_actions_v2.py [--max-stocks N]
"""
from __future__ import annotations
import argparse
import pathlib
import time

import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
OUTDIR = AKQ / "evidence" / "v34_adj_20261007"
RAW_CSV = OUTDIR / "corporate_actions_raw.csv"
TOKEN = pathlib.Path("/media/felix/f/quant/aurumq-rl/.qbot_token").read_text().strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--max-stocks", type=int, default=95)
args = parser.parse_args()

universe = sorted(pl.read_parquet(V34, columns=["ts_code"])["ts_code"].unique().to_list())

done = set()
if RAW_CSV.exists():
    prev = pd.read_csv(RAW_CSV, dtype={"ts_code": str})
    done = set(prev["ts_code"].unique())
    log(f"resume: {len(done)} stocks already fetched")

todo = [c for c in universe if c not in done][:args.max_stocks]
log(f"universe={len(universe)}, todo this run={len(todo)}")

import tushare as ts
pro = ts.pro_api(TOKEN)

buf = []
t0 = time.time()
for i, code in enumerate(todo, 1):
    for attempt in range(3):
        try:
            df = pro.dividend(ts_code=code)
            if df is not None and len(df):
                df = df.copy()
                df["fetch_ts_code"] = code
                buf.append(df)
            else:
                buf.append(pd.DataFrame({"fetch_ts_code": [code]}))  # mark as fetched (empty)
            break
        except Exception as e:
            if attempt == 2:
                log(f"  FAIL {code}: {str(e)[:80]}")
            else:
                time.sleep(2 + attempt * 2)
    time.sleep(0.35)
    if i % 30 == 0:
        log(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)")
        # incremental save
        if buf:
            chunk = pd.concat(buf, ignore_index=True)
            if RAW_CSV.exists():
                chunk.to_csv(RAW_CSV, mode="a", header=False, index=False)
            else:
                chunk.to_csv(RAW_CSV, mode="w", header=True, index=False)
            buf = []

if buf:
    chunk = pd.concat(buf, ignore_index=True)
    if RAW_CSV.exists():
        chunk.to_csv(RAW_CSV, mode="a", header=False, index=False)
    else:
        chunk.to_csv(RAW_CSV, mode="w", header=True, index=False)

n_done = pd.read_csv(RAW_CSV, dtype={"ts_code": str})["fetch_ts_code"].nunique() if RAW_CSV.exists() else 0
log(f"batch done in {time.time()-t0:.0f}s; total stocks fetched so far: {n_done}/{len(universe)}")
