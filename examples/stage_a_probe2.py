"""Probe v2: compare matrix_v34_ADJ diffs vs builder-style v34 diffs — direct value-level.

For the 20 sessions before a probe date: compute diffs via the EXACT builder code
(v34 panel, adj fwd, both filters), then print side-by-side vs matrix for factors.
"""
import pathlib
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
ENTRY_TO_EXIT_SHIFT = 21
ROUND_TRIP_COST = 0.005
LOOKBACK = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


mat = pl.read_parquet(MAT)
factors = [c for c in mat.columns if c != "trade_date"]
log(f"factors from matrix: {len(factors)}")
assert len(factors) == 428

# ── builder-style diffs (exact replication) ──
panel_full = (
    pl.scan_parquet(V34)
    .select(["trade_date", "ts_code", "adj_open", "adj_close"] + factors)
    .sort(["ts_code", "trade_date"])
    .with_columns([
        pl.col("adj_open").shift(-1).over("ts_code").alias("_open_t1"),
        pl.col("adj_close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
    ])
    .with_columns((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_net"))
    .select(["trade_date", "ts_code", "_fwd_net"] + factors)
    .collect()
)
log(f"panel_full: {panel_full.shape}")

all_dates = [str(x)[:10] for x in panel_full["trade_date"].unique().sort().to_list()]

# probe two windows: early (2010) and mid (2015-06)
for probe in ["2010-01-06", "2015-06-01"]:
    i0 = all_dates.index(probe)
    pds = all_dates[i0 - LOOKBACK:i0]
    sub = panel_full.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(pds))

    check_factors = ["gtja_gtja_002", "l_ami", "talib_RSI", "alpha_alpha042", "mom12m_jt"]
    print(f"\n===== probe {probe}, window {pds[0]}..{pds[-1]} =====")
    for f in check_factors:
        d = sub.select(["trade_date", "_fwd_net", f])
        d = d.filter(pl.col(f).is_finite() & pl.col("_fwd_net").is_finite())
        d = d.with_columns(pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk"))
        agg = d.group_by("trade_date").agg([
            (pl.col("_fwd_net").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top"),
            (pl.col("_fwd_net").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot"),
        ]).with_columns((pl.col("_top") - pl.col("_bot")).alias("b_diff")).sort("trade_date")

        m = mat.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d").is_in(pds)).sort("trade_date")
        cmp = m.select(["trade_date", f]).join(agg.select(["trade_date", "b_diff"]), on="trade_date", how="left")
        c = cmp.with_columns((pl.col(f) - pl.col("b_diff")).abs().alias("delta"))
        n_ok = c.filter(pl.col("delta") < 1e-12).height
        mx = c["delta"].max()
        print(f"  {f}: identical_dates={n_ok}/{c.height}, max|delta|={mx:.6g}")
        if n_ok < c.height:
            show = c.filter(pl.col("delta") > 1e-12).head(3)
            print(show)

log("DONE probe")
