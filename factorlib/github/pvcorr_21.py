"""pvcorr_21 — standalone GitHub/academic factor formula.

Canonical source: /media/felix/f/quant/akquant-factor-backtest/examples/v34_build_part1_factors.py
The function body is copied from the canonical v34 formula source and adapted
only for the factorlib panel key ``stock_code``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from factorlib._ops.github_ops import prepare as _prepare

CLOSE = pl.col("close")

def compute(panel):
    d = _prepare(panel)
    w = d.with_columns([
        (CLOSE / CLOSE.shift(1).over("stock_code") - 1.0).alias("_ret"),
        (pl.col("vol") / pl.col("vol").shift(1).over("stock_code") - 1.0).alias("_volchg"),
    ])
    w = w.with_columns([
        pl.col("_ret").shift(1).over("stock_code").alias("_ret_lag"),
        pl.col("_volchg").shift(1).over("stock_code").alias("_volchg_lag"),
    ])
    w = w.with_columns((pl.col("_ret_lag") * pl.col("_volchg_lag")).alias("_xy"))
    w = w.with_columns([
        pl.col("_xy").rolling_mean(21).over("stock_code").alias("_exy"),
        pl.col("_ret_lag").rolling_mean(21).over("stock_code").alias("_ex"),
        pl.col("_volchg_lag").rolling_mean(21).over("stock_code").alias("_ey"),
        pl.col("_ret_lag").rolling_std(21).over("stock_code").alias("_sx"),
        pl.col("_volchg_lag").rolling_std(21).over("stock_code").alias("_sy"),
    ])
    return w.with_columns(((pl.col("_exy") - pl.col("_ex") * pl.col("_ey")) / (pl.col("_sx") * pl.col("_sy") + 1e-12)).alias("pvcorr_21"))["pvcorr_21"]
