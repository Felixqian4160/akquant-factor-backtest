"""alpha050 — standalone alpha factor.

Alpha #050 — Negative ts_max of rank(corr(rank(volume), rank(vwap), 5)).

WorldQuant Formula
------------------
    -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

Required panel columns: ``volume``, ``vwap``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha050 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #050 — Negative ts_max of rank(corr(rank(volume), rank(vwap), 5)).

    WorldQuant Formula
    ------------------
        -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

    Required panel columns: ``volume``, ``vwap``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(cs_rank(pl.col('volume')).alias('__a050_rv'), cs_rank(pl.col('vwap')).alias('__a050_rw'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a050_rv'), pl.col('__a050_rw'), 5).alias('__a050_corr'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a050_corr')).alias('__a050_rc'))
    return staged3.select((-1.0 * ts_max(pl.col('__a050_rc'), 5)).alias('alpha050')).to_series()
