"""alpha030 — standalone alpha factor.

Alpha #030 — Three-day sign-momentum rank scaled by short-vs-long volume sum.

WorldQuant Formula
------------------
    ((1.0 - rank(sign(close - delay(close, 1)) +
                 sign(delay(close, 1) - delay(close, 2)) +
                 sign(delay(close, 2) - delay(close, 3)))) *
     sum(volume, 5)) / sum(volume, 20)

Required panel columns: ``close``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha030 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #030 — Three-day sign-momentum rank scaled by short-vs-long volume sum.

    WorldQuant Formula
    ------------------
        ((1.0 - rank(sign(close - delay(close, 1)) +
                     sign(delay(close, 1) - delay(close, 2)) +
                     sign(delay(close, 2) - delay(close, 3)))) *
         sum(volume, 5)) / sum(volume, 20)

    Required panel columns: ``close``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    dc1 = delta(pl.col('close'), 1)
    inner = sign_(dc1) + sign_(delay(dc1, 1)) + sign_(delay(dc1, 2))
    staged = panel.with_columns(inner.alias('__a030_inner'))
    return staged.select(((1.0 - cs_rank(pl.col('__a030_inner'))) * ts_sum(pl.col('volume'), 5) / ts_sum(pl.col('volume'), 20)).alias('alpha030')).to_series()
