"""alpha027 — standalone alpha factor.

Alpha #027 — Sign of correlation(rank(volume), rank(vwap)) majority.

WorldQuant Formula
------------------
    ((0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
     ? -1 : 1)

Polars Implementation Notes
---------------------------
Two-stage CS rank → TS corr → TS sum → CS rank, then threshold at 0.5.
The classic STHSF implementation has a discontinuity at 0.5; we
follow the same convention.

Required panel columns: ``volume``, ``vwap``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha027 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #027 — Sign of correlation(rank(volume), rank(vwap)) majority.

    WorldQuant Formula
    ------------------
        ((0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
         ? -1 : 1)

    Polars Implementation Notes
    ---------------------------
    Two-stage CS rank → TS corr → TS sum → CS rank, then threshold at 0.5.
    The classic STHSF implementation has a discontinuity at 0.5; we
    follow the same convention.

    Required panel columns: ``volume``, ``vwap``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(cs_rank(pl.col('volume')).alias('__a027_rv'), cs_rank(pl.col('vwap')).alias('__a027_rw'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a027_rv'), pl.col('__a027_rw'), 6).alias('__a027_corr'))
    staged3 = staged2.with_columns((ts_sum(pl.col('__a027_corr'), 2) / 2.0).alias('__a027_avg'))
    return staged3.select(pl.when(cs_rank(pl.col('__a027_avg')) > 0.5).then(-1.0).otherwise(1.0).alias('alpha027')).to_series()
