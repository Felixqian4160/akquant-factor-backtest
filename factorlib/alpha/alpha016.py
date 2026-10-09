"""alpha016 — standalone alpha factor.

Alpha #016 — Negative rank of high-rank vs volume-rank covariance.

WorldQuant Formula
------------------
    -1 * rank(covariance(rank(high), rank(volume), 5))

Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha016 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #016 — Negative rank of high-rank vs volume-rank covariance.

    WorldQuant Formula
    ------------------
        -1 * rank(covariance(rank(high), rank(volume), 5))

    Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(cs_rank(pl.col('high')).alias('__a016_rh'), cs_rank(pl.col('volume')).alias('__a016_rv'))
    staged2 = staged.with_columns(ts_cov(pl.col('__a016_rh'), pl.col('__a016_rv'), 5).alias('__a016_cov'))
    return staged2.select((-1.0 * cs_rank(pl.col('__a016_cov'))).alias('alpha016')).to_series()
