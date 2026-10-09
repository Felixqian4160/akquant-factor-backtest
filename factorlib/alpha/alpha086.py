"""alpha086 — standalone alpha factor.

Alpha #086 — Close vs sum(adv20) corr ts_rank inequality with body rank.

WorldQuant Formula
------------------
    (Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
     rank(open + close - vwap - open)) * -1

Required panel columns: ``close``, ``adv20``, ``open``, ``vwap``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha086 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #086 — Close vs sum(adv20) corr ts_rank inequality with body rank.

    WorldQuant Formula
    ------------------
        (Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
         rank(open + close - vwap - open)) * -1

    Required panel columns: ``close``, ``adv20``, ``open``, ``vwap``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ts_sum(pl.col('adv20'), 15).alias('__a086_sadv'))
    staged2 = staged.with_columns(ts_corr(pl.col('close'), pl.col('__a086_sadv'), 6).alias('__a086_corr'))
    rhs = pl.col('open') + pl.col('close') - (pl.col('vwap') + pl.col('open'))
    staged3 = staged2.with_columns(ts_rank(pl.col('__a086_corr'), 20).alias('__a086_p1'), cs_rank(rhs).alias('__a086_p2'))
    return staged3.select(((pl.col('__a086_p1') < pl.col('__a086_p2')).cast(pl.Float64) * -1.0).alias('alpha086')).to_series()
