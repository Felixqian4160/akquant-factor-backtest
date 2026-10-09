"""alpha062 — standalone alpha factor.

Alpha #062 — Open vs midpoint rank inequality, gated by vwap-adv20 corr.

WorldQuant Formula
------------------
    (rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
     rank(((rank(open) + rank(open)) <
           (rank((high + low) / 2) + rank(high))))) * -1

Required panel columns: ``vwap``, ``adv20``, ``open``, ``high``, ``low``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha062 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #062 — Open vs midpoint rank inequality, gated by vwap-adv20 corr.

    WorldQuant Formula
    ------------------
        (rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
         rank(((rank(open) + rank(open)) <
               (rank((high + low) / 2) + rank(high))))) * -1

    Required panel columns: ``vwap``, ``adv20``, ``open``, ``high``, ``low``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ts_sum(pl.col('adv20'), 22).alias('__a062_sadv'))
    staged2 = staged.with_columns(ts_corr(pl.col('vwap'), pl.col('__a062_sadv'), 10).alias('__a062_c1'))
    inner_b = (cs_rank(pl.col('open')) + cs_rank(pl.col('open'))).cast(pl.Float64) < (cs_rank((pl.col('high') + pl.col('low')) / 2.0) + cs_rank(pl.col('high'))).cast(pl.Float64)
    staged3 = staged2.with_columns(cs_rank(inner_b.cast(pl.Float64)).alias('__a062_p2'), cs_rank(pl.col('__a062_c1')).alias('__a062_p1'))
    return staged3.select(((pl.col('__a062_p1') < pl.col('__a062_p2')).cast(pl.Float64) * -1.0).alias('alpha062')).to_series()
