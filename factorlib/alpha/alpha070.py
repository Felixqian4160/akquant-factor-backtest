"""alpha070 — standalone alpha factor.

Alpha #070 — Power of vwap-delta rank by IndNeutralize(close)-adv50 corr ts_rank.

WorldQuant Formula
------------------
    (rank(delta(vwap, 1.29456))^
     Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50,
                         17.8256), 17.9171)) * -1

Required panel columns: ``vwap``, ``close``, ``adv50``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha070 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #070 — Power of vwap-delta rank by IndNeutralize(close)-adv50 corr ts_rank.

    WorldQuant Formula
    ------------------
        (rank(delta(vwap, 1.29456))^
         Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50,
                             17.8256), 17.9171)) * -1

    Required panel columns: ``vwap``, ``close``, ``adv50``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('close'), 'industry').alias('__a070_ic'), delta(pl.col('vwap'), 1).alias('__a070_dv'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a070_ic'), pl.col('adv50'), 18).alias('__a070_corr'))
    return staged2.select((cs_rank(pl.col('__a070_dv')).pow(ts_rank(pl.col('__a070_corr'), 18)) * -1.0).alias('alpha070')).to_series()
