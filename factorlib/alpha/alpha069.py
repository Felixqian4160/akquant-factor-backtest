"""alpha069 — standalone alpha factor.

Alpha #069 — Power of vwap-delta-max-rank by close-blend-corr ts_rank.

WorldQuant Formula
------------------
    (rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry),
                       2.72412), 4.79344))^
     Ts_Rank(correlation(close * 0.490655 + vwap * (1 - 0.490655),
                         adv20, 4.92416), 9.0615)) * -1

Required panel columns: ``vwap``, ``close``, ``adv20``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha069 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #069 — Power of vwap-delta-max-rank by close-blend-corr ts_rank.

    WorldQuant Formula
    ------------------
        (rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry),
                           2.72412), 4.79344))^
         Ts_Rank(correlation(close * 0.490655 + vwap * (1 - 0.490655),
                             adv20, 4.92416), 9.0615)) * -1

    Required panel columns: ``vwap``, ``close``, ``adv20``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('vwap'), 'industry').alias('__a069_iv'))
    staged2 = staged.with_columns(delta(pl.col('__a069_iv'), 3).alias('__a069_div'))
    staged3 = staged2.with_columns(ts_max(pl.col('__a069_div'), 5).alias('__a069_max'), (pl.col('close') * 0.490655 + pl.col('vwap') * (1.0 - 0.490655)).alias('__a069_blend'))
    staged4 = staged3.with_columns(ts_corr(pl.col('__a069_blend'), pl.col('adv20'), 5).alias('__a069_corr'))
    return staged4.select((cs_rank(pl.col('__a069_max')).pow(ts_rank(pl.col('__a069_corr'), 9)) * -1.0).alias('alpha069')).to_series()
