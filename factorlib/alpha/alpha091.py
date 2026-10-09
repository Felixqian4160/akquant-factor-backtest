"""alpha091 — standalone alpha factor.

Alpha #091 — Diff of double-decay-corr ts_rank and vwap-adv30 corr decay rank.

WorldQuant Formula
------------------
    (Ts_Rank(decay_linear(decay_linear(correlation(
        IndNeutralize(close, IndClass.industry), volume, 9.74928
    ), 16.398), 3.83219), 4.8667) -
     rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1

Required panel columns: ``close``, ``volume``, ``vwap``, ``adv30``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha091 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #091 — Diff of double-decay-corr ts_rank and vwap-adv30 corr decay rank.

    WorldQuant Formula
    ------------------
        (Ts_Rank(decay_linear(decay_linear(correlation(
            IndNeutralize(close, IndClass.industry), volume, 9.74928
        ), 16.398), 3.83219), 4.8667) -
         rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1

    Required panel columns: ``close``, ``volume``, ``vwap``, ``adv30``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('close'), 'industry').alias('__a091_ic'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a091_ic'), pl.col('volume'), 10).alias('__a091_c1'), ts_corr(pl.col('vwap'), pl.col('adv30'), 4).alias('__a091_c2'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a091_c1'), 16).alias('__a091_dec_inner'))
    staged4 = staged3.with_columns(ts_decay_linear(pl.col('__a091_dec_inner'), 4).alias('__a091_dec_outer'), ts_decay_linear(pl.col('__a091_c2'), 3).alias('__a091_dec2'))
    return staged4.select(((ts_rank(pl.col('__a091_dec_outer'), 5) - cs_rank(pl.col('__a091_dec2'))) * -1.0).alias('alpha091')).to_series()
