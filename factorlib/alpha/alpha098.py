"""alpha098 — standalone alpha factor.

Alpha #098 — Diff of vwap-adv5-corr decay rank and rank-corr ts_argmin nest.

WorldQuant Formula
------------------
    rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418),
                      7.18088)) -
    rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(
        rank(open), rank(adv15), 20.8187
    ), 8.62571), 6.95668), 8.07206))

Required panel columns: ``vwap``, ``adv5``, ``open``, ``adv15``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha098 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #098 — Diff of vwap-adv5-corr decay rank and rank-corr ts_argmin nest.

    WorldQuant Formula
    ------------------
        rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418),
                          7.18088)) -
        rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(
            rank(open), rank(adv15), 20.8187
        ), 8.62571), 6.95668), 8.07206))

    Required panel columns: ``vwap``, ``adv5``, ``open``, ``adv15``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ts_sum(pl.col('adv5'), 26).alias('__a098_sadv'), cs_rank(pl.col('open')).alias('__a098_ro'), cs_rank(pl.col('adv15')).alias('__a098_ra'))
    staged2 = staged.with_columns(ts_corr(pl.col('vwap'), pl.col('__a098_sadv'), 5).alias('__a098_c1'), ts_corr(pl.col('__a098_ro'), pl.col('__a098_ra'), 21).alias('__a098_c2'))
    staged3 = staged2.with_columns(ts_argmin(pl.col('__a098_c2'), 9).alias('__a098_am'))
    staged4 = staged3.with_columns(ts_rank(pl.col('__a098_am'), 7).alias('__a098_tr'), ts_decay_linear(pl.col('__a098_c1'), 7).alias('__a098_d1'))
    staged5 = staged4.with_columns(ts_decay_linear(pl.col('__a098_tr'), 8).alias('__a098_d2'))
    return staged5.select((cs_rank(pl.col('__a098_d1')) - cs_rank(pl.col('__a098_d2'))).alias('alpha098')).to_series()
