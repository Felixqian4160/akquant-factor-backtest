"""alpha096 — standalone alpha factor.

Alpha #096 — Negative max of two ts_rank(decay(corr)) composites.

WorldQuant Formula
------------------
    max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume),
                                          3.83878), 4.16783), 8.38151),
        Ts_Rank(decay_linear(Ts_ArgMax(correlation(
            Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459
        ), 12.6556), 14.0365), 13.4143)) * -1

Required panel columns: ``vwap``, ``volume``, ``close``, ``adv60``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha096 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #096 — Negative max of two ts_rank(decay(corr)) composites.

    WorldQuant Formula
    ------------------
        max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume),
                                              3.83878), 4.16783), 8.38151),
            Ts_Rank(decay_linear(Ts_ArgMax(correlation(
                Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459
            ), 12.6556), 14.0365), 13.4143)) * -1

    Required panel columns: ``vwap``, ``volume``, ``close``, ``adv60``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(cs_rank(pl.col('vwap')).alias('__a096_rv'), cs_rank(pl.col('volume')).alias('__a096_rl'), ts_rank(pl.col('close'), 7).alias('__a096_trc'), ts_rank(pl.col('adv60'), 4).alias('__a096_tra'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a096_rv'), pl.col('__a096_rl'), 4).alias('__a096_c1'), ts_corr(pl.col('__a096_trc'), pl.col('__a096_tra'), 4).alias('__a096_c2'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a096_c1'), 4).alias('__a096_d1'), ts_argmax(pl.col('__a096_c2'), 13).alias('__a096_am'))
    staged4 = staged3.with_columns(ts_decay_linear(pl.col('__a096_am'), 14).alias('__a096_d2'))
    staged5 = staged4.with_columns(ts_rank(pl.col('__a096_d1'), 8).alias('__a096_p1'), ts_rank(pl.col('__a096_d2'), 13).alias('__a096_p2'))
    return staged5.select((pl.max_horizontal(pl.col('__a096_p1'), pl.col('__a096_p2')) * -1.0).alias('alpha096')).to_series()
