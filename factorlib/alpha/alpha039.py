"""alpha039 — standalone alpha factor.

Alpha #039 — Negative rank of momentum-weighted-by-volume scaled by 250d return rank.

WorldQuant Formula
------------------
    (-1 * rank(delta(close, 7) *
               (1 - rank(decay_linear(volume / adv20, 9))))) *
    (1 + rank(sum(returns, 250)))

Required panel columns: ``close``, ``volume``, ``adv20``, ``returns``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha039 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #039 — Negative rank of momentum-weighted-by-volume scaled by 250d return rank.

    WorldQuant Formula
    ------------------
        (-1 * rank(delta(close, 7) *
                   (1 - rank(decay_linear(volume / adv20, 9))))) *
        (1 + rank(sum(returns, 250)))

    Required panel columns: ``close``, ``volume``, ``adv20``, ``returns``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ts_decay_linear(pl.col('volume') / pl.col('adv20'), 9).alias('__a039_dec'), delta(pl.col('close'), 7).alias('__a039_dc'), ts_sum(pl.col('returns'), 250).alias('__a039_sret'))
    staged2 = staged.with_columns((pl.col('__a039_dc') * (1.0 - cs_rank(pl.col('__a039_dec')))).alias('__a039_inner'))
    return staged2.select((-1.0 * cs_rank(pl.col('__a039_inner')) * (1.0 + cs_rank(pl.col('__a039_sret')))).alias('alpha039')).to_series()
