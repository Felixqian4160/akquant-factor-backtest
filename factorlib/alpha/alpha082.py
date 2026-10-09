"""alpha082 — standalone alpha factor.

Alpha #082 — Negative min of open-delta-decay rank and IndNeutralize(volume)-open corr ts_rank.

WorldQuant Formula
------------------
    min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
        Ts_Rank(decay_linear(correlation(
            IndNeutralize(volume, IndClass.sector),
            open * 0.634196 + open * (1 - 0.634196),
            17.4842
        ), 6.92131), 13.4283)) * -1

Required panel columns: ``open``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha082 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #082 — Negative min of open-delta-decay rank and IndNeutralize(volume)-open corr ts_rank.

    WorldQuant Formula
    ------------------
        min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
            Ts_Rank(decay_linear(correlation(
                IndNeutralize(volume, IndClass.sector),
                open * 0.634196 + open * (1 - 0.634196),
                17.4842
            ), 6.92131), 13.4283)) * -1

    Required panel columns: ``open``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_open = pl.col('open') * 0.634196 + pl.col('open') * (1.0 - 0.634196)
    staged = panel.with_columns(ind_neutralize(pl.col('volume'), 'industry').alias('__a082_iv'), delta(pl.col('open'), 1).alias('__a082_do'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a082_iv'), blend_open, 17).alias('__a082_corr'), ts_decay_linear(pl.col('__a082_do'), 15).alias('__a082_d1'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a082_corr'), 7).alias('__a082_d2'))
    staged4 = staged3.with_columns(cs_rank(pl.col('__a082_d1')).alias('__a082_p1'), ts_rank(pl.col('__a082_d2'), 13).alias('__a082_p2'))
    return staged4.select((pl.min_horizontal(pl.col('__a082_p1'), pl.col('__a082_p2')) * -1.0).alias('alpha082')).to_series()
