"""alpha031 — standalone alpha factor.

Alpha #031 — Triple-rank decay of inverse delta(close, 10) plus other parts.

WorldQuant Formula
------------------
    rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) +
    rank(-1 * delta(close, 3)) +
    sign(scale(correlation(adv20, low, 12)))

Required panel columns: ``close``, ``adv20``, ``low``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha031 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #031 — Triple-rank decay of inverse delta(close, 10) plus other parts.

    WorldQuant Formula
    ------------------
        rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) +
        rank(-1 * delta(close, 3)) +
        sign(scale(correlation(adv20, low, 12)))

    Required panel columns: ``close``, ``adv20``, ``low``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(delta(pl.col('close'), 10).alias('__a031_d10'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_d10')).alias('__a031_r1'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_r1')).alias('__a031_rr'))
    staged = staged.with_columns((-1.0 * pl.col('__a031_rr')).alias('__a031_inner'))
    staged = staged.with_columns(ts_decay_linear(pl.col('__a031_inner'), 10).alias('__a031_dec'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_dec')).alias('__a031_pr1'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_pr1')).alias('__a031_pr2'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_pr2')).alias('__a031_p1'))
    staged = staged.with_columns(delta(pl.col('close'), 3).alias('__a031_d3'))
    staged = staged.with_columns((-1.0 * pl.col('__a031_d3')).alias('__a031_neg_d3'))
    staged = staged.with_columns(cs_rank(pl.col('__a031_neg_d3')).alias('__a031_p2'))
    staged = staged.with_columns(ts_corr(pl.col('adv20'), pl.col('low'), 12).alias('__a031_corr'))
    staged = staged.with_columns(cs_scale(pl.col('__a031_corr')).alias('__a031_corr_s'))
    staged = staged.with_columns(sign_(pl.col('__a031_corr_s')).alias('__a031_p3'))
    return staged.select((pl.col('__a031_p1') + pl.col('__a031_p2') + pl.col('__a031_p3')).alias('alpha031')).to_series()
