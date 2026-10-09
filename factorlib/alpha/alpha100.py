"""alpha100 — standalone alpha factor.

Alpha #100 — Subindustry-neutralised body * volume signal minus close-rank-corr.

WorldQuant Formula
------------------
    0 - 1 * (
        (1.5 * scale(IndNeutralize(IndNeutralize(
            rank(((close - low) - (high - close)) / (high - low) * volume),
            IndClass.subindustry), IndClass.subindustry)) -
         scale(IndNeutralize(
             (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
             IndClass.subindustry
         ))
        ) * (volume / adv20)
    )

Required panel columns: ``close``, ``low``, ``high``, ``volume``,
``adv20``, ``stock_code``, ``trade_date``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha100 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #100 — Subindustry-neutralised body * volume signal minus close-rank-corr.

    WorldQuant Formula
    ------------------
        0 - 1 * (
            (1.5 * scale(IndNeutralize(IndNeutralize(
                rank(((close - low) - (high - close)) / (high - low) * volume),
                IndClass.subindustry), IndClass.subindustry)) -
             scale(IndNeutralize(
                 (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
                 IndClass.subindustry
             ))
            ) * (volume / adv20)
        )

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``adv20``, ``stock_code``, ``trade_date``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    body = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low')) * pl.col('volume')
    staged = panel.with_columns(cs_rank(body).alias('__a100_rb'), cs_rank(pl.col('adv20')).alias('__a100_ra'))
    staged2 = staged.with_columns(ind_neutralize(pl.col('__a100_rb'), 'sub_industry').alias('__a100_n1'), ts_corr(pl.col('close'), pl.col('__a100_ra'), 5).alias('__a100_c1'), ts_argmin(pl.col('close'), 30).alias('__a100_am'))
    staged3 = staged2.with_columns(ind_neutralize(pl.col('__a100_n1'), 'sub_industry').alias('__a100_n2'), cs_rank(pl.col('__a100_am')).alias('__a100_ram'))
    staged4 = staged3.with_columns((pl.col('__a100_c1') - pl.col('__a100_ram')).alias('__a100_part2_inner'), cs_scale(pl.col('__a100_n2')).alias('__a100_s1'))
    staged5 = staged4.with_columns(ind_neutralize(pl.col('__a100_part2_inner'), 'sub_industry').alias('__a100_n3'))
    staged6 = staged5.with_columns(cs_scale(pl.col('__a100_n3')).alias('__a100_s2'))
    return staged6.select((-1.0 * ((1.5 * pl.col('__a100_s1') - pl.col('__a100_s2')) * (pl.col('volume') / pl.col('adv20')))).alias('alpha100')).to_series()
