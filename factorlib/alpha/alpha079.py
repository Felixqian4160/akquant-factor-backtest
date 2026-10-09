"""alpha079 — standalone alpha factor.

Alpha #079 — Industry-neutralised close-open blend delta rank inequality.

WorldQuant Formula
------------------
    rank(delta(IndNeutralize(close * 0.60733 + open * (1 - 0.60733),
                             IndClass.sector), 1.23438)) <
    rank(correlation(Ts_Rank(vwap, 3.60973),
                     Ts_Rank(adv150, 9.18637), 14.6644))

Required panel columns: ``close``, ``open``, ``vwap``, ``adv150``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha079 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #079 — Industry-neutralised close-open blend delta rank inequality.

    WorldQuant Formula
    ------------------
        rank(delta(IndNeutralize(close * 0.60733 + open * (1 - 0.60733),
                                 IndClass.sector), 1.23438)) <
        rank(correlation(Ts_Rank(vwap, 3.60973),
                         Ts_Rank(adv150, 9.18637), 14.6644))

    Required panel columns: ``close``, ``open``, ``vwap``, ``adv150``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('close') * 0.60733 + pl.col('open') * (1.0 - 0.60733)
    staged = panel.with_columns(ind_neutralize(blend, 'industry').alias('__a079_ib'))
    staged2 = staged.with_columns(delta(pl.col('__a079_ib'), 1).alias('__a079_dib'), ts_rank(pl.col('vwap'), 4).alias('__a079_trv'), ts_rank(pl.col('adv150'), 9).alias('__a079_tra'))
    staged3 = staged2.with_columns(ts_corr(pl.col('__a079_trv'), pl.col('__a079_tra'), 15).alias('__a079_c'))
    return staged3.select((cs_rank(pl.col('__a079_dib')) < cs_rank(pl.col('__a079_c'))).cast(pl.Float64).alias('alpha079')).to_series()
