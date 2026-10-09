"""alpha090 — standalone alpha factor.

Alpha #090 — Negative power composite of close-from-max rank and IndNeutralize(adv40)-low corr.

WorldQuant Formula
------------------
    (rank(close - ts_max(close, 4.66719))^
     Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry),
                         low, 5.38375), 3.21856)) * -1

Required panel columns: ``close``, ``adv40``, ``low``, ``stock_code``,
``trade_date``, ``industry``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha090 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #090 — Negative power composite of close-from-max rank and IndNeutralize(adv40)-low corr.

    WorldQuant Formula
    ------------------
        (rank(close - ts_max(close, 4.66719))^
         Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry),
                             low, 5.38375), 3.21856)) * -1

    Required panel columns: ``close``, ``adv40``, ``low``, ``stock_code``,
    ``trade_date``, ``industry``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('adv40'), 'sub_industry').alias('__a090_ia'), (pl.col('close') - ts_max(pl.col('close'), 5)).alias('__a090_diff'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a090_ia'), pl.col('low'), 5).alias('__a090_corr'))
    return staged2.select((cs_rank(pl.col('__a090_diff')).pow(ts_rank(pl.col('__a090_corr'), 3)) * -1.0).alias('alpha090')).to_series()
