"""alpha067 — standalone alpha factor.

Alpha #067 — high-from-min power composite, sector-neutralised.

WorldQuant Formula
------------------
    (rank(high - ts_min(high, 2.14593))^
     rank(correlation(IndNeutralize(vwap, IndClass.sector),
                      IndNeutralize(adv20, IndClass.subindustry),
                      6.02936))) * -1

Required panel columns: ``high``, ``vwap``, ``adv20``, ``stock_code``,
``trade_date``, ``industry``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha067 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #067 — high-from-min power composite, sector-neutralised.

    WorldQuant Formula
    ------------------
        (rank(high - ts_min(high, 2.14593))^
         rank(correlation(IndNeutralize(vwap, IndClass.sector),
                          IndNeutralize(adv20, IndClass.subindustry),
                          6.02936))) * -1

    Required panel columns: ``high``, ``vwap``, ``adv20``, ``stock_code``,
    ``trade_date``, ``industry``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('vwap'), 'industry').alias('__a067_iv'), ind_neutralize(pl.col('adv20'), 'sub_industry').alias('__a067_ia'), (pl.col('high') - ts_min(pl.col('high'), 2)).alias('__a067_diff'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a067_iv'), pl.col('__a067_ia'), 6).alias('__a067_corr'))
    return staged2.select((cs_rank(pl.col('__a067_diff')).pow(cs_rank(pl.col('__a067_corr'))) * -1.0).alias('alpha067')).to_series()
