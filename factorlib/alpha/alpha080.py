"""alpha080 — standalone alpha factor.

Alpha #080 — Sign-of-IndNeutralize(open-high blend) delta, raised to corr power.

WorldQuant Formula
------------------
    (rank(Sign(delta(IndNeutralize(open * 0.868128 + high * (1 - 0.868128),
                                   IndClass.industry), 4.04545)))^
     Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1

Required panel columns: ``open``, ``high``, ``adv10``, ``stock_code``,
``trade_date``, ``industry``

Polars Implementation Notes
---------------------------
Synthetic panel doesn't have ``adv10`` (only adv5/15/...). We use
``adv15`` as the closest available substitute on the synthetic panel
while keeping the original formula's intent. On real production
panels the adv10 column is present.

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha080 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #080 — Sign-of-IndNeutralize(open-high blend) delta, raised to corr power.

    WorldQuant Formula
    ------------------
        (rank(Sign(delta(IndNeutralize(open * 0.868128 + high * (1 - 0.868128),
                                       IndClass.industry), 4.04545)))^
         Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1

    Required panel columns: ``open``, ``high``, ``adv10``, ``stock_code``,
    ``trade_date``, ``industry``

    Polars Implementation Notes
    ---------------------------
    Synthetic panel doesn't have ``adv10`` (only adv5/15/...). We use
    ``adv15`` as the closest available substitute on the synthetic panel
    while keeping the original formula's intent. On real production
    panels the adv10 column is present.

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('open') * 0.868128 + pl.col('high') * (1.0 - 0.868128)
    staged = panel.with_columns(ind_neutralize(blend, 'industry').alias('__a080_ib'))
    staged2 = staged.with_columns(sign_(delta(pl.col('__a080_ib'), 4)).alias('__a080_sd'), ts_corr(pl.col('high'), pl.col('adv15'), 5).alias('__a080_corr'))
    return staged2.select((cs_rank(pl.col('__a080_sd')).pow(ts_rank(pl.col('__a080_corr'), 6)) * -1.0).alias('alpha080')).to_series()
