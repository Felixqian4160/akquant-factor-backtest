"""alpha063 — standalone alpha factor.

Alpha #063 — Negative diff between two decay-linear rank composites.

WorldQuant Formula
------------------
    (rank(decay_linear(delta(IndNeutralize(close, IndClass.industry),
                              2.25164), 8.22237)) -
     rank(decay_linear(correlation(
          vwap * 0.318108 + open * (1 - 0.318108),
          sum(adv180, 37.2467), 13.557), 12.2883))) * -1

Required panel columns: ``close``, ``vwap``, ``open``, ``adv180``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha063 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #063 — Negative diff between two decay-linear rank composites.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(IndNeutralize(close, IndClass.industry),
                                  2.25164), 8.22237)) -
         rank(decay_linear(correlation(
              vwap * 0.318108 + open * (1 - 0.318108),
              sum(adv180, 37.2467), 13.557), 12.2883))) * -1

    Required panel columns: ``close``, ``vwap``, ``open``, ``adv180``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('close'), 'industry').alias('__a063_ic'))
    staged2 = staged.with_columns(delta(pl.col('__a063_ic'), 2).alias('__a063_dic'), ts_sum(pl.col('adv180'), 37).alias('__a063_sadv'), (pl.col('vwap') * 0.318108 + pl.col('open') * (1.0 - 0.318108)).alias('__a063_blend'))
    staged3 = staged2.with_columns(ts_corr(pl.col('__a063_blend'), pl.col('__a063_sadv'), 14).alias('__a063_c'), ts_decay_linear(pl.col('__a063_dic'), 8).alias('__a063_d1'))
    staged4 = staged3.with_columns(ts_decay_linear(pl.col('__a063_c'), 12).alias('__a063_d2'))
    return staged4.select(((cs_rank(pl.col('__a063_d1')) - cs_rank(pl.col('__a063_d2'))) * -1.0).alias('alpha063')).to_series()
