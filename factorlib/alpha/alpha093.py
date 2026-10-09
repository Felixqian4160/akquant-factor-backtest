"""alpha093 — standalone alpha factor.

Alpha #093 — Industry-neutralised vwap corr decay ts_rank divided by close-blend delta-decay rank.

WorldQuant Formula
------------------
    Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry),
                                      adv81, 17.4193), 19.848), 7.54455) /
    rank(decay_linear(delta(close * 0.524434 + vwap * (1 - 0.524434),
                             2.77377), 16.2664))

Required panel columns: ``vwap``, ``adv81``, ``close``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha093 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #093 — Industry-neutralised vwap corr decay ts_rank divided by close-blend delta-decay rank.

    WorldQuant Formula
    ------------------
        Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry),
                                          adv81, 17.4193), 19.848), 7.54455) /
        rank(decay_linear(delta(close * 0.524434 + vwap * (1 - 0.524434),
                                 2.77377), 16.2664))

    Required panel columns: ``vwap``, ``adv81``, ``close``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('close') * 0.524434 + pl.col('vwap') * (1.0 - 0.524434)
    staged = panel.with_columns(ind_neutralize(pl.col('vwap'), 'industry').alias('__a093_iv'), delta(blend, 3).alias('__a093_db'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a093_iv'), pl.col('adv81'), 17).alias('__a093_corr'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a093_corr'), 20).alias('__a093_d1'), ts_decay_linear(pl.col('__a093_db'), 16).alias('__a093_d2'))
    return staged3.select((ts_rank(pl.col('__a093_d1'), 8) / cs_rank(pl.col('__a093_d2'))).alias('alpha093')).to_series()
