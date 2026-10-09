"""alpha076 — standalone alpha factor.

Alpha #076 — Negative max of vwap-delta-decay rank and IndNeutralize(low)-adv81 corr ts_rank.

WorldQuant Formula
------------------
    max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
        Ts_Rank(decay_linear(Ts_Rank(correlation(
            IndNeutralize(low, IndClass.sector), adv81, 8.14941
        ), 19.569), 17.1543), 19.383)) * -1

Required panel columns: ``vwap``, ``low``, ``adv81``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha076 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #076 — Negative max of vwap-delta-decay rank and IndNeutralize(low)-adv81 corr ts_rank.

    WorldQuant Formula
    ------------------
        max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
            Ts_Rank(decay_linear(Ts_Rank(correlation(
                IndNeutralize(low, IndClass.sector), adv81, 8.14941
            ), 19.569), 17.1543), 19.383)) * -1

    Required panel columns: ``vwap``, ``low``, ``adv81``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('low'), 'industry').alias('__a076_il'), delta(pl.col('vwap'), 1).alias('__a076_dv'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a076_il'), pl.col('adv81'), 8).alias('__a076_corr'))
    staged3 = staged2.with_columns(ts_rank(pl.col('__a076_corr'), 20).alias('__a076_tr1'), ts_decay_linear(pl.col('__a076_dv'), 12).alias('__a076_dec1'))
    staged4 = staged3.with_columns(ts_decay_linear(pl.col('__a076_tr1'), 17).alias('__a076_dec2'))
    staged5 = staged4.with_columns(cs_rank(pl.col('__a076_dec1')).alias('__a076_p1'), ts_rank(pl.col('__a076_dec2'), 19).alias('__a076_p2'))
    return staged5.select((pl.max_horizontal(pl.col('__a076_p1'), pl.col('__a076_p2')) * -1.0).alias('alpha076')).to_series()
