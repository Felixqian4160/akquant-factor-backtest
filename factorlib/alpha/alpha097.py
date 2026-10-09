"""alpha097 — standalone alpha factor.

Alpha #097 — Diff of low-vwap-blend industry-neutralised delta-decay rank and ts_rank corr nest.

WorldQuant Formula
------------------
    (rank(decay_linear(delta(IndNeutralize(
        low * 0.721001 + vwap * (1 - 0.721001), IndClass.industry
     ), 3.3705), 20.4523)) -
     Ts_Rank(decay_linear(Ts_Rank(correlation(
         Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547
     ), 18.5925), 15.7152), 6.71659)) * -1

Required panel columns: ``low``, ``vwap``, ``adv60``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha097 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #097 — Diff of low-vwap-blend industry-neutralised delta-decay rank and ts_rank corr nest.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(IndNeutralize(
            low * 0.721001 + vwap * (1 - 0.721001), IndClass.industry
         ), 3.3705), 20.4523)) -
         Ts_Rank(decay_linear(Ts_Rank(correlation(
             Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547
         ), 18.5925), 15.7152), 6.71659)) * -1

    Required panel columns: ``low``, ``vwap``, ``adv60``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('low') * 0.721001 + pl.col('vwap') * (1.0 - 0.721001)
    staged = panel.with_columns(ind_neutralize(blend, 'industry').alias('__a097_ib'), ts_rank(pl.col('low'), 8).alias('__a097_trl'), ts_rank(pl.col('adv60'), 17).alias('__a097_tra'))
    staged2 = staged.with_columns(delta(pl.col('__a097_ib'), 3).alias('__a097_dib'), ts_corr(pl.col('__a097_trl'), pl.col('__a097_tra'), 5).alias('__a097_corr'))
    staged3 = staged2.with_columns(ts_rank(pl.col('__a097_corr'), 19).alias('__a097_tr1'), ts_decay_linear(pl.col('__a097_dib'), 20).alias('__a097_d1'))
    staged4 = staged3.with_columns(ts_decay_linear(pl.col('__a097_tr1'), 16).alias('__a097_d2'))
    return staged4.select(((cs_rank(pl.col('__a097_d1')) - ts_rank(pl.col('__a097_d2'), 7)) * -1.0).alias('alpha097')).to_series()
