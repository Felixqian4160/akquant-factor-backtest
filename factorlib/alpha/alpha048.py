"""alpha048 — standalone alpha factor.

Alpha #048 — Sub-industry neutralised long-corr-of-deltas / sum-of-squared-returns.

WorldQuant Formula
------------------
    IndNeutralize(
        (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
         delta(close, 1)) / close,
        IndClass.subindustry
    ) /
    sum((delta(close, 1) / delay(close, 1))^2, 250)

Polars Implementation Notes
---------------------------
250-day windows on a 60-day synthetic panel produce all-null output —
the steady-state test only verifies the structural shape, not value
coverage. On real production panels with >= 250 days the values are
well-defined.

Required panel columns: ``close``, ``stock_code``, ``trade_date``,
``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha048 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #048 — Sub-industry neutralised long-corr-of-deltas / sum-of-squared-returns.

    WorldQuant Formula
    ------------------
        IndNeutralize(
            (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
             delta(close, 1)) / close,
            IndClass.subindustry
        ) /
        sum((delta(close, 1) / delay(close, 1))^2, 250)

    Polars Implementation Notes
    ---------------------------
    250-day windows on a 60-day synthetic panel produce all-null output —
    the steady-state test only verifies the structural shape, not value
    coverage. On real production panels with >= 250 days the values are
    well-defined.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``,
    ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    dclose = delta(pl.col('close'), 1)
    dclose_lag = delta(delay(pl.col('close'), 1), 1)
    staged = panel.with_columns(dclose.alias('__a048_dc'), dclose_lag.alias('__a048_dcl'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a048_dc'), pl.col('__a048_dcl'), 250).alias('__a048_corr'), (pl.col('__a048_dc') / delay(pl.col('close'), 1)).alias('__a048_pct'))
    staged3 = staged2.with_columns((pl.col('__a048_corr') * pl.col('__a048_dc') / pl.col('close')).alias('__a048_num'), ts_sum(pl.col('__a048_pct').pow(2.0), 250).alias('__a048_den'))
    staged4 = staged3.with_columns(ind_neutralize(pl.col('__a048_num'), 'sub_industry').alias('__a048_neut'))
    return staged4.select((pl.col('__a048_neut') / pl.col('__a048_den')).alias('alpha048')).to_series()
