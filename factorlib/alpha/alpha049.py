"""alpha049 — standalone alpha factor.

Alpha #049 — Conditional reversal based on close-acceleration threshold.

WorldQuant Formula
------------------
    ((((delay(close, 20) - delay(close, 10)) / 10 -
       (delay(close, 10) - close) / 10) < -0.1) ? 1 :
     (-1 * (close - delay(close, 1))))

Required panel columns: ``close``, ``stock_code``, ``trade_date``,
``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha049 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #049 — Conditional reversal based on close-acceleration threshold.

    WorldQuant Formula
    ------------------
        ((((delay(close, 20) - delay(close, 10)) / 10 -
           (delay(close, 10) - close) / 10) < -0.1) ? 1 :
         (-1 * (close - delay(close, 1))))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``,
    ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    inner = (delay(pl.col('close'), 20) - delay(pl.col('close'), 10)) / 10.0 - (delay(pl.col('close'), 10) - pl.col('close')) / 10.0
    return panel.select(pl.when(inner < -0.1).then(pl.lit(1.0)).otherwise(-1.0 * delta(pl.col('close'), 1)).alias('alpha049')).to_series()
