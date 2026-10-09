"""alpha020 — standalone alpha factor.

Alpha #020 — Triple gap-rank product (sign-flipped).

WorldQuant Formula
------------------
    -1 * rank(open - delay(high, 1)) *
          rank(open - delay(close, 1)) *
          rank(open - delay(low, 1))

Required panel columns: ``open``, ``high``, ``close``, ``low``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha020 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #020 — Triple gap-rank product (sign-flipped).

    WorldQuant Formula
    ------------------
        -1 * rank(open - delay(high, 1)) *
              rank(open - delay(close, 1)) *
              rank(open - delay(low, 1))

    Required panel columns: ``open``, ``high``, ``close``, ``low``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns((pl.col('open') - delay(pl.col('high'), 1)).alias('__a020_h'), (pl.col('open') - delay(pl.col('close'), 1)).alias('__a020_c'), (pl.col('open') - delay(pl.col('low'), 1)).alias('__a020_l'))
    return staged.select((-1.0 * cs_rank(pl.col('__a020_h')) * cs_rank(pl.col('__a020_c')) * cs_rank(pl.col('__a020_l'))).alias('alpha020')).to_series()
