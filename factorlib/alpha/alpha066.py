"""alpha066 — standalone alpha factor.

Alpha #066 — Negative blend of vwap-delta decay rank and intraday-skew ts_rank.

WorldQuant Formula
------------------
    (rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
     Ts_Rank(decay_linear(
         ((low * 0.96633 + low * (1 - 0.96633)) - vwap) /
         (open - (high + low) / 2), 11.4157
     ), 6.72611)) * -1

Required panel columns: ``vwap``, ``low``, ``high``, ``open``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha066 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #066 — Negative blend of vwap-delta decay rank and intraday-skew ts_rank.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
         Ts_Rank(decay_linear(
             ((low * 0.96633 + low * (1 - 0.96633)) - vwap) /
             (open - (high + low) / 2), 11.4157
         ), 6.72611)) * -1

    Required panel columns: ``vwap``, ``low``, ``high``, ``open``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_low = pl.col('low') * 0.96633 + pl.col('low') * (1.0 - 0.96633)
    midpoint = (pl.col('high') + pl.col('low')) / 2.0
    inner2 = (blend_low - pl.col('vwap')) / (pl.col('open') - midpoint)
    staged = panel.with_columns(delta(pl.col('vwap'), 4).alias('__a066_dv'), inner2.alias('__a066_skew'))
    staged2 = staged.with_columns(ts_decay_linear(pl.col('__a066_dv'), 7).alias('__a066_d1'), ts_decay_linear(pl.col('__a066_skew'), 11).alias('__a066_d2'))
    return staged2.select(((cs_rank(pl.col('__a066_d1')) + ts_rank(pl.col('__a066_d2'), 7)) * -1.0).alias('alpha066')).to_series()
