"""alpha059 — standalone alpha factor.

Alpha #059 — Sector-neutralised vwap blend correlated with volume, decayed.

WorldQuant Formula
------------------
    -1 * Ts_Rank(decay_linear(
        correlation(IndNeutralize(
            vwap * 0.728317 + vwap * (1 - 0.728317),
            IndClass.industry
        ), volume, 4.25197), 16.2289
    ), 8.19648)

Polars Implementation Notes
---------------------------
The convex blend is degenerate (always equal to ``vwap``); we keep
the literal formula. Windows truncated: 4, 16, 8.

Required panel columns: ``vwap``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha059 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #059 — Sector-neutralised vwap blend correlated with volume, decayed.

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(decay_linear(
            correlation(IndNeutralize(
                vwap * 0.728317 + vwap * (1 - 0.728317),
                IndClass.industry
            ), volume, 4.25197), 16.2289
        ), 8.19648)

    Polars Implementation Notes
    ---------------------------
    The convex blend is degenerate (always equal to ``vwap``); we keep
    the literal formula. Windows truncated: 4, 16, 8.

    Required panel columns: ``vwap``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('vwap') * 0.728317 + pl.col('vwap') * (1.0 - 0.728317)
    staged = panel.with_columns(ind_neutralize(blend, 'industry').alias('__a059_iv'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a059_iv'), pl.col('volume'), 4).alias('__a059_corr'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a059_corr'), 16).alias('__a059_dec'))
    return staged3.select((-1.0 * ts_rank(pl.col('__a059_dec'), 8)).alias('alpha059')).to_series()
