"""alpha058 — standalone alpha factor.

Alpha #058 — Negative ts_rank of decay(corr(IndNeutralize(vwap), volume)).

WorldQuant Formula
------------------
    -1 * Ts_Rank(decay_linear(
        correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795),
        7.89291
    ), 5.50322)

Polars Implementation Notes
---------------------------
Continuous windows truncated to int (4, 8, 6). Sector neutralisation
is applied via ``ind_neutralize(vwap, "industry")`` — AurumQ's
available proxy for IndClass.sector.

Required panel columns: ``vwap``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha058 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #058 — Negative ts_rank of decay(corr(IndNeutralize(vwap), volume)).

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(decay_linear(
            correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795),
            7.89291
        ), 5.50322)

    Polars Implementation Notes
    ---------------------------
    Continuous windows truncated to int (4, 8, 6). Sector neutralisation
    is applied via ``ind_neutralize(vwap, "industry")`` — AurumQ's
    available proxy for IndClass.sector.

    Required panel columns: ``vwap``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(ind_neutralize(pl.col('vwap'), 'industry').alias('__a058_ivwap'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a058_ivwap'), pl.col('volume'), 4).alias('__a058_corr'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a058_corr'), 8).alias('__a058_dec'))
    return staged3.select((-1.0 * ts_rank(pl.col('__a058_dec'), 6)).alias('alpha058')).to_series()
