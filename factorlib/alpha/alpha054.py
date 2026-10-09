"""alpha054 — standalone alpha factor.

Alpha #054 — Open^5 / Close^5 weighted intraday tail signal.

WorldQuant Formula (Kakushadze 2015, eq. 54)
--------------------------------------------
    ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

Legacy AQML Expression
----------------------
    -1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))

Polars Implementation Notes
---------------------------
1. Element-wise only — no rolling, no rank.
2. The denominator ``low - high`` is non-positive on every regular
   trading day (low <= high). It is zero when high == low (a fully
   static day), which yields ``inf``/``nan`` — STHSF reference shows
   the same behaviour, so we don't guard against it.

Required panel columns: ``low``, ``high``, ``open``, ``close``.

Direction: ``reverse``
Category: ``breakout``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/breakout.py

Usage:
    from factorlib.alpha.alpha054 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, if_then_else, ts_corr, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #054 — Open^5 / Close^5 weighted intraday tail signal.

    WorldQuant Formula (Kakushadze 2015, eq. 54)
    --------------------------------------------
        ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    Legacy AQML Expression
    ----------------------
        -1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))

    Polars Implementation Notes
    ---------------------------
    1. Element-wise only — no rolling, no rank.
    2. The denominator ``low - high`` is non-positive on every regular
       trading day (low <= high). It is zero when high == low (a fully
       static day), which yields ``inf``/``nan`` — STHSF reference shows
       the same behaviour, so we don't guard against it.

    Required panel columns: ``low``, ``high``, ``open``, ``close``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    open5 = pl.col('open').pow(5)
    close5 = pl.col('close').pow(5)
    numer = -1.0 * (pl.col('low') - pl.col('close')) * open5
    denom = (pl.col('low') - pl.col('high')) * close5
    expr = numer / denom
    return panel.select(expr.alias('alpha054').cast(pl.Float64)).to_series()
