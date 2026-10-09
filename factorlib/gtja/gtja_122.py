"""gtja_122 — standalone gtja factor.

GTJA #122 — Triple-SMA log-close TSI-style oscillator.

Guotai Junan Formula
--------------------
    (SMA^3(LOG(CLOSE),13,2) - DELAY(SMA^3(LOG(CLOSE),13,2), 1)) /
     DELAY(SMA^3(LOG(CLOSE),13,2), 1)

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_122 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #122 — Triple-SMA log-close TSI-style oscillator.

    Guotai Junan Formula
    --------------------
        (SMA^3(LOG(CLOSE),13,2) - DELAY(SMA^3(LOG(CLOSE),13,2), 1)) /
         DELAY(SMA^3(LOG(CLOSE),13,2), 1)
    """
    log_c = log_(pl.col('close'))
    triple = sma(sma(sma(log_c, 13, 2), 13, 2), 13, 2)
    df = panel.with_columns(triple.alias('__t'))
    expr = ((pl.col('__t') - delay(pl.col('__t'), 1)) / delay(pl.col('__t'), 1)).alias('gtja_122')
    return df.select(expr).to_series()
