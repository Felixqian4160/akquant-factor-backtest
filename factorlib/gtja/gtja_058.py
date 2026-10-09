"""gtja_058 — standalone gtja factor.

GTJA Alpha #058 — % of up-days over 20d × 100 (vwap-anchored).

Guotai Junan Formula
--------------------
    COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_058 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #058 — % of up-days over 20d × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col('vwap')
    cond = (v > delay(v, 1)).cast(pl.Float64)
    return panel.select((sum_(cond, 20) / 20.0 * 100.0).alias('gtja_058').cast(pl.Float64)).to_series()
