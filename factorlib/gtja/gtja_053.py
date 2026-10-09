"""gtja_053 — standalone gtja factor.

GTJA Alpha #053 — % of up-days over 12d × 100.

Guotai Junan Formula
--------------------
    COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_053 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #053 — % of up-days over 12d × 100.

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    cond = (c > delay(c, 1)).cast(pl.Float64)
    return panel.select((sum_(cond, 12) / 12.0 * 100.0).alias('gtja_053').cast(pl.Float64)).to_series()
