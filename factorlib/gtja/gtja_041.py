"""gtja_041 — standalone gtja factor.

GTJA Alpha #041 — Negated rank of 5d-max of 3d VWAP delta.

Guotai Junan Formula
--------------------
    RANK(MAX(DELTA(VWAP, 3), 5)) * -1

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_041 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #041 — Negated rank of 5d-max of 3d VWAP delta.

    Guotai Junan Formula
    --------------------
        RANK(MAX(DELTA(VWAP, 3), 5)) * -1

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = ts_max(delta(pl.col('vwap'), 3), 5)
    staged = panel.with_columns(inner.alias('__g041_x'))
    return staged.select((rank(pl.col('__g041_x')) * -1.0).alias('gtja_041').cast(pl.Float64)).to_series()
