"""gtja_060 — standalone gtja factor.

GTJA Alpha #060 — 20d sum of normalised mid-range × volume.

Guotai Junan Formula
--------------------
    SUM(((C - L) - (H - C)) / (H - L) * VOLUME, 20)

Required panel columns: ``close``, ``low``, ``high``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_060 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #060 — 20d sum of normalised mid-range × volume.

    Guotai Junan Formula
    --------------------
        SUM(((C - L) - (H - C)) / (H - L) * VOLUME, 20)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low'))
    return panel.select(sum_(base * pl.col('volume'), 20).alias('gtja_060').cast(pl.Float64)).to_series()
