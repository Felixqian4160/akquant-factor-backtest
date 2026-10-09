"""gtja_002 — standalone gtja factor.

GTJA Alpha #002 — One-period delta of normalised mid-range.

Guotai Junan Formula
--------------------
    (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

Required panel columns: ``close``, ``low``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_002 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #002 — One-period delta of normalised mid-range.

    Guotai Junan Formula
    --------------------
        (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    base = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low'))
    return panel.select((-1.0 * delta(base, 1)).alias('gtja_002').cast(pl.Float64)).to_series()
