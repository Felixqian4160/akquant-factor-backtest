"""gtja_011 — standalone gtja factor.

GTJA Alpha #011 — 6-day sum of normalised mid-range times volume.

Guotai Junan Formula
--------------------
    SUM((2*CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME, 6)

Required panel columns: ``close``, ``low``, ``high``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_011 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #011 — 6-day sum of normalised mid-range times volume.

    Guotai Junan Formula
    --------------------
        SUM((2*CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME, 6)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (2.0 * pl.col('close') - pl.col('low') - pl.col('high')) / (pl.col('high') - pl.col('low'))
    return panel.select(sum_(base * pl.col('volume'), 6).alias('gtja_011').cast(pl.Float64)).to_series()
