"""gtja_017 — standalone gtja factor.

GTJA Alpha #017 — Rank(VWAP - 15d-max-VWAP) raised to 5d close delta.

Guotai Junan Formula
--------------------
    RANK(VWAP - MAX(VWAP, 15)) ^ DELTA(CLOSE, 5)

Numerical safety
----------------
The raw formula has ``rank ∈ [0, 1] ^ delta(close, 5)``. When the
exponent gets large in absolute value (delta(close, 5) reaches ±50
on adj_factor-glitchy days), the result blows up to 1e+308 and float
cast to inf. We use ``safe_pow_clip`` which clips the exponent to
[-3, 3] — bounded growth/decay rate, no overflow. Economic intent
preserved (a 3-fold price move is the absolute ceiling that matters
for momentum). Previous behaviour produced 4621 inf cells/year on
real data (max_finite = 1.4e+308); this fix eliminates them.

Required panel columns: ``vwap``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_017 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #017 — Rank(VWAP - 15d-max-VWAP) raised to 5d close delta.

    Guotai Junan Formula
    --------------------
        RANK(VWAP - MAX(VWAP, 15)) ^ DELTA(CLOSE, 5)

    Numerical safety
    ----------------
    The raw formula has ``rank ∈ [0, 1] ^ delta(close, 5)``. When the
    exponent gets large in absolute value (delta(close, 5) reaches ±50
    on adj_factor-glitchy days), the result blows up to 1e+308 and float
    cast to inf. We use ``safe_pow_clip`` which clips the exponent to
    [-3, 3] — bounded growth/decay rate, no overflow. Economic intent
    preserved (a 3-fold price move is the absolute ceiling that matters
    for momentum). Previous behaviour produced 4621 inf cells/year on
    real data (max_finite = 1.4e+308); this fix eliminates them.

    Required panel columns: ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = pl.col('vwap') - ts_max(pl.col('vwap'), 15)
    staged = panel.with_columns(inner.alias('__g017_inner'))
    staged = staged.with_columns(rank(pl.col('__g017_inner')).alias('__g017_r'))
    return staged.select(safe_pow_clip(pl.col('__g017_r'), delta(pl.col('close'), 5)).alias('gtja_017').cast(pl.Float64)).to_series()
