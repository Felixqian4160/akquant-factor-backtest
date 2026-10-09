"""gtja_007 — standalone gtja factor.

GTJA Alpha #007 — VWAP-close 3d max+min ranks × volume-delta rank.

Guotai Junan Formula
--------------------
    (RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))) *
    RANK(DELTA(VOLUME, 3))

Required panel columns: ``vwap``, ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_007 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #007 — VWAP-close 3d max+min ranks × volume-delta rank.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))) *
        RANK(DELTA(VOLUME, 3))

    Required panel columns: ``vwap``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    diff = pl.col('vwap') - pl.col('close')
    staged = panel.with_columns(ts_max(diff, 3).alias('__g007_max'), ts_min(diff, 3).alias('__g007_min'), delta(pl.col('volume'), 3).alias('__g007_dv'))
    return staged.select(((rank(pl.col('__g007_max')) + rank(pl.col('__g007_min'))) * rank(pl.col('__g007_dv'))).alias('gtja_007').cast(pl.Float64)).to_series()
