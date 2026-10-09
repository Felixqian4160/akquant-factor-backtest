"""gtja_001 — standalone gtja factor.

GTJA Alpha #001 — Volume change rank vs intraday return correlation.

Guotai Junan Formula
--------------------
    (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

Reference: Daic115/alpha191 alpha191_001 (formula only)

Required panel columns: ``volume``, ``close``, ``open``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_001 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #001 — Volume change rank vs intraday return correlation.

    Guotai Junan Formula
    --------------------
        (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

    Reference: Daic115/alpha191 alpha191_001 (formula only)

    Required panel columns: ``volume``, ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    rv = delta(log_(pl.col('volume')), 1)
    rret = (pl.col('close') - pl.col('open')) / pl.col('open')
    staged = panel.with_columns(rv.alias('__g001_dlv'), rret.alias('__g001_ret'))
    staged = staged.with_columns(rank(pl.col('__g001_dlv')).alias('__g001_rv'), rank(pl.col('__g001_ret')).alias('__g001_rret'))
    return staged.select((-1.0 * corr(pl.col('__g001_rv'), pl.col('__g001_rret'), 6)).alias('gtja_001').cast(pl.Float64)).to_series()
