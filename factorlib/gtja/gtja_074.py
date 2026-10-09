"""gtja_074 — standalone gtja factor.

GTJA Alpha #074 — Rank(corr(sum-weighted, sum-MA-V, 7)) + rank(corr(rank-VWAP, rank-V, 6)).

Guotai Junan Formula
--------------------
    RANK(CORR(SUM(L*0.35 + VWAP*0.65, 20), SUM(MEAN(V, 40), 20), 7)) +
    RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))

Required panel columns: ``low``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_074 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #074 — Rank(corr(sum-weighted, sum-MA-V, 7)) + rank(corr(rank-VWAP, rank-V, 6)).

    Guotai Junan Formula
    --------------------
        RANK(CORR(SUM(L*0.35 + VWAP*0.65, 20), SUM(MEAN(V, 40), 20), 7)) +
        RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))

    Required panel columns: ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    weighted = pl.col('low') * 0.35 + pl.col('vwap') * 0.65
    arm1_corr = corr(sum_(weighted, 20), sum_(mean(pl.col('volume'), 40), 20), 7)
    staged = panel.with_columns(arm1_corr.alias('__g074_c1'), rank(pl.col('vwap')).alias('__g074_rw'), rank(pl.col('volume')).alias('__g074_rv'))
    staged = staged.with_columns(corr(pl.col('__g074_rw'), pl.col('__g074_rv'), 6).alias('__g074_c2'))
    staged = staged.with_columns(rank(pl.col('__g074_c1')).alias('__g074_r1'), rank(pl.col('__g074_c2')).alias('__g074_r2'))
    return staged.select((pl.col('__g074_r1') + pl.col('__g074_r2')).alias('gtja_074').cast(pl.Float64)).to_series()
