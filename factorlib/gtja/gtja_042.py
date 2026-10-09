"""gtja_042 — standalone gtja factor.

GTJA Alpha #042 — Negated rank-of-std times rolling H-V correlation.

Guotai Junan Formula
--------------------
    -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)

Required panel columns: ``high``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_042 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #042 — Negated rank-of-std times rolling H-V correlation.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(std_(pl.col('high'), 10).alias('__g042_s'), corr(pl.col('high'), pl.col('volume'), 10).alias('__g042_c'))
    return staged.select((-1.0 * rank(pl.col('__g042_s')) * pl.col('__g042_c')).alias('gtja_042').cast(pl.Float64)).to_series()
