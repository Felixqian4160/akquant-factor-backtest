"""gtja_054 — standalone gtja factor.

GTJA Alpha #054 — Negated rank of std-of-asymmetric-spread + close-open corr.

Guotai Junan Formula
--------------------
    -1 * RANK(STD(|C-O| + (C-O), 10) + CORR(C, O, 10))

Required panel columns: ``close``, ``open``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_054 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #054 — Negated rank of std-of-asymmetric-spread + close-open corr.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(|C-O| + (C-O), 10) + CORR(C, O, 10))

    Required panel columns: ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    diff = pl.col('close') - pl.col('open')
    inner = std_(abs_(diff) + diff, 10) + corr(pl.col('close'), pl.col('open'), 10)
    staged = panel.with_columns(inner.alias('__g054_x'))
    return staged.select((-1.0 * rank(pl.col('__g054_x'))).alias('gtja_054').cast(pl.Float64)).to_series()
