"""gtja_116 — standalone gtja factor.

GTJA #116 — REGBETA(CLOSE, SEQUENCE(20), 20).

Rolling slope of CLOSE on a monotonic 1..N time index. Daic115 uses
qlib's ``rolling_slope`` (slope on per-stock row index). We replicate
that semantics by building a per-stock row counter (``cum_count``)
and feeding it into our native :func:`regbeta`. Since the x-axis is
monotonic, ``regbeta`` (cov/var) produces the same OLS slope as
qlib's ``rolling_slope``.

Direction: ``normal``. Quality flag: ``0``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_116 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #116 — REGBETA(CLOSE, SEQUENCE(20), 20).

    Rolling slope of CLOSE on a monotonic 1..N time index. Daic115 uses
    qlib's ``rolling_slope`` (slope on per-stock row index). We replicate
    that semantics by building a per-stock row counter (``cum_count``)
    and feeding it into our native :func:`regbeta`. Since the x-axis is
    monotonic, ``regbeta`` (cov/var) produces the same OLS slope as
    qlib's ``rolling_slope``.

    Direction: ``normal``. Quality flag: ``0``.
    """
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns(row_idx.alias('__row_idx'))
    expr = regbeta(pl.col('close'), pl.col('__row_idx'), 20).alias('gtja_116')
    return df.select(expr).to_series()
