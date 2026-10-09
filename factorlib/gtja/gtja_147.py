"""gtja_147 — standalone gtja factor.

GTJA #147 — REGBETA(MEAN(CLOSE,12), SEQUENCE(12)).

Daic115's reference uses qlib's ``rolling_slope`` on MEAN(CLOSE,12).
We use our native :func:`regbeta` against a per-stock row index — a
monotonic x-axis, which produces the same OLS slope.

Direction: ``normal``. Quality flag: ``0``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_147 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #147 — REGBETA(MEAN(CLOSE,12), SEQUENCE(12)).

    Daic115's reference uses qlib's ``rolling_slope`` on MEAN(CLOSE,12).
    We use our native :func:`regbeta` against a per-stock row index — a
    monotonic x-axis, which produces the same OLS slope.

    Direction: ``normal``. Quality flag: ``0``.
    """
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns([mean(pl.col('close'), 12).alias('__mc'), row_idx.alias('__row_idx')])
    expr = regbeta(pl.col('__mc'), pl.col('__row_idx'), 12).alias('gtja_147')
    return df.select(expr).to_series()
