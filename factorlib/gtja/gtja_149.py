"""gtja_149 — standalone gtja factor.

GTJA #149 — Downside-beta vs benchmark over 252 days.

Guotai Junan Formula
--------------------
    REGBETA(
        FILTER(CLOSE/DELAY(CLOSE,1)-1, BMK < DELAY(BMK,1)),
        FILTER(BMK/DELAY(BMK,1)-1, BMK < DELAY(BMK,1)),
        252)

Benchmark sourcing
------------------
The Daic115 reference uses cross-section mean of CLOSE/DELAY(CLOSE,1)
returns as proxy for CSI300 — a known degraded data source per the
alpha191 handoff doc §4.1. We follow the same proxy here so unit
tests on the synthetic panel match the reference parquet.

Production wiring to the real CSI300 OHLC (available in our
``index_daily`` table from 2015-10 onwards) is a Phase D task. The
formula itself is correct; only the data source is degraded.

Daic115 also drops the ``FILTER`` step (commented out) — we match
that and feed the full series into REGBETA, again for parity.

Direction: ``normal``. Quality flag: ``0``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_149 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #149 — Downside-beta vs benchmark over 252 days.

    Guotai Junan Formula
    --------------------
        REGBETA(
            FILTER(CLOSE/DELAY(CLOSE,1)-1, BMK < DELAY(BMK,1)),
            FILTER(BMK/DELAY(BMK,1)-1, BMK < DELAY(BMK,1)),
            252)

    Benchmark sourcing
    ------------------
    The Daic115 reference uses cross-section mean of CLOSE/DELAY(CLOSE,1)
    returns as proxy for CSI300 — a known degraded data source per the
    alpha191 handoff doc §4.1. We follow the same proxy here so unit
    tests on the synthetic panel match the reference parquet.

    Production wiring to the real CSI300 OHLC (available in our
    ``index_daily`` table from 2015-10 onwards) is a Phase D task. The
    formula itself is correct; only the data source is degraded.

    Daic115 also drops the ``FILTER`` step (commented out) — we match
    that and feed the full series into REGBETA, again for parity.

    Direction: ``normal``. Quality flag: ``0``.
    """
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns(ret.alias('__ret'))
    bench = pl.col('__ret').mean().over('trade_date')
    df = df.with_columns(bench.alias('__bench'))
    expr = regbeta(pl.col('__ret'), pl.col('__bench'), 252).alias('gtja_149')
    return df.select(expr).to_series()
