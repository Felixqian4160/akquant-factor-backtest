"""gtja_181 — standalone gtja factor.

GTJA #181 — Skewness-adjusted return-vs-benchmark composite (errata).

Guotai Junan Formula
--------------------
    SUM(((CLOSE/DELAY(CLOSE,1)-1) - MEAN(C/Cprev-1, 20)) -
        (BMK - MEAN(BMK,20))^2, 20) /
    SUM((BMK - MEAN(BMK,20))^3)

Benchmark sourcing
------------------
Same proxy approach as :func:`gtja_149` — Daic115's reference uses
cross-section mean of CLOSE returns. Production wiring to CSI300
OHLC is Phase D.

Direction: ``normal``. Quality flag: ``1`` (errata + degraded data).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_181 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #181 — Skewness-adjusted return-vs-benchmark composite (errata).

    Guotai Junan Formula
    --------------------
        SUM(((CLOSE/DELAY(CLOSE,1)-1) - MEAN(C/Cprev-1, 20)) -
            (BMK - MEAN(BMK,20))^2, 20) /
        SUM((BMK - MEAN(BMK,20))^3)

    Benchmark sourcing
    ------------------
    Same proxy approach as :func:`gtja_149` — Daic115's reference uses
    cross-section mean of CLOSE returns. Production wiring to CSI300
    OHLC is Phase D.

    Direction: ``normal``. Quality flag: ``1`` (errata + degraded data).
    """
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns(ret.alias('__ret'))
    bench = pl.col('__ret').mean().over('trade_date')
    df = df.with_columns(bench.alias('__bench'))
    df = df.with_columns([(pl.col('__ret') - mean(pl.col('__ret'), 20)).alias('__centered'), (pl.col('__bench') - mean(pl.col('__bench'), 20)).alias('__bcent')])
    num = sum_(pl.col('__centered') - pl.col('__bcent').pow(2), 20)
    den = sum_(pl.col('__bcent').pow(3), 20)
    expr = (num / den).alias('gtja_181')
    return df.select(expr).to_series()
