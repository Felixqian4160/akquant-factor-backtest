"""gtja_182 — standalone gtja factor.

GTJA #182 — Co-movement count: (C>O & BMK_C>BMK_O) | (C<O & BMK_C<BMK_O).

Guotai Junan Formula
--------------------
    COUNT(
        (CLOSE > OPEN & BMK_C > BMK_O) | (CLOSE < OPEN & BMK_C < BMK_O),
        20
    ) / 20

Benchmark sourcing — cross-section mean of OHLC (matches Daic115).
Phase D: switch to true CSI300.

Direction: ``normal``. Quality flag: ``0``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_182 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #182 — Co-movement count: (C>O & BMK_C>BMK_O) | (C<O & BMK_C<BMK_O).

    Guotai Junan Formula
    --------------------
        COUNT(
            (CLOSE > OPEN & BMK_C > BMK_O) | (CLOSE < OPEN & BMK_C < BMK_O),
            20
        ) / 20

    Benchmark sourcing — cross-section mean of OHLC (matches Daic115).
    Phase D: switch to true CSI300.

    Direction: ``normal``. Quality flag: ``0``.
    """
    df = panel.with_columns([pl.col('close').mean().over('trade_date').alias('__bc'), pl.col('open').mean().over('trade_date').alias('__bo')])
    bench_up = pl.col('__bc') > pl.col('__bo')
    stock_up = pl.col('close') > pl.col('open')
    same_dir = stock_up == bench_up
    expr = (sum_(same_dir.cast(pl.Float64), 20) / 20.0).alias('gtja_182')
    return df.select(expr).to_series()
