"""gtja_075 — standalone gtja factor.

GTJA Alpha #075 — Conditional up-day count ratio vs benchmark down-days.

Guotai Junan Formula
--------------------
    COUNT(C > O & BENCH_C < BENCH_O, 50) / COUNT(BENCH_C < BENCH_O, 50)

Daic115 substitutes a CS-mean return for the benchmark. We do the
same: ``bench_ret = mean(returns)`` per trade_date, then `bench<0`
is our "benchmark down" indicator.

Required panel columns: ``close``, ``open``, ``returns``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_075 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #075 — Conditional up-day count ratio vs benchmark down-days.

    Guotai Junan Formula
    --------------------
        COUNT(C > O & BENCH_C < BENCH_O, 50) / COUNT(BENCH_C < BENCH_O, 50)

    Daic115 substitutes a CS-mean return for the benchmark. We do the
    same: ``bench_ret = mean(returns)`` per trade_date, then `bench<0`
    is our "benchmark down" indicator.

    Required panel columns: ``close``, ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    cs_ret = pl.col('returns').mean().over('trade_date')
    bench_dn = cs_ret < 0.0
    cond_a = (pl.col('close') > pl.col('open')) & bench_dn
    cond_b = (pl.col('close') != pl.col('open')) & bench_dn
    a = sum_(cond_a.cast(pl.Float64), 50)
    b = sum_(cond_b.cast(pl.Float64), 50)
    return panel.select((a / b).alias('gtja_075').cast(pl.Float64)).to_series()
