"""gtja_180 — standalone gtja factor.

GTJA #180 — Conditional momentum vs negative volume.

Guotai Junan Formula
--------------------
    MEAN(VOLUME,20) < VOLUME ?
        -TSRANK(|DELTA(CLOSE,7)|,60) * SIGN(DELTA(CLOSE,7))
        : -VOLUME

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_180 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #180 — Conditional momentum vs negative volume.

    Guotai Junan Formula
    --------------------
        MEAN(VOLUME,20) < VOLUME ?
            -TSRANK(|DELTA(CLOSE,7)|,60) * SIGN(DELTA(CLOSE,7))
            : -VOLUME
    """
    df = panel.with_columns(delta(pl.col('close'), 7).alias('__d7'))
    cond = mean(pl.col('volume'), 20) < pl.col('volume')
    branch_true = -1.0 * ts_rank(pl.col('__d7').abs(), 60) * sign_(pl.col('__d7'))
    branch_false = -1.0 * pl.col('volume')
    expr = pl.when(cond).then(branch_true).otherwise(branch_false).alias('gtja_180')
    return df.select(expr).to_series()
