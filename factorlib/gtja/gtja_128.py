"""gtja_128 — standalone gtja factor.

GTJA #128 — Money-flow index over 14 days using typical price.

Guotai Junan Formula
--------------------
    100 - 100 / (1 + SUMIF(TP*V, 14, TP > prev_TP) /
                      SUMIF(TP*V, 14, TP < prev_TP))

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_128 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #128 — Money-flow index over 14 days using typical price.

    Guotai Junan Formula
    --------------------
        100 - 100 / (1 + SUMIF(TP*V, 14, TP > prev_TP) /
                          SUMIF(TP*V, 14, TP < prev_TP))
    """
    tp = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    df = panel.with_columns(tp.alias('__tp'))
    df = df.with_columns(delay(pl.col('__tp'), 1).alias('__ptp'))
    null_mask = pl.col('__ptp').is_null()
    cond = pl.col('__tp') > pl.col('__ptp')
    tp_v = pl.col('__tp') * pl.col('volume')
    pos = pl.when(null_mask).then(None).when(cond).then(tp_v).otherwise(0.0)
    neg = pl.when(null_mask).then(None).when(~cond).then(tp_v).otherwise(0.0)
    df = df.with_columns([sum_(pos, 14).alias('__sp'), sum_(neg, 14).alias('__sn')])
    expr = (100.0 - 100.0 / (1.0 + pl.col('__sp') / (pl.col('__sn') + 1e-07))).alias('gtja_128')
    return df.select(expr).to_series()
