"""gtja_124 — standalone gtja factor.

GTJA #124 — (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_124 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #124 — (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)."""
    df = panel.with_columns(ts_max(pl.col('close'), 30).alias('__tc'))
    df = df.with_columns(rank(pl.col('__tc')).alias('__r'))
    df = df.with_columns(decay_linear(pl.col('__r'), 2).alias('__dl'))
    expr = ((pl.col('close') - pl.col('vwap')) / pl.col('__dl')).alias('gtja_124')
    return df.select(expr).to_series()
