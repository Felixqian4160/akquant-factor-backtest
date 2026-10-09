"""gtja_142 — standalone gtja factor.

GTJA #142 — Triple-rank acceleration product.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_142 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #142 — Triple-rank acceleration product."""
    df = panel.with_columns([ts_rank(pl.col('close'), 10).alias('__tc'), delta(delta(pl.col('close'), 1), 1).alias('__d2'), ts_rank(pl.col('volume') / mean(pl.col('volume'), 20), 5).alias('__tv')])
    df = df.with_columns([rank(pl.col('__tc')).alias('__r1'), rank(pl.col('__d2')).alias('__r2'), rank(pl.col('__tv')).alias('__r3')])
    return df.select((-1.0 * pl.col('__r1') * pl.col('__r2') * pl.col('__r3')).alias('gtja_142')).to_series()
