"""gtja_156 — standalone gtja factor.

GTJA #156 — MAX of two decay-linear ranks * -1.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_156 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #156 — MAX of two decay-linear ranks * -1."""
    a = pl.col('vwap') - delay(pl.col('vwap'), 5)
    b_inner = pl.col('open') * 0.15 + pl.col('low') * 0.85
    b = -delta(b_inner, 2) / b_inner
    df = panel.with_columns([decay_linear(a, 3).alias('__dla'), decay_linear(b, 3).alias('__dlb')])
    df = df.with_columns([rank(pl.col('__dla')).alias('__r1'), rank(pl.col('__dlb')).alias('__r2')])
    expr = (pl.max_horizontal([pl.col('__r1'), pl.col('__r2')]) * -1.0).alias('gtja_156')
    return df.select(expr).to_series()
