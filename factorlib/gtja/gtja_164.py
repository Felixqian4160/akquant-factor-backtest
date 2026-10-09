"""gtja_164 — standalone gtja factor.

GTJA #164 — Daic115 reference (reciprocal-diff stochastic, SMA13 smoothed).

Note: Daic115's parens in the original are slightly off — we follow
their parsing literally for parity.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_164 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #164 — Daic115 reference (reciprocal-diff stochastic, SMA13 smoothed).

    Note: Daic115's parens in the original are slightly off — we follow
    their parsing literally for parity.
    """
    pc = delay(pl.col('close'), 1)
    diff = pl.col('close') - pc
    cond = pl.col('close') > pc
    rec = pl.when(cond).then(1.0 / diff).otherwise(1.0)
    inner = rec - ts_min(rec, 12) / (pl.col('high') - pl.col('low')) * 100.0
    expr = sma(inner, 13, 2).alias('gtja_164')
    return panel.select(expr).to_series()
