"""gtja_129 — standalone gtja factor.

GTJA #129 — SUM(IFELSE(dC<0, |dC|, 0), 12). Down-move 12-day cumsum.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_129 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #129 — SUM(IFELSE(dC<0, |dC|, 0), 12). Down-move 12-day cumsum."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    expr = sum_(pl.when(dc < 0).then(dc.abs()).otherwise(0.0), 12).alias('gtja_129')
    return panel.select(expr).to_series()
