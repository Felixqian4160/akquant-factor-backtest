"""gtja_187 — standalone gtja factor.

GTJA #187 — SUM(O<=prev_O ? 0 : MAX(H-O, O-prev_O), 20). Open-gap up cumsum.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_187 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #187 — SUM(O<=prev_O ? 0 : MAX(H-O, O-prev_O), 20). Open-gap up cumsum."""
    po = delay(pl.col('open'), 1)
    body = pl.max_horizontal([pl.col('high') - pl.col('open'), pl.col('open') - po])
    masked = pl.when(pl.col('open') <= po).then(0.0).otherwise(body)
    expr = sum_(masked, 20).alias('gtja_187')
    return panel.select(expr).to_series()
