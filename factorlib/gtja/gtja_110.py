"""gtja_110 — standalone gtja factor.

GTJA #110 — SUM(MAX(0,H-prev_C),20) / SUM(MAX(0,prev_C-L),20) * 100.

Buying-pressure / selling-pressure ratio over 20 days.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_110 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #110 — SUM(MAX(0,H-prev_C),20) / SUM(MAX(0,prev_C-L),20) * 100.

    Buying-pressure / selling-pressure ratio over 20 days.
    """
    pc = delay(pl.col('close'), 1)
    up = pl.when(pl.col('high') - pc > 0).then(pl.col('high') - pc).otherwise(0.0)
    dn = pl.when(pc - pl.col('low') > 0).then(pc - pl.col('low')).otherwise(0.0)
    expr = (sum_(up, 20) / sum_(dn, 20) * 100.0).alias('gtja_110')
    return panel.select(expr).to_series()
