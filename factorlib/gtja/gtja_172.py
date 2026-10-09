"""gtja_172 — standalone gtja factor.

GTJA #172 — DI-difference oscillator (DX) averaged over 6 days.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_172 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #172 — DI-difference oscillator (DX) averaged over 6 days."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pl.col('high') - pc).abs()
    c = (pl.col('low') - pc).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    hd = pl.col('high') - delay(pl.col('high'), 1)
    ld = delay(pl.col('low'), 1) - pl.col('low')
    pos_ld = pl.when((ld > 0) & (ld > hd)).then(ld).otherwise(0.0)
    pos_hd = pl.when((hd > 0) & (hd > ld)).then(hd).otherwise(0.0)
    sum_tr = sum_(tr, 14)
    p1 = sum_(pos_ld, 14) * 100.0 / sum_tr
    p2 = sum_(pos_hd, 14) * 100.0 / sum_tr
    expr = mean((p1 - p2).abs() / (p1 + p2) * 100.0, 6).alias('gtja_172')
    return panel.select(expr).to_series()
