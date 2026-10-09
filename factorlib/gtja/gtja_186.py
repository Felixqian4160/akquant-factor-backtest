"""gtja_186 — standalone gtja factor.

GTJA #186 — Smoothed DI-difference oscillator.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_186 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #186 — Smoothed DI-difference oscillator."""
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
    p3 = (p1 - p2).abs() / (p1 + p2) * 100.0
    expr = ((mean(p3, 6) + delay(mean(p3, 6), 6)) / 2.0).alias('gtja_186')
    return panel.select(expr).to_series()
