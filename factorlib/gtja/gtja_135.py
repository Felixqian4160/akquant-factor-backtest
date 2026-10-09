"""gtja_135 — standalone gtja factor.

GTJA #135 — SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1).

The leading nulls (from DELAY(C,20) and DELAY(...,1)) must propagate
through the SMA — :func:`_ops.sma` honours ``ignore_nulls`` so that
nulls do not pollute the EWMA recursion.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_135 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #135 — SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1).

    The leading nulls (from DELAY(C,20) and DELAY(...,1)) must propagate
    through the SMA — :func:`_ops.sma` honours ``ignore_nulls`` so that
    nulls do not pollute the EWMA recursion.
    """
    inner = pl.col('close') / delay(pl.col('close'), 20)
    expr = sma(delay(inner, 1), 20, 1).alias('gtja_135')
    return panel.select(expr).to_series()
