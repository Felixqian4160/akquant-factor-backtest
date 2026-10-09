"""gtja_063 — standalone gtja factor.

GTJA Alpha #063 — 6-day RSI: SMA(MAX(C-C-1, 0), 6, 1) / SMA(|C-C-1|, 6, 1) × 100.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_063 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #063 — 6-day RSI: SMA(MAX(C-C-1, 0), 6, 1) / SMA(|C-C-1|, 6, 1) × 100.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col('vwap')
    diff = v - delay(v, 1)
    up = pl.max_horizontal(diff, pl.lit(0.0))
    return panel.select((sma(up, 6, 1) / sma(abs_(diff), 6, 1) * 100.0).alias('gtja_063').cast(pl.Float64)).to_series()
