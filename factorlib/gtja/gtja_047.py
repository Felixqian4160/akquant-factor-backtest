"""gtja_047 — standalone gtja factor.

GTJA Alpha #047 — Smoothed RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 9, 1).

Required panel columns: ``high``, ``low``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_047 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #047 — Smoothed RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 9, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col('high'), 6)
    lmin = ts_min(pl.col('low'), 6)
    raw = (hmax - pl.col('close')) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 9, 1).alias('gtja_047').cast(pl.Float64)).to_series()
