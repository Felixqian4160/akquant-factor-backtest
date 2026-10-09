"""gtja_082 — standalone gtja factor.

GTJA Alpha #082 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 20, 1).

Required panel columns: ``high``, ``low``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_082 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #082 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 20, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col('high'), 6)
    lmin = ts_min(pl.col('low'), 6)
    raw = (hmax - pl.col('close')) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 20, 1).alias('gtja_082').cast(pl.Float64)).to_series()
