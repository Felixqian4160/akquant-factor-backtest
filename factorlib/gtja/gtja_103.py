"""gtja_103 — standalone gtja factor.

GTJA #103 — (20-LOWDAY(LOW,20))/20*100 — recency of recent low.

Implemented via :func:`ts_min` + per-stock back-search using a
closed-form: distance-to-min as ``20 - argmin``. We use ``regbeta``-
style trick: Daic115 has a slow ``LOWDAY`` here. We fall back to the
operator's slow path :func:`_ops.lowday`.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_103 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank, lowday

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #103 — (20-LOWDAY(LOW,20))/20*100 — recency of recent low.

    Implemented via :func:`ts_min` + per-stock back-search using a
    closed-form: distance-to-min as ``20 - argmin``. We use ``regbeta``-
    style trick: Daic115 has a slow ``LOWDAY`` here. We fall back to the
    operator's slow path :func:`_ops.lowday`.
    """

    expr = ((20.0 - lowday(pl.col('low'), 20)) / 20.0 * 100.0).alias('gtja_103')
    return panel.select(expr).to_series()
