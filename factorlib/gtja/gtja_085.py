"""gtja_085 — standalone gtja factor.

GTJA Alpha #085 — Volume-ratio TS-rank × negated 7d-close-delta TS-rank.

Guotai Junan Formula
--------------------
    TSRANK(V / MEAN(V, 20), 20) * TSRANK(-1 * DELTA(C, 7), 8)

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_085 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #085 — Volume-ratio TS-rank × negated 7d-close-delta TS-rank.

    Guotai Junan Formula
    --------------------
        TSRANK(V / MEAN(V, 20), 20) * TSRANK(-1 * DELTA(C, 7), 8)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    arm1 = ts_rank(vol_ratio, 20)
    arm2 = ts_rank(-1.0 * delta(pl.col('close'), 7), 8)
    return panel.select((arm1 * arm2).alias('gtja_085').cast(pl.Float64)).to_series()
