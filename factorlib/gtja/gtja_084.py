"""gtja_084 — standalone gtja factor.

GTJA Alpha #084 — 20d signed-volume sum (close-direction).

Guotai Junan Formula
--------------------
    SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 20)

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_084 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #084 — 20d signed-volume sum (close-direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    c = pl.col('close')
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    signed_vol = pl.when(cond_up).then(pl.col('volume')).otherwise(pl.when(cond_dn).then(-pl.col('volume')).otherwise(0.0))
    return panel.select(sum_(signed_vol, 20).alias('gtja_084').cast(pl.Float64)).to_series()
