"""gtja_003 — standalone gtja factor.

GTJA Alpha #003 — 6-day sum of close-vs-extreme conditional flow.

Guotai Junan Formula
--------------------
    SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
         MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

Required panel columns: ``close``, ``low``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_003 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #003 — 6-day sum of close-vs-extreme conditional flow.

    Guotai Junan Formula
    --------------------
        SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
             MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    delay1 = delay(pl.col('close'), 1)
    cond_up = pl.col('close') > delay1
    cond_dn = pl.col('close') < delay1
    pivot = ifelse(cond_up, pl.min_horizontal(pl.col('low'), delay1), pl.max_horizontal(pl.col('high'), delay1))
    inner = pl.when(cond_up | cond_dn).then(pl.col('close') - pivot).otherwise(0.0)
    return panel.select(sum_(inner, 6).alias('gtja_003').cast(pl.Float64)).to_series()
