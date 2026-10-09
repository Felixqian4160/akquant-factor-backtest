"""gtja_076 — standalone gtja factor.

GTJA Alpha #076 — STD(|ret|/V, 20) / MEAN(|ret|/V, 20).

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_076 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #076 — STD(|ret|/V, 20) / MEAN(|ret|/V, 20).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    c = pl.col('close')
    rel_ret_per_v = abs_(c / delay(c, 1) - 1.0) / pl.col('volume')
    return panel.select((std_(rel_ret_per_v, 20) / mean(rel_ret_per_v, 20)).alias('gtja_076').cast(pl.Float64)).to_series()
