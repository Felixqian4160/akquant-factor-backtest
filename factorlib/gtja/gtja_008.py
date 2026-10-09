"""gtja_008 — standalone gtja factor.

GTJA Alpha #008 — Negated rank of 4d delta of HL10+VWAP80 weighted price.

Guotai Junan Formula
--------------------
    RANK(DELTA(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4) * -1)

Daic115 wrote this as ``-1*(H+L)*0.1 + VWAP*0.8`` (different op
precedence) which differs from the spec. We follow Daic115 for
parity with the reference parquet.

Required panel columns: ``high``, ``low``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_008 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #008 — Negated rank of 4d delta of HL10+VWAP80 weighted price.

    Guotai Junan Formula
    --------------------
        RANK(DELTA(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4) * -1)

    Daic115 wrote this as ``-1*(H+L)*0.1 + VWAP*0.8`` (different op
    precedence) which differs from the spec. We follow Daic115 for
    parity with the reference parquet.

    Required panel columns: ``high``, ``low``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = -1.0 * (pl.col('high') + pl.col('low')) * 0.1 + pl.col('vwap') * 0.8
    staged = panel.with_columns(delta(val, 4).alias('__g008_d'))
    return staged.select(rank(pl.col('__g008_d')).alias('gtja_008').cast(pl.Float64)).to_series()
