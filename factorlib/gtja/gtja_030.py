"""gtja_030 — standalone gtja factor.

GTJA Alpha #030 — STUB (Fama-French residual^2 WMA).

Guotai Junan Formula
--------------------
    WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

Daic115 marks this as ``unfinished=True`` and returns ``None``.
A faithful implementation requires Fama-French factor exposures
which the synthetic panel does not provide. We return an all-NaN
series and tag with quality_flag=2 (stub). Reference parquet does
not include gtja_030; reference test is skipped.

Direction: ``normal``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_030 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #030 — STUB (Fama-French residual^2 WMA).

    Guotai Junan Formula
    --------------------
        WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

    Daic115 marks this as ``unfinished=True`` and returns ``None``.
    A faithful implementation requires Fama-French factor exposures
    which the synthetic panel does not provide. We return an all-NaN
    series and tag with quality_flag=2 (stub). Reference parquet does
    not include gtja_030; reference test is skipped.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select((pl.col('close') * 0.0 + float('nan')).alias('gtja_030').cast(pl.Float64)).to_series()
