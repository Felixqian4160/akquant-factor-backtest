"""gtja_010 — standalone gtja factor.

GTJA Alpha #010 — Rank of conditional 20d-return-std-or-close squared.

Guotai Junan Formula
--------------------
    (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))

Daic115 implementation uses ``np.maximum(alpha, 5)`` which is an
elementwise scalar floor (probably an upstream bug — should be
``ts_max(alpha, 5)``). We match Daic115 for reference parity.

Required panel columns: ``close``, ``returns``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_010 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #010 — Rank of conditional 20d-return-std-or-close squared.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))

    Daic115 implementation uses ``np.maximum(alpha, 5)`` which is an
    elementwise scalar floor (probably an upstream bug — should be
    ``ts_max(alpha, 5)``). We match Daic115 for reference parity.

    Required panel columns: ``close``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    ret = pl.col('returns')
    cond_branch = ifelse(ret < 0.0, std_(ret, 20), pl.col('close'))
    inner = cond_branch * cond_branch
    floored = pl.max_horizontal(inner, pl.lit(5.0))
    staged = panel.with_columns(floored.alias('__g010_inner'))
    return staged.select(rank(pl.col('__g010_inner')).alias('gtja_010').cast(pl.Float64)).to_series()
