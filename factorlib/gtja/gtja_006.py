"""gtja_006 — standalone gtja factor.

GTJA Alpha #006 — Negated rank of sign of 4d weighted O/H delta.

Guotai Junan Formula
--------------------
    (RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4))) * -1)

Required panel columns: ``open``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_006 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #006 — Negated rank of sign of 4d weighted O/H delta.

    Guotai Junan Formula
    --------------------
        (RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4))) * -1)

    Required panel columns: ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = pl.col('open') * 0.85 + pl.col('high') * 0.15
    sgn = sign_(delta(val, 4))
    staged = panel.with_columns(sgn.alias('__g006_sgn'))
    return staged.select((-1.0 * rank(pl.col('__g006_sgn'))).alias('gtja_006').cast(pl.Float64)).to_series()
