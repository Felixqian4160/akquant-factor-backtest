"""gtja_144 — standalone gtja factor.

GTJA #144 — Down-day average abs-return / log-amount.

Daic115's reference uses ``log(amount)`` rather than raw amount in
the denominator (deviation from the paper). We follow Daic115 for
parity.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_144 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #144 — Down-day average abs-return / log-amount.

    Daic115's reference uses ``log(amount)`` rather than raw amount in
    the denominator (deviation from the paper). We follow Daic115 for
    parity.
    """
    pc = delay(pl.col('close'), 1)
    abs_ret = (pl.col('close') / pc - 1.0).abs() / log_(pl.col('amount'))
    cond = pl.col('close') < pc
    masked = pl.when(cond).then(abs_ret).otherwise(0.0)
    expr = (sum_(masked, 20) / count_(cond, 20)).alias('gtja_144')
    return panel.select(expr).to_series()
