"""gtja_051 — standalone gtja factor.

GTJA Alpha #051 — Down-share asymmetric range ratio over 12d.

Guotai Junan Formula
--------------------
    cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
    cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
    part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
    SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12))

Required panel columns: ``high``, ``low``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_051 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #051 — Down-share asymmetric range ratio over 12d.

    Guotai Junan Formula
    --------------------
        cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
        cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
        part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
        SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col('high')
    lw = pl.col('low')
    cond1 = h + lw <= delay(h, 1) + delay(lw, 1)
    cond2 = h + lw >= delay(h, 1) + delay(lw, 1)
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_a = sumif(part, 12, ~cond1)
    s_b = sumif(part, 12, ~cond2)
    expr = s_a / (s_a + s_b + 1e-07)
    return panel.select(expr.alias('gtja_051').cast(pl.Float64)).to_series()
