"""gtja_049 — standalone gtja factor.

GTJA Alpha #049 — Down-day range share over 12d.

Guotai Junan Formula
--------------------
    cond = (H + L) >= (DELAY(H, 1) + DELAY(L, 1))
    part = MAX(|H - DELAY(H, 1)|, |L - DELAY(L, 1)|)
    SUM((!cond ? part : 0), 12) /
    (SUM((!cond ? part : 0), 12) + SUM((cond ? part : 0), 12))

Required panel columns: ``high``, ``low``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_049 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #049 — Down-day range share over 12d.

    Guotai Junan Formula
    --------------------
        cond = (H + L) >= (DELAY(H, 1) + DELAY(L, 1))
        part = MAX(|H - DELAY(H, 1)|, |L - DELAY(L, 1)|)
        SUM((!cond ? part : 0), 12) /
        (SUM((!cond ? part : 0), 12) + SUM((cond ? part : 0), 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col('high')
    lw = pl.col('low')
    cond = h + lw >= delay(h, 1) + delay(lw, 1)
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_dn = sumif(part, 12, ~cond)
    s_up = sumif(part, 12, cond)
    return panel.select((s_dn / (s_dn + s_up)).alias('gtja_049').cast(pl.Float64)).to_series()
