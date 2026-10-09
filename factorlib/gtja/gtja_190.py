"""gtja_190 — standalone gtja factor.

GTJA #190 — Log-asymmetric return classifier.

Guotai Junan Formula
--------------------
    LOG((COUNT(p1>p2,20)-1) * SUMIF((p1-p2)^2,20,p1<p2) /
        ((COUNT(p1<p2,20)) * SUMIF((p1-p2)^2,20,p1>p2)))
    where p1 = C/prev_C - 1, p2 = (C/C-19)^(1/20) - 1

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_190 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #190 — Log-asymmetric return classifier.

    Guotai Junan Formula
    --------------------
        LOG((COUNT(p1>p2,20)-1) * SUMIF((p1-p2)^2,20,p1<p2) /
            ((COUNT(p1<p2,20)) * SUMIF((p1-p2)^2,20,p1>p2)))
        where p1 = C/prev_C - 1, p2 = (C/C-19)^(1/20) - 1
    """
    pc = delay(pl.col('close'), 1)
    pc19 = delay(pl.col('close'), 19)
    p1 = pl.col('close') / pc - 1.0
    p2 = (pl.col('close') / pc19).pow(1.0 / 20.0) - 1.0
    df = panel.with_columns([p1.alias('__p1'), p2.alias('__p2')])
    df = df.with_columns((pl.col('__p1') - pl.col('__p2')).pow(2).alias('__sq'))
    cond_gt = pl.col('__p1') > pl.col('__p2')
    cond_lt = pl.col('__p1') < pl.col('__p2')
    sumif_lt = sumif(pl.col('__sq'), 20, cond_lt)
    sumif_gt = sumif(pl.col('__sq'), 20, cond_gt)
    cnt_gt = count_(cond_gt, 20)
    cnt_lt = count_(cond_lt, 20)
    expr = log_((cnt_gt - 1.0) * sumif_lt / (cnt_lt * sumif_gt + 1e-12)).alias('gtja_190')
    return df.select(expr).to_series()
