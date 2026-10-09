"""gtja_166 — standalone gtja factor.

GTJA #166 — Errata (Daic115 simplification of skewness-of-returns).

Guotai Junan Formula (paper, with errata)
------------------------------------------
    -20 * 19^1.5 * SUM(part1 - mean(part1,20), 20) /
    ((20-1)*(20-2)*(SUM((part^2,20))^1.5))
    where part = CLOSE / DELAY(CLOSE,1)

Daic115 simplifies to:
    5 * SUM(part-1 - mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5

Listed in errata. We follow Daic115's simplification. Quality flag = 1.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_166 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #166 — Errata (Daic115 simplification of skewness-of-returns).

    Guotai Junan Formula (paper, with errata)
    ------------------------------------------
        -20 * 19^1.5 * SUM(part1 - mean(part1,20), 20) /
        ((20-1)*(20-2)*(SUM((part^2,20))^1.5))
        where part = CLOSE / DELAY(CLOSE,1)

    Daic115 simplifies to:
        5 * SUM(part-1 - mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5

    Listed in errata. We follow Daic115's simplification. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    part = pl.col('close') / delay(pl.col('close'), 1)
    p1 = part - 1.0
    df = panel.with_columns([(p1 - mean(p1, 20)).alias('__centered'), mean(part, 20).alias('__mp')])
    expr = (5.0 * sum_(pl.col('__centered'), 20) / sum_(pl.col('__mp').pow(2), 20).pow(1.5)).alias('gtja_166')
    return df.select(expr).to_series()
