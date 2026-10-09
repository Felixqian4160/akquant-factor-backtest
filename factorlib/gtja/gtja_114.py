"""gtja_114 — standalone gtja factor.

GTJA #114 — Range-over-MA scaled by VWAP-Close gap.

Numerical safety
----------------
Three potential div-by-zero spots, all hit on A-share limit-up days
(一字板, high=low=close=vwap):
  1. ``(high - low) / mean(close, 5)`` — denominator on suspended /
     IPO bars can be 0
  2. ``part / (vwap - close)`` — vwap-close=0 on limit-up days
  3. outer ``(rpd*rrv) / den`` — den ≈ 0 when part = 0

Original code had ``+ 1e-7`` only for #2, which let ~7000 inf cells
through per year. Replaced all three with ``safe_div``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_114 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #114 — Range-over-MA scaled by VWAP-Close gap.

    Numerical safety
    ----------------
    Three potential div-by-zero spots, all hit on A-share limit-up days
    (一字板, high=low=close=vwap):
      1. ``(high - low) / mean(close, 5)`` — denominator on suspended /
         IPO bars can be 0
      2. ``part / (vwap - close)`` — vwap-close=0 on limit-up days
      3. outer ``(rpd*rrv) / den`` — den ≈ 0 when part = 0

    Original code had ``+ 1e-7`` only for #2, which let ~7000 inf cells
    through per year. Replaced all three with ``safe_div``.
    """
    part = safe_div(pl.col('high') - pl.col('low'), mean(pl.col('close'), 5))
    df = panel.with_columns([part.alias('__p'), delay(part, 2).alias('__pd')])
    df = df.with_columns([rank(pl.col('__pd')).alias('__rpd'), rank(rank(pl.col('volume'))).alias('__rrv')])
    den = safe_div(pl.col('__p'), pl.col('vwap') - pl.col('close'))
    df = df.with_columns(den.alias('__den'))
    expr = safe_div(pl.col('__rpd') * pl.col('__rrv'), pl.col('__den')).alias('gtja_114')
    return df.select(expr).to_series()
