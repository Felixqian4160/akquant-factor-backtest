"""alpha018 — standalone alpha factor.

Alpha #018 — body volatility plus body plus close-open correlation, ranked.

WorldQuant Formula
------------------
    -1 * rank((stddev(abs((close - open)), 5) +
               (close - open)) +
              correlation(close, open, 10))

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Std(Abs(close - open), 5) +
              (close - open) +
              Ts_Corr(close, open, 10))

Polars Implementation Notes
---------------------------
1. ``Ts_Std(Abs(close - open), 5)``: 5-day std of body magnitude.
2. Plus today's body ``close - open``.
3. Plus 10-day rolling correlation between close and open (NaN/inf
   replaced with null then handled by CS rank's nulls treatment).
4. CS rank, sign-flipped.

Required panel columns: ``close``, ``open``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha018 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #018 — body volatility plus body plus close-open correlation, ranked.

    WorldQuant Formula
    ------------------
        -1 * rank((stddev(abs((close - open)), 5) +
                   (close - open)) +
                  correlation(close, open, 10))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Std(Abs(close - open), 5) +
                  (close - open) +
                  Ts_Corr(close, open, 10))

    Polars Implementation Notes
    ---------------------------
    1. ``Ts_Std(Abs(close - open), 5)``: 5-day std of body magnitude.
    2. Plus today's body ``close - open``.
    3. Plus 10-day rolling correlation between close and open (NaN/inf
       replaced with null then handled by CS rank's nulls treatment).
    4. CS rank, sign-flipped.

    Required panel columns: ``close``, ``open``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    body = pl.col('close') - pl.col('open')
    body_std = ts_std(body.abs(), 5)
    corr = ts_corr_safe(pl.col('close'), pl.col('open'), 10)
    inner = body_std + body + corr
    staged = panel.with_columns(inner.alias('__a018_inner'))
    return staged.select((-1.0 * cs_rank(pl.col('__a018_inner'))).alias('alpha018')).to_series()
