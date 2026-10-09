"""gtja_143 — standalone gtja factor.

GTJA #143 — Cumulative product of up-day returns (recursive SELF).

Guotai Junan Formula
--------------------
    CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF

The recursive ``SELF`` term makes the formula a path-dependent recursion:
on up days the running value is *multiplied* by the up-day return, on
other days it is carried forward. Two reasonable interpretations:

    (a) literal: SELF_t = ratio_t * SELF_{t-1}, where ratio_t is the
        up-day raw return ~ 0.02. This decays SELF toward 0 quickly and
        is economically meaningless.
    (b) compounded: SELF_t = (1 + ratio_t) * SELF_{t-1} = close_t /
        delay(close, 1)_t * SELF_{t-1}, which gives the cumulative
        return path of a "buy-and-hold-on-up-days" strategy.

Daic115 leaves the body commented out; we adopt (b) — the economically
meaningful interpretation that matches how related GTJA factors (#018,
#053) compose returns. SELF starts at 1.0; non-up days carry forward
unchanged. The result is monotone non-decreasing per stock and grows
roughly as the cumulative product of up-day prices.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``normal``. Quality flag: ``0``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_143 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #143 — Cumulative product of up-day returns (recursive SELF).

    Guotai Junan Formula
    --------------------
        CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF

    The recursive ``SELF`` term makes the formula a path-dependent recursion:
    on up days the running value is *multiplied* by the up-day return, on
    other days it is carried forward. Two reasonable interpretations:

        (a) literal: SELF_t = ratio_t * SELF_{t-1}, where ratio_t is the
            up-day raw return ~ 0.02. This decays SELF toward 0 quickly and
            is economically meaningless.
        (b) compounded: SELF_t = (1 + ratio_t) * SELF_{t-1} = close_t /
            delay(close, 1)_t * SELF_{t-1}, which gives the cumulative
            return path of a "buy-and-hold-on-up-days" strategy.

    Daic115 leaves the body commented out; we adopt (b) — the economically
    meaningful interpretation that matches how related GTJA factors (#018,
    #053) compose returns. SELF starts at 1.0; non-up days carry forward
    unchanged. The result is monotone non-decreasing per stock and grows
    roughly as the cumulative product of up-day prices.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``. Quality flag: ``0``.
    """
    delayed = pl.col('close').shift(1).over(TS_PART)
    factor = pl.when(pl.col('close') > delayed).then(pl.col('close') / delayed).when(delayed.is_null()).then(None).otherwise(1.0)
    staged = panel.with_columns(factor.alias('__g143_factor'))
    staged = staged.with_columns(pl.col('__g143_factor').fill_null(1.0).cum_prod().over(TS_PART).alias('__g143_cum'))
    staged = staged.with_columns(pl.when(delayed.is_null()).then(None).otherwise(pl.col('__g143_cum')).alias('gtja_143').cast(pl.Float64))
    return staged.select('gtja_143').to_series()
