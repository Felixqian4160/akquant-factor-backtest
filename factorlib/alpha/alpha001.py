"""alpha001 — standalone alpha factor.

Alpha #001 — Rank of squared-clip ts_argmax within past 5 days.

WorldQuant Formula (Kakushadze 2015, eq. 1)
-------------------------------------------
    rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

Legacy AQML Expression (deprecated 2026-04-29, kept as cross-check)
-------------------------------------------------------------------
    Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5

Polars Implementation Notes
---------------------------
1. Conditional input: when returns < 0 use 20-day std of returns
   (volatility regime), otherwise use raw close
2. Square with sign preservation amplifies extreme moves
3. ts_argmax: position (0..4) of max within last 5 rows
4. Cross-section rank centered at 0.5 (subtract 0.5 -> [-0.5, +0.5])

Required panel columns: ``returns``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

References
----------
- Kakushadze 2015, "101 Formulaic Alphas", arXiv:1601.00991, eq. 1
- STHSF/alpha101 (MIT) for pandas reference impl

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha001 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #001 — Rank of squared-clip ts_argmax within past 5 days.

    WorldQuant Formula (Kakushadze 2015, eq. 1)
    -------------------------------------------
        rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

    Legacy AQML Expression (deprecated 2026-04-29, kept as cross-check)
    -------------------------------------------------------------------
        Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5

    Polars Implementation Notes
    ---------------------------
    1. Conditional input: when returns < 0 use 20-day std of returns
       (volatility regime), otherwise use raw close
    2. Square with sign preservation amplifies extreme moves
    3. ts_argmax: position (0..4) of max within last 5 rows
    4. Cross-section rank centered at 0.5 (subtract 0.5 -> [-0.5, +0.5])

    Required panel columns: ``returns``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``

    References
    ----------
    - Kakushadze 2015, "101 Formulaic Alphas", arXiv:1601.00991, eq. 1
    - STHSF/alpha101 (MIT) for pandas reference impl
    """
    cond_input = pl.when(pl.col('returns') < 0).then(ts_std(pl.col('returns'), 20)).otherwise(pl.col('close'))
    sq = signed_power(cond_input, 2.0)
    arg = ts_argmax(sq, 5)
    staged = panel.with_columns(arg.alias('__a001_arg'))
    return staged.select((cs_rank(pl.col('__a001_arg')) - 0.5).alias('alpha001')).to_series()
