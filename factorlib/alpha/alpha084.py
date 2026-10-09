"""alpha084 — standalone alpha factor.

Alpha #084 — VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised).

WorldQuant Formula (Kakushadze 2015, eq. 84)
--------------------------------------------
    SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
                delta(close, 4.96796))

Legacy AQML Expression (linearised exponent)
--------------------------------------------
    SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)

Polars Implementation Notes
---------------------------
1. The original WorldQuant formula uses a fractional-day rolling max
   and an exponent equal to a per-row delta; our migrated AQML form
   linearises the exponent to ``1.0`` and rounds the windows to
   integers (15 and 21). We follow the migrated form verbatim — this
   matches both the legacy AQML evaluator and the STHSF reference.
2. With exponent=1 the operation degenerates to identity (sign · |x|),
   so the alpha simplifies to just the rolling rank itself.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha084 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #084 — VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised).

    WorldQuant Formula (Kakushadze 2015, eq. 84)
    --------------------------------------------
        SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
                    delta(close, 4.96796))

    Legacy AQML Expression (linearised exponent)
    --------------------------------------------
        SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)

    Polars Implementation Notes
    ---------------------------
    1. The original WorldQuant formula uses a fractional-day rolling max
       and an exponent equal to a per-row delta; our migrated AQML form
       linearises the exponent to ``1.0`` and rounds the windows to
       integers (15 and 21). We follow the migrated form verbatim — this
       matches both the legacy AQML evaluator and the STHSF reference.
    2. With exponent=1 the operation degenerates to identity (sign · |x|),
       so the alpha simplifies to just the rolling rank itself.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap_minus_max = pl.col('vwap') - ts_max(pl.col('vwap'), 15)
    rk = ts_rank(vwap_minus_max, 21)
    expr = signed_power(rk, 1.0)
    return panel.select(expr.alias('alpha084').cast(pl.Float64)).to_series()
