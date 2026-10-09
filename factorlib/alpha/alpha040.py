"""alpha040 — standalone alpha factor.

Alpha #040 — high-vol amplitude weighted by high-volume correlation.

WorldQuant Formula
------------------
    -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)

Polars Implementation Notes
---------------------------
1. CS rank of 10-day std of high prices (volatility regime).
2. 10-day rolling correlation between high and volume (price-volume
   confirmation).
3. Multiply with sign flip: large vol + positive corr ⇒ persistent
   breakout that mean-reverts.

Required panel columns: ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha040 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #040 — high-vol amplitude weighted by high-volume correlation.

    WorldQuant Formula
    ------------------
        -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)

    Polars Implementation Notes
    ---------------------------
    1. CS rank of 10-day std of high prices (volatility regime).
    2. 10-day rolling correlation between high and volume (price-volume
       confirmation).
    3. Multiply with sign flip: large vol + positive corr ⇒ persistent
       breakout that mean-reverts.

    Required panel columns: ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    std_h = ts_std(pl.col('high'), 10)
    corr_hv = ts_corr_safe(pl.col('high'), pl.col('volume'), 10)
    staged = panel.with_columns(std_h.alias('__a040_std'), corr_hv.alias('__a040_corr'))
    return staged.select((-1.0 * cs_rank(pl.col('__a040_std')) * pl.col('__a040_corr')).alias('alpha040')).to_series()
