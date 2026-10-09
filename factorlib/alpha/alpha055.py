"""alpha055 — standalone alpha factor.

Alpha #055 — Negative correlation between %K rank and volume rank.

WorldQuant Formula
------------------
    -1 * correlation(
        rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
        rank(volume),
        6,
    )

Legacy AQML Expression
----------------------
    -1 * Ts_Corr(
        Rank((close - Ts_Min(low, 12)) / (Ts_Max(high, 12) - Ts_Min(low, 12))),
        Rank(volume),
        6,
    )

Polars Implementation Notes
---------------------------
Compute the %K series TS-wise, CS rank it, CS rank volume, then TS corr.

Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha055 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #055 — Negative correlation between %K rank and volume rank.

    WorldQuant Formula
    ------------------
        -1 * correlation(
            rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
            rank(volume),
            6,
        )

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(
            Rank((close - Ts_Min(low, 12)) / (Ts_Max(high, 12) - Ts_Min(low, 12))),
            Rank(volume),
            6,
        )

    Polars Implementation Notes
    ---------------------------
    Compute the %K series TS-wise, CS rank it, CS rank volume, then TS corr.

    Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    pct_k = (pl.col('close') - ts_min(pl.col('low'), 12)) / (ts_max(pl.col('high'), 12) - ts_min(pl.col('low'), 12))
    staged = panel.with_columns(pct_k.alias('__a055_pk'))
    staged2 = staged.with_columns(cs_rank(pl.col('__a055_pk')).alias('__a055_rk'), cs_rank(pl.col('volume')).alias('__a055_rv'))
    return staged2.select((-1.0 * ts_corr(pl.col('__a055_rk'), pl.col('__a055_rv'), 6)).alias('alpha055')).to_series()
