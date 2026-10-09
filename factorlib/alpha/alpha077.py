"""alpha077 — standalone alpha factor.

Alpha #077 — Min of two decayed-rank features.

WorldQuant Formula
------------------
    min(
      rank(decay_linear((high+low)/2 + high - vwap - high, 20)),
      rank(decay_linear(correlation((high+low)/2, adv40, 3), 6))
    )

Legacy AQML Expression
----------------------
    Min(
      Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)),
      Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6))
    )

Polars Implementation Notes
---------------------------
Two parallel TS+CS chains combined by element-wise min.

Required panel columns: ``high``, ``low``, ``vwap``, ``adv40``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha077 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #077 — Min of two decayed-rank features.

    WorldQuant Formula
    ------------------
        min(
          rank(decay_linear((high+low)/2 + high - vwap - high, 20)),
          rank(decay_linear(correlation((high+low)/2, adv40, 3), 6))
        )

    Legacy AQML Expression
    ----------------------
        Min(
          Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)),
          Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6))
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel TS+CS chains combined by element-wise min.

    Required panel columns: ``high``, ``low``, ``vwap``, ``adv40``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    payload = mid + pl.col('high') - pl.col('vwap') - pl.col('high')
    staged = panel.with_columns(ts_decay_linear(payload, 20).alias('__a077_dl1'), ts_decay_linear(ts_corr(mid, pl.col('adv40'), 3), 6).alias('__a077_dl2'))
    staged2 = staged.with_columns(cs_rank(pl.col('__a077_dl1')).alias('__a077_r1'), cs_rank(pl.col('__a077_dl2')).alias('__a077_r2'))
    return staged2.select(pmin(pl.col('__a077_r1'), pl.col('__a077_r2')).alias('alpha077')).to_series()
