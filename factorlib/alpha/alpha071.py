"""alpha071 — standalone alpha factor.

Alpha #071 — Decayed correlation vs decayed weighted-price-square rank, max.

WorldQuant Formula
------------------
    max(
      ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16),
      ts_rank(decay_linear(rank(low + open - 2*vwap)^2, 16), 4)
    )

Legacy AQML Expression
----------------------
    Max(
      Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16),
      Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4)
    )

Polars Implementation Notes
---------------------------
Two parallel TS chains, combined element-wise via ``max_horizontal``.

Required panel columns: ``close``, ``adv180``, ``low``, ``open``, ``vwap``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha071 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #071 — Decayed correlation vs decayed weighted-price-square rank, max.

    WorldQuant Formula
    ------------------
        max(
          ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16),
          ts_rank(decay_linear(rank(low + open - 2*vwap)^2, 16), 4)
        )

    Legacy AQML Expression
    ----------------------
        Max(
          Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16),
          Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4)
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel TS chains, combined element-wise via ``max_horizontal``.

    Required panel columns: ``close``, ``adv180``, ``low``, ``open``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(ts_rank(pl.col('close'), 3).alias('__a071_trc'), ts_rank(pl.col('adv180'), 12).alias('__a071_tra'), cs_rank(pl.col('low') + pl.col('open') - pl.col('vwap') - pl.col('vwap')).alias('__a071_rlovw'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a071_trc'), pl.col('__a071_tra'), 18).alias('__a071_corr'), pl.col('__a071_rlovw').pow(2).alias('__a071_rsq'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a071_corr'), 4).alias('__a071_dl1'), ts_decay_linear(pl.col('__a071_rsq'), 16).alias('__a071_dl2'))
    p1 = ts_rank(pl.col('__a071_dl1'), 16)
    p2 = ts_rank(pl.col('__a071_dl2'), 4)
    return staged3.select(pmax(p1, p2).alias('alpha071')).to_series()
