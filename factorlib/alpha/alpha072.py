"""alpha072 — standalone alpha factor.

Alpha #072 — Decayed mid-price vs adv40 corr / decayed VWAP-volume corr.

WorldQuant Formula
------------------
    rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
    / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

Legacy AQML Expression
----------------------
    Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10))
    / Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))

Polars Implementation Notes
---------------------------
Two TS chains each ending in CS rank; the final result is their ratio.

Required panel columns: ``high``, ``low``, ``adv40``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha072 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #072 — Decayed mid-price vs adv40 corr / decayed VWAP-volume corr.

    WorldQuant Formula
    ------------------
        rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
        / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

    Legacy AQML Expression
    ----------------------
        Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10))
        / Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))

    Polars Implementation Notes
    ---------------------------
    Two TS chains each ending in CS rank; the final result is their ratio.

    Required panel columns: ``high``, ``low``, ``adv40``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    staged = panel.with_columns(ts_rank(pl.col('vwap'), 4).alias('__a072_trv'), ts_rank(pl.col('volume'), 19).alias('__a072_trvol'))
    staged2 = staged.with_columns(ts_corr(mid, pl.col('adv40'), 9).alias('__a072_corr1'), ts_corr(pl.col('__a072_trv'), pl.col('__a072_trvol'), 7).alias('__a072_corr2'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a072_corr1'), 10).alias('__a072_dl1'), ts_decay_linear(pl.col('__a072_corr2'), 3).alias('__a072_dl2'))
    return staged3.select((cs_rank(pl.col('__a072_dl1')) / cs_rank(pl.col('__a072_dl2'))).alias('alpha072')).to_series()
