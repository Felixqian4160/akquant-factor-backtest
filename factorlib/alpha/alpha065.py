"""alpha065 — standalone alpha factor.

Alpha #065 — Volume-weighted price vs adv60 corr rank vs open-min rank.

WorldQuant Formula
------------------
    if rank(correlation(open*0.0078 + vwap*0.9922, sum(adv60, 9), 6))
          < rank(open - ts_min(open, 14))
    then -1 else 1

Legacy AQML Expression
----------------------
    If(Rank(Ts_Corr(open*0.0078 + vwap*0.9922, Ts_Sum(adv60, 9), 6))
            < Rank(open - Ts_Min(open, 14)), -1, 1)

Polars Implementation Notes
---------------------------
AQML uses ``Ts_Sum`` (rolling sum); STHSF reference uses ``sma`` (rolling
mean), so STHSF parity may diverge by a constant scale factor that
drops out of CS rank — but tie-breaking around boundary cases can
flip a few signs. We follow AQML.

Required panel columns: ``open``, ``vwap``, ``adv60``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha065 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #065 — Volume-weighted price vs adv60 corr rank vs open-min rank.

    WorldQuant Formula
    ------------------
        if rank(correlation(open*0.0078 + vwap*0.9922, sum(adv60, 9), 6))
              < rank(open - ts_min(open, 14))
        then -1 else 1

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(open*0.0078 + vwap*0.9922, Ts_Sum(adv60, 9), 6))
                < Rank(open - Ts_Min(open, 14)), -1, 1)

    Polars Implementation Notes
    ---------------------------
    AQML uses ``Ts_Sum`` (rolling sum); STHSF reference uses ``sma`` (rolling
    mean), so STHSF parity may diverge by a constant scale factor that
    drops out of CS rank — but tie-breaking around boundary cases can
    flip a few signs. We follow AQML.

    Required panel columns: ``open``, ``vwap``, ``adv60``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('open') * 0.0078 + pl.col('vwap') * 0.9922
    sum_adv60 = ts_sum(pl.col('adv60'), 9)
    staged = panel.with_columns(ts_corr(weighted, sum_adv60, 6).alias('__a065_corr'), (pl.col('open') - ts_min(pl.col('open'), 14)).alias('__a065_omin'))
    staged2 = staged.with_columns(cs_rank(pl.col('__a065_corr')).alias('__a065_rcorr'), cs_rank(pl.col('__a065_omin')).alias('__a065_romin'))
    return staged2.select(if_then_else(pl.col('__a065_rcorr') < pl.col('__a065_romin'), -1.0, 1.0).cast(pl.Float64).alias('alpha065')).to_series()
