"""alpha073 — standalone alpha factor.

Alpha #073 — Negative max of decayed VWAP delta rank and blend reversal rank.

WorldQuant Formula
------------------
    -1 * max(
        rank(decay_linear(delta(vwap, 5), 3)),
        ts_rank(decay_linear(-delta(open*0.147155 + low*0.852845, 2)
                / (open*0.147155 + low*0.852845), 3), 17)
    )

Legacy AQML Expression
----------------------
    -1 * Max(
        Rank(Ts_DecayLinear(Delta(vwap, 5), 3)),
        Ts_Rank(Ts_DecayLinear(
            -1 * Delta(open * 0.147155 + low * 0.852845, 2)
            / (open * 0.147155 + low * 0.852845), 3), 17)
    )

Polars Implementation Notes
---------------------------
The paper's fractional windows round to the standard STHSF integer
windows: 5, 3, 2, 3 and 17.

Required panel columns: ``vwap``, ``open``, ``low``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha073 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #073 — Negative max of decayed VWAP delta rank and blend reversal rank.

    WorldQuant Formula
    ------------------
        -1 * max(
            rank(decay_linear(delta(vwap, 5), 3)),
            ts_rank(decay_linear(-delta(open*0.147155 + low*0.852845, 2)
                    / (open*0.147155 + low*0.852845), 3), 17)
        )

    Legacy AQML Expression
    ----------------------
        -1 * Max(
            Rank(Ts_DecayLinear(Delta(vwap, 5), 3)),
            Ts_Rank(Ts_DecayLinear(
                -1 * Delta(open * 0.147155 + low * 0.852845, 2)
                / (open * 0.147155 + low * 0.852845), 3), 17)
        )

    Polars Implementation Notes
    ---------------------------
    The paper's fractional windows round to the standard STHSF integer
    windows: 5, 3, 2, 3 and 17.

    Required panel columns: ``vwap``, ``open``, ``low``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    blend = pl.col('open') * 0.147155 + pl.col('low') * 0.852845
    staged1 = panel.with_columns(ts_decay_linear(delta(pl.col('vwap'), 5), 3).alias('__a073_p1_raw'), ts_decay_linear(-1.0 * delta(blend, 2) / blend, 3).alias('__a073_p2_raw'))
    staged2 = staged1.with_columns(cs_rank(pl.col('__a073_p1_raw')).alias('__a073_p1'), ts_rank(pl.col('__a073_p2_raw'), 17).alias('__a073_p2'))
    return staged2.select((-1.0 * pmax(pl.col('__a073_p1'), pl.col('__a073_p2'))).alias('alpha073').cast(pl.Float64)).to_series()
