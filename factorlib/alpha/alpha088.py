"""alpha088 — standalone alpha factor.

Alpha #088 — Min of decayed rank-spread and decayed correlation rank.

WorldQuant Formula
------------------
    min(
      rank(decay_linear(rank(open) + rank(low) - rank(high) - rank(close), 8)),
      ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
    )

Legacy AQML Expression
----------------------
    Min(
      Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - (Rank(high) + Rank(close)), 8)),
      Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3)
    )

Polars Implementation Notes
---------------------------
Two parallel chains, combined by element-wise min.

Required panel columns: ``open``, ``low``, ``high``, ``close``, ``adv60``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha088 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #088 — Min of decayed rank-spread and decayed correlation rank.

    WorldQuant Formula
    ------------------
        min(
          rank(decay_linear(rank(open) + rank(low) - rank(high) - rank(close), 8)),
          ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
        )

    Legacy AQML Expression
    ----------------------
        Min(
          Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - (Rank(high) + Rank(close)), 8)),
          Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3)
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel chains, combined by element-wise min.

    Required panel columns: ``open``, ``low``, ``high``, ``close``, ``adv60``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    spread = cs_rank(pl.col('open')) + cs_rank(pl.col('low')) - cs_rank(pl.col('high')) - cs_rank(pl.col('close'))
    staged = panel.with_columns(spread.alias('__a088_spread'), ts_rank(pl.col('close'), 8).alias('__a088_trc'), ts_rank(pl.col('adv60'), 21).alias('__a088_tra'))
    staged2 = staged.with_columns(ts_decay_linear(pl.col('__a088_spread'), 8).alias('__a088_dl1'), ts_corr(pl.col('__a088_trc'), pl.col('__a088_tra'), 8).alias('__a088_corr'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a088_corr'), 7).alias('__a088_dl2'), cs_rank(pl.col('__a088_dl1')).alias('__a088_r1'))
    p1 = pl.col('__a088_r1')
    p2 = ts_rank(pl.col('__a088_dl2'), 3)
    return staged3.select(pmin(p1, p2).alias('alpha088')).to_series()
