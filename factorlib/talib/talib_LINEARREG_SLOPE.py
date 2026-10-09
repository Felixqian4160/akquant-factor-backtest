"""talib_LINEARREG_SLOPE — canonical TA-Lib formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/rebuild_panel_with_talib.py
Indicator: LINEARREG_SLOPE
Output kind: S; output index: None
"""
from __future__ import annotations

import akquant.talib as tq
import polars as pl
from factorlib._ops.talib_ops import compute_talib


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute talib_LINEARREG_SLOPE from sorted OHLCV panel rows."""
    return compute_talib(
        panel,
        lambda c, h, l, o, v: tq.LINEARREG_SLOPE(c, timeperiod=14, as_series=True),
        name="talib_LINEARREG_SLOPE",
        output_index=None,
    )
