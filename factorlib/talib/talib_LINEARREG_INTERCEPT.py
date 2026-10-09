"""talib_LINEARREG_INTERCEPT — canonical TA-Lib formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/rebuild_panel_with_talib.py
Indicator: LINEARREG_INTERCEPT
Output kind: S; output index: None
"""
from __future__ import annotations

import akquant.talib as tq
import polars as pl
from factorlib._ops.talib_ops import compute_talib


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute talib_LINEARREG_INTERCEPT from sorted OHLCV panel rows."""
    return compute_talib(
        panel,
        lambda c, h, l, o, v: tq.LINEARREG_INTERCEPT(c, timeperiod=14, as_series=True),
        name="talib_LINEARREG_INTERCEPT",
        output_index=None,
    )
