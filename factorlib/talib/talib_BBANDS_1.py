"""talib_BBANDS_1 — canonical TA-Lib formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/rebuild_panel_with_talib.py
Indicator: BBANDS
Output kind: T3; output index: 1
"""
from __future__ import annotations

import akquant.talib as tq
import polars as pl
from factorlib._ops.talib_ops import compute_talib


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute talib_BBANDS_1 from sorted OHLCV panel rows."""
    return compute_talib(
        panel,
        lambda c, h, l, o, v: tq.BBANDS(c, timeperiod=20, as_series=True),
        name="talib_BBANDS_1",
        output_index=1,
    )
