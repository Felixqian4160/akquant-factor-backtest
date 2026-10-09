"""talib_OBV — canonical TA-Lib formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/rebuild_panel_with_talib.py
Indicator: OBV
Output kind: S; output index: None
"""
from __future__ import annotations

import akquant.talib as tq
import polars as pl
from factorlib._ops.talib_ops import compute_talib


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute talib_OBV from sorted OHLCV panel rows."""
    return compute_talib(
        panel,
        lambda c, h, l, o, v: tq.OBV(c, v, as_series=True),
        name="talib_OBV",
        output_index=None,
    )
