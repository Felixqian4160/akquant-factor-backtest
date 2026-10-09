"""l_turna — Quantactix academic factor formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/v30_quantactix_factors.py
The formula is recomputed from panel columns; no factor values are copied.
"""
from __future__ import annotations

import polars as pl


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute l_turna from a stock_code/trade_date-sorted panel."""
    required = ["stock_code", "trade_date", "close", "amount", "circ_cap", "returns"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"l_turna: missing columns {missing}")
    return panel.select((pl.col("amount") / pl.col("circ_cap").clip(lower_bound=1)).rolling_mean(252).over("stock_code").clip(lower_bound=1e-9).log().alias("l_turna")).to_series()
