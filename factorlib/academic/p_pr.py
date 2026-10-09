"""p_pr — Quantactix academic factor formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/v30_quantactix_factors.py
The formula is recomputed from panel columns; no factor values are copied.
"""
from __future__ import annotations

import polars as pl


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute p_pr from a stock_code/trade_date-sorted panel."""
    required = ["stock_code", "trade_date", "close", "amount", "circ_cap", "returns"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"p_pr: missing columns {missing}")
    return panel.select(pl.col("close").clip(lower_bound=0.01).log().alias("p_pr")).to_series()
