"""p_m1 — Quantactix academic factor formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/v30_quantactix_factors.py
The formula is recomputed from panel columns; no factor values are copied.
"""
from __future__ import annotations

import polars as pl


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute p_m1 from a stock_code/trade_date-sorted panel."""
    required = ["stock_code", "trade_date", "close", "amount", "circ_cap", "returns"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"p_m1: missing columns {missing}")
    return panel.select((pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(42).over("stock_code") - 1).alias("p_m1")).to_series()
