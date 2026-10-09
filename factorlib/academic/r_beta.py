"""r_beta — Quantactix academic factor formula.

Source: /media/felix/f/quant/akquant-factor-backtest/examples/v30_quantactix_factors.py
The formula is recomputed from panel columns; no factor values are copied.
"""
from __future__ import annotations

import polars as pl


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute r_beta from a stock_code/trade_date-sorted panel."""
    required = ["stock_code", "trade_date", "close", "amount", "circ_cap", "returns"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"r_beta: missing columns {missing}")
    market = (
        panel.group_by("trade_date").agg(pl.col("close").mean().alias("_market_close"))
        .sort("trade_date")
        .with_columns((pl.col("_market_close") / pl.col("_market_close").shift(1) - 1).alias("_market_return"))
    )
    staged = panel.join(market.select(["trade_date", "_market_return"]), on="trade_date", how="left")
    cov = pl.rolling_cov(pl.col("returns"), pl.col("_market_return"), window_size=60, min_samples=60).over("stock_code")
    var = pl.col("_market_return").rolling_var(window_size=60, min_samples=60).over("stock_code")
    return staged.select((cov / pl.when(var == 0).then(None).otherwise(var)).alias("r_beta")).to_series()
