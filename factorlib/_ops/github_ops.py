"""Panel preparation for standalone GitHub/academic formulas."""
from __future__ import annotations

import polars as pl


def prepare(panel: pl.DataFrame) -> pl.DataFrame:
    d = panel.sort(["stock_code", "trade_date"])
    if "volume" not in d.columns and "vol" in d.columns:
        d = d.with_columns(pl.col("vol").alias("volume"))
    if "vol" not in d.columns and "volume" in d.columns:
        d = d.with_columns(pl.col("volume").alias("vol"))
    # Keep missing fundamental inputs explicit: the formula will produce nulls,
    # rather than silently substituting an unrelated field.
    for col in ("bps", "grossprofit_margin", "netprofit_yoy"):
        if col not in d.columns:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    if "returns" not in d.columns:
        d = d.with_columns((pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1).alias("returns"))
    market = (
        d.group_by("trade_date").agg(pl.col("close").mean().alias("_mc"))
        .sort("trade_date")
        .with_columns([
            (pl.col("_mc") / pl.col("_mc").shift(1) - 1).alias("_mkt_ret"),
            (pl.col("_mc").shift(21) / pl.col("_mc").shift(126) - 1).alias("_mkt_ret6"),
        ])
    )
    return d.join(market.select(["trade_date", "_mkt_ret", "_mkt_ret6"]), on="trade_date", how="left")
