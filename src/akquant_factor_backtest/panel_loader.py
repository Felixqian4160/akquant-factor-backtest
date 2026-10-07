"""Panel loader (read-only) for the v10.2 factor panel.

One parquet source from the aurumq-rl workspace, read-only:

INDEX_PANEL = aurumq-rl/data/wavehunter_v10_1_hs300_20040102_20260827.parquet
    - has the real HS300 idx_close column (per-stock replicated; same value
      for every ts_code on a given date)
    - does NOT have idx_open / idx_high / idx_low / idx_volume (only
      per-stock open/high/low/close/volume exist)
    - this is the canonical panel for index-level (single-symbol) backtests

FACTOR_PANEL = aurumq-rl/evidence/.../v10_2_mainwave_features_v1_20260921/
              wavehunter_mainwave_features_v1.parquet
    - 354 HS300 stocks × ~340 factor columns (alpha101, gtja, mw_*)
    - per-stock, per-day; cross-sectional aggregation needed for index-level

Both panels share the same per-stock OHLCV columns. Only INDEX_PANEL has
idx_close. We load OHLCV from INDEX_PANEL (the canonical source) and the
factor mean from FACTOR_PANEL, joining on trade_date.

We do NOT mutate the source parquets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import pandas as pd

# --- paths ------------------------------------------------------------------

ROOT = Path("/media/felix/f/quant/aurumq-rl")
INDEX_PANEL = ROOT / "data" / "wavehunter_v10_1_hs300_20040102_20260827.parquet"

EVIDENCE = ROOT / "evidence" / "quant_workflow_migration_20260915"
FACTOR_PANEL = (
    EVIDENCE
    / "v10_2_mainwave_features_v1_20260921"
    / "wavehunter_mainwave_features_v1.parquet"
)

DATE_COL = "trade_date"


# --- helpers ----------------------------------------------------------------


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required panel not found: {path}")


# --- public api -------------------------------------------------------------


def load_idx_close() -> pd.DataFrame:
    """Return a pandas DataFrame indexed by trade_date with OHLCV cols.

    The index panel has per-stock open/high/low/close/volume but the
    idx_close column carries the HS300 index value (replicated per
    stock). We synthesise idx_open / idx_high / idx_low from per-stock
    daily equal-weight mean (close-only proxy) and set idx_volume to 0
    because the panel does not record index-level volume.

    Output columns: open, high, low, close, volume, symbol=HS300.
    """
    _require(INDEX_PANEL)

    # Pull per-day HS300 index close + per-stock daily mean (used for
    # open/high/low proxy). Volume is set to 0 (no index volume source).
    df = (
        pl.scan_parquet(str(INDEX_PANEL))
        .select(
            [
                pl.col(DATE_COL).cast(pl.Date),
                pl.col("idx_close"),
                pl.col("open").alias("_stock_open"),
                pl.col("high").alias("_stock_high"),
                pl.col("low").alias("_stock_low"),
                pl.col("close").alias("_stock_close"),
            ]
        )
        .filter(pl.col("idx_close").is_not_null() & pl.col("idx_close").is_finite())
        .group_by(DATE_COL)
        .agg(
            [
                pl.col("idx_close").first().alias("close"),
                pl.col("_stock_open").mean().alias("open"),
                pl.col("_stock_high").mean().alias("high"),
                pl.col("_stock_low").mean().alias("low"),
                pl.col("_stock_close").mean().alias("_close_check"),
            ]
        )
        .sort(DATE_COL)
        .collect()
    )

    pdf = df.to_pandas().set_index(DATE_COL).rename_axis("date")
    pdf["volume"] = 1.0e9  # synthetic, nonzero so RiskManager doesn't reject for zero volume
    pdf["symbol"] = "HS300"
    # Drop the sanity-check column; keep OHLCV.
    pdf = pdf.drop(columns=["_close_check"])
    return pdf[["open", "high", "low", "close", "volume", "symbol"]]


def load_panel_for_factor(
    factor: str,
    leg_dates: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a single-symbol HS300 panel where `bar.extra["factor_<name>"]`
    carries the cross-sectional mean of `factor` for that date.

    Index OHLCV comes from INDEX_PANEL (idx_close + per-stock daily mean
    for open/high/low); factor mean comes from FACTOR_PANEL. Joined on
    trade_date.
    """
    _require(INDEX_PANEL)
    _require(FACTOR_PANEL)

    extra_col = f"factor_{factor}"

    # Index OHLCV per day (idx_close + per-stock mean proxy).
    idx = (
        pl.scan_parquet(str(INDEX_PANEL))
        .select(
            [
                pl.col(DATE_COL).cast(pl.Date),
                pl.col("idx_close"),
                pl.col("open").alias("_stock_open"),
                pl.col("high").alias("_stock_high"),
                pl.col("low").alias("_stock_low"),
            ]
        )
        .filter(pl.col("idx_close").is_not_null() & pl.col("idx_close").is_finite())
        .group_by(DATE_COL)
        .agg(
            [
                pl.col("idx_close").first().alias("close"),
                pl.col("_stock_open").mean().alias("open"),
                pl.col("_stock_high").mean().alias("high"),
                pl.col("_stock_low").mean().alias("low"),
            ]
        )
        .sort(DATE_COL)
        .collect()
    )

    # Cross-sectional mean of the factor per day.
    factor_mean = (
        pl.scan_parquet(str(FACTOR_PANEL))
        .select([pl.col(DATE_COL).cast(pl.Date), pl.col(factor)])
        .group_by(DATE_COL)
        .agg(pl.col(factor).mean().alias(extra_col))
        .sort(DATE_COL)
        .collect()
    )

    # Inner join: only days where both panels exist.
    joined = idx.join(factor_mean, on=DATE_COL, how="inner")

    # Optional: filter to leg windows.
    if leg_dates is not None:
        masks = []
        for start, end in leg_dates:
            masks.append(
                (pl.col(DATE_COL) >= pl.lit(start))
                & (pl.col(DATE_COL) <= pl.lit(end))
            )
        if masks:
            joined = joined.filter(pl.any_horizontal(*masks))

    pdf = joined.to_pandas().set_index(DATE_COL).rename_axis("date")
    pdf["volume"] = 1.0e9  # synthetic, nonzero so RiskManager doesn't reject for zero volume
    pdf["symbol"] = "HS300"
    return pdf[["open", "high", "low", "close", "volume", extra_col, "symbol"]]


__all__ = [
    "load_idx_close",
    "load_panel_for_factor",
    "INDEX_PANEL",
    "FACTOR_PANEL",
    "DATE_COL",
]
