"""Per-stock execution helper for standalone TA-Lib formula modules."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl


def compute_talib(
    panel: pl.DataFrame,
    fn: Callable,
    *,
    name: str,
    output_index: int | None,
) -> pl.Series:
    """Apply one TA-Lib formula per stock and return an aligned Series.

    The caller must provide rows; this helper partitions by ``stock_code``
    internally so input order does not need to be pre-sorted.
    The helper accepts either ``volume`` or the legacy ``vol`` field.
    """
    required = {"stock_code", "trade_date", "open", "high", "low", "close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"missing panel columns for {name}: {missing}")
    volume_col = "volume" if "volume" in panel.columns else "vol" if "vol" in panel.columns else None
    if volume_col is None:
        raise ValueError(f"missing volume/vol panel column for {name}")

    pieces: list[np.ndarray] = []
    for group in panel.partition_by("stock_code", maintain_order=True):
        frame = group.select(["open", "high", "low", "close", volume_col]).to_pandas()
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        open_ = frame["open"].astype(float)
        volume = frame[volume_col].astype(float)
        out = fn(close, high, low, open_, volume)
        if output_index is not None:
            if not isinstance(out, tuple):
                raise TypeError(f"{name}: expected tuple output, got {type(out).__name__}")
            out = out[output_index]
        if isinstance(out, pd.Series):
            arr = out.to_numpy(dtype=float, na_value=np.nan)
        elif isinstance(out, pd.DataFrame):
            arr = out.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
        else:
            arr = np.asarray(out, dtype=float)
        if len(arr) != group.height:
            raise ValueError(f"{name}: output length {len(arr)} != group rows {group.height}")
        pieces.append(arr)
    if not pieces:
        return pl.Series(name, [], dtype=pl.Float64)
    return pl.Series(name, np.concatenate(pieces))
