"""Check akquant MAX2/MIN2 semantics and v34 talib_MAX2/MIN2 column status."""
import inspect

import numpy as np
import akquant.talib as tq

print("MAX2 doc:", (inspect.getdoc(tq.MAX2) or "")[:200])
print("MAX2 source head:")
print(inspect.getsource(tq.MAX2)[:400])
print("---")
r = tq.MAX2(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0]), as_series=True)
print("MAX2 example:", list(r))

import polars as pl

V34 = "/media/felix/f/quant/akquant-factor-backtest/data/wavehunter_hs300_v34_adj_20261007.parquet"
df = pl.read_parquet(V34, columns=["talib_MAX2", "talib_MIN2"])
for col in ("talib_MAX2", "talib_MIN2"):
    s = df[col]
    print(col, "nulls:", s.null_count(), "/", s.len(), "mean:", s.mean())
