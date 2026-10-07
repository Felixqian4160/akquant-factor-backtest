"""Empirical semantics test for AKQuant CorporateAction (Split/Dividend).

Goal: determine exact semantics of CorporateAction(symbol, date, type, value):
  - Split: shares multiplier = (1+value)? or value? 
  - Dividend: cash per share?
  - date: applied to positions held before that date (ex-date semantics)

Method: monkeypatch bte.Engine with a proxy factory; inject action; run mini
backtest on real data slice; inspect final equity + shares vs expectations.
"""
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import akquant as aq
import akquant.backtest.engine as bte

PANEL = pathlib.Path("data/wavehunter_hs300_v34_adj_20261007.parquet")
SYM = "000001.SZ"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── build real data slice: 15 trading days from 2020-01-02 ──
d = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "open", "high", "low", "close", "vol"])
s = (d.filter((pl.col("ts_code") == SYM) & (pl.col("trade_date") >= pl.datetime(2020, 1, 2))
              & (pl.col("trade_date") <= pl.datetime(2020, 2, 10)))
     .sort("trade_date"))
print(s.head(15))
pdf = s.to_pandas().set_index("trade_date").rename_axis("date")
pdf["symbol"] = SYM
pdf["volume"] = 1.0e9
pdf = pdf.drop(columns=["ts_code"])

DATES = [d.date() for d in pdf.index]
print("dates:", DATES[:12])
ACTION_DATE = DATES[5]  # 6th trading day


def run_variant(actions, label):
    # monkeypatch engine factory
    REAL_ENGINE = aq.Engine

    class EngineProxy:
        def __init__(self):
            object.__setattr__(self, "_e", REAL_ENGINE())

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_e"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_e"), name, value)

        def add_data(self, feed):
            self._e.add_data(feed)
            for a in actions:
                self._e.add_corporate_action(a)

    bte.Engine = lambda: EngineProxy()

    class BuyHold(aq.Strategy):
        def on_bar(self, bar):
            pass

        def on_cross_section(self, trading_date, timestamp):
            if str(trading_date)[:10] == str(DATES[0]):
                self.rebalance_weights(target_weights={SYM: 0.99}, liquidate_unmentioned=False)

    try:
        result = aq.run_backtest(
            data={SYM: pdf},
            strategy=BuyHold,
            initial_cash=1_000_000.0,
            commission_rate=0.0,
            slippage=None,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=1,
        )
        nav = [[str(ts)[:10], float(v)] for ts, v in result.equity_curve.items()]
        # find last snapshot positions
        snaps = getattr(result, "snapshots", None)
        pos_info = None
        if snaps:
            last_ts, last_pos = snaps[-1]
            for p in last_pos:
                try:
                    pos_info = {k: getattr(p, k) for k in dir(p) if not k.startswith("_") and not callable(getattr(p, k))}
                except Exception:
                    pass
        print(f"--- {label} ---")
        print("  last nav:", nav[-1] if nav else None)
        print("  snapshots n:", len(snaps) if snaps else 0)
        if pos_info:
            keep = {k: v for k, v in pos_info.items() if k in
                    ("symbol", "quantity", "available", "cost_price", "market_value", "avg_price")}
            print("  last position:", keep)
        # try portfolio access
        try:
            print("  result dir (subset):", [a for a in dir(result) if not a.startswith("_")][:40])
        except Exception:
            pass
        return result
    finally:
        bte.Engine = REAL_ENGINE


# reference: no action
r0 = run_variant([], "BASELINE (no action)")

# Split semantic disambiguation: value=0.5
# (a) shares x (1+0.5)=1.5 ; (b) shares x 0.5 ; (c) 10送5 => x1.5
ca_split = aq.CorporateAction(symbol=SYM, date=ACTION_DATE,
                              action_type=aq.CorporateActionType.Split, value=0.5)
r1 = run_variant([ca_split], "SPLIT value=0.5")

# Dividend semantic: value=5.0 => cash += shares x 5 if per-share
ca_div = aq.CorporateAction(symbol=SYM, date=ACTION_DATE,
                            action_type=aq.CorporateActionType.Dividend, value=5.0)
r2 = run_variant([ca_div], "DIVIDEND value=5.0")

print("\nprice at action date:", float(pdf.loc[s.name if hasattr(s, 'name') else pdf.index[5], "close"]) if len(pdf) > 5 else None)
print("close series head:", [float(x) for x in pdf["close"].head(12)])
