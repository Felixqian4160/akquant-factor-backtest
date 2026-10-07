"""Diagnostic: isolate AKQuant Split action mechanics on a REAL case (000630.SZ 2015-10-23, s=5).

Tests:
  T1: buy day1, hold across event -> NAV path before/after (no action vs action)
  T2: buy AFTER event -> qty must be unaffected by the past action
"""
import datetime as dt
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl
import akquant as aq
import akquant.backtest.engine as bte

PANEL_RAW = pathlib.Path("data/wavehunter_hs300_v34_adj_20261007.parquet")
SYM = "601633.SH"
EVENT = dt.date(2015, 10, 13)

d = pl.read_parquet(PANEL_RAW, columns=["trade_date", "ts_code", "open", "high", "low", "close", "vol"])
s = (d.filter((pl.col("ts_code") == SYM) & (pl.col("trade_date") >= pl.datetime(2015, 9, 20))
              & (pl.col("trade_date") <= pl.datetime(2015, 10, 28)))
     .sort("trade_date"))
print(s)
pdf = s.to_pandas().set_index("trade_date").rename_axis("date")
pdf["symbol"] = SYM
pdf["volume"] = 1.0e9
pdf = pdf.drop(columns=["ts_code"])
DATES = [x.date() for x in pdf.index]
print("dates:", DATES)


def make_runner(actions, buy_date_str):
    class BuyHold(aq.Strategy):
        def on_bar(self, bar):
            pass
        def on_cross_section(self, trading_date, timestamp):
            if str(trading_date)[:10] == buy_date_str:
                self.rebalance_weights(target_weights={SYM: 0.50}, liquidate_unmentioned=False)

    real = bte.Engine

    class Proxy:
        def __init__(self):
            object.__setattr__(self, "_e", real())
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_e"), name)
        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_e"), name, value)
        def add_data(self, feed):
            self._e.add_data(feed)
            for a in actions:
                self._e.add_corporate_action(a)

    bte.Engine = lambda: Proxy()
    try:
        result = aq.run_backtest(data={SYM: pdf}, strategy=BuyHold, initial_cash=1_000_000.0,
                                 commission_rate=0.0, slippage=None, t_plus_one=False,
                                 fill_policy=aq.NextOpen(), lot_size=1)
    finally:
        bte.Engine = real
    nav = [[str(ts)[:10], float(v)] for ts, v in result.equity_curve.items()]
    snaps = result.snapshots
    qtys = {}
    for ts, positions in snaps:
        ds = str(pd.Timestamp(ts).date())
        for p in positions:
            try:
                if getattr(p, "symbol", None) == SYM:
                    qtys[ds] = float(getattr(p, "quantity", 0))
            except Exception:
                pass
    return nav, qtys


sp = aq.CorporateAction(symbol=SYM, date=EVENT, action_type=aq.CorporateActionType.Split, value=5.0)
dv = aq.CorporateAction(symbol=SYM, date=EVENT, action_type=aq.CorporateActionType.Dividend, value=0.18)

BUY1 = str(DATES[0])
BUY2 = str(DATES[16])  # after the event (event is around index 10)

print(f"\n=== T1: buy {BUY1}, hold across event ===")
nav0, q0 = make_runner([], BUY1)
nav1, q1 = make_runner([sp, dv], BUY1)
print("NO-ACTION  nav:", [(x[0][5:], round(x[1])) for x in nav0])
print("WITH-ACTION nav:", [(x[0][5:], round(x[1])) for x in nav1])
print("NO-ACTION qty:", q0)
print("WITH-ACTION qty:", q1)

print(f"\n=== T2: buy {BUY2} (AFTER event) ===")
nav2, q2 = make_runner([sp, dv], BUY2)
print("WITH-ACTION(2) qty:", q2)
print("WITH-ACTION(2) nav:", [(x[0][5:], round(x[1])) for x in nav2])
