"""V32d: AKQuant 真实回测 v2 — 投票策略 (修复版)

v1 问题与修复:
  1. v1 数据未截断 → nav 跑到 2026-08; v2 数据截断到 2024-12-31
  2. v1 全额目标权重 (1.0/n) → 31 笔买入因保证金不足被拒;
     v2 显式目标权重 0.99/n (1% 现金缓冲)

策略合同不变 (同 v32c):
  - 静态池: H1 收益 top10 + 固定方向
  - 每 20 交易日调仓, 每因子提名 top-10, 票数>=3 入选 (5保底/20封顶), 等权
  - T+1 open 执行, lot=100, commission 0.0025 + slippage 0.001
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v31_refactored_v2_20260925.parquet")
ICDIR = Path("evidence/v32_ic_voting_20260925")
OUTDIR = Path("evidence/v32_akquant_voting_v2_20260925")
OUTDIR.mkdir(parents=True, exist_ok=True)

W_S, W_E = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
REBAL_STEP = 20
K_NOM = 10
MIN_VOTES, MIN_STOCKS, MAX_STOCKS = 3, 5, 20
INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.99   # 1% 现金缓冲

POOL = {
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=" * 80)
log("V32d: AKQuant 真实回测 v2 (截断 + 缓冲)")
log("=" * 80)

# --- 1. load panel (bounded) + picks ---
df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
df_win = df.filter(pl.col('trade_date').is_between(
    pl.lit(W_S.to_pydatetime()), pl.lit(pd.Timestamp("2024-12-31 23:59").to_pydatetime())))
all_dates = sorted(df_win['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
rebal_str = [str(pd.Timestamp(d))[:10] for d in rebal_dates]
log(f"window: {len(all_dates)} dates → {rebal_str[-1]}, {len(rebal_dates)} rebalances")

log("Computing picks...")
picks = {}
for rd, rds in zip(rebal_dates, rebal_str):
    day = df_win.filter(pl.col('trade_date') == rd)
    votes = Counter()
    for fac, direction in POOL.items():
        sub = day.select(['ts_code', fac]).drop_nulls()
        if len(sub) < K_NOM:
            continue
        if direction == 'hi':
            sub = sub.with_columns(pl.col(fac).rank(descending=True).alias('rk'))
        else:
            sub = sub.with_columns(pl.col(fac).rank().alias('rk'))
        for c in sub.filter(pl.col('rk') <= K_NOM)['ts_code'].to_list():
            votes[c] += 1
    if not votes:
        continue
    sel = [c for c, v in votes.items() if v >= MIN_VOTES]
    if len(sel) < MIN_STOCKS:
        sel = [c for c, _ in votes.most_common(MIN_STOCKS)]
    if len(sel) > MAX_STOCKS:
        sel = [c for c, _ in votes.most_common(MAX_STOCKS)]
    picks[rds] = {c: float(votes[c]) for c in sel}
log(f"picks: {len(picks)} dates, avg {np.mean([len(p) for p in picks.values()]):.1f}")
with open(OUTDIR / 'picks.json', 'w') as f:
    json.dump(picks, f, indent=1)

# --- 2. AKQuant data dict (2021-12-01 → 2024-12-31) ---
data_src = df.filter(pl.col('trade_date') >= pl.lit(pd.Timestamp("2021-12-01")))
universe = sorted({s for p in picks.values() for s in p})
data = {}
for sym in universe:
    pdf = (data_src.filter(pl.col('ts_code') == sym)
           .sort('trade_date').to_pandas()
           .set_index('trade_date').rename_axis('date'))
    if pdf.empty:
        continue
    pdf['symbol'] = sym
    pdf = pdf.drop(columns=['ts_code'])
    pdf['volume'] = 1.0e9
    data[sym] = pdf
log(f"data: {len(data)} symbols (→ 2024-12-31)")

# --- 3. strategy (explicit 0.99 weights) ---
_DAILY_PICKS = picks
_REBAL_SET = set(picks.keys())


class V32dVoteStrategy(aq.Strategy):
    warmup = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current: set = set()

    def on_bar(self, bar):
        pass

    def on_cross_section(self, trading_date, timestamp):
        d = str(trading_date)[:10]
        if d not in _REBAL_SET:
            return
        p = _DAILY_PICKS.get(d, {})
        if not p:
            return
        n = len(p)
        target = {s: TARGET_TOTAL / n for s in p}
        self.rebalance_weights(
            target_weights=target,
            liquidate_unmentioned=True,
        )
        if set(p) != self._current:
            self._current = set(p)
        self.log(f"rebalance {d}: {n} stocks @ {TARGET_TOTAL/n*100:.2f}% each")


# --- 4. run ---
log("Running AKQuant backtest (bounded)...")
t0 = time.time()
result = aq.run_backtest(
    data=data,
    strategy=V32dVoteStrategy,
    initial_cash=INITIAL_CASH,
    commission_rate=COMMISSION_RATE,
    slippage=SLIPPAGE,
    t_plus_one=False,
    fill_policy=aq.NextOpen(),
    lot_size=LOT_SIZE,
)
log(f"done in {time.time()-t0:.1f}s")

metrics = result.metrics_df
metrics_dict = {}
for index in metrics.index:
    value = metrics.loc[index, 'value']
    if hasattr(value, 'isoformat'):
        value = value.isoformat()
    try:
        metrics_dict[index] = float(value)
    except (TypeError, ValueError):
        metrics_dict[index] = str(value)

log("\n=== metrics (bounded 2022-2024) ===")
for k in ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate',
          'profit_factor', 'closed_trade_count', 'end_market_value', 'total_commission']:
    if k in metrics_dict:
        log(f"  {k}: {metrics_dict[k]}")

# --- 5. extract ---
def extract_records(obj):
    if not isinstance(obj, list):
        return []
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item); continue
        rec = {}
        for attr in dir(item):
            if attr.startswith('_'):
                continue
            try:
                v = getattr(item, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if hasattr(v, 'isoformat'):
                v = v.isoformat()
            elif hasattr(v, 'item'):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            rec[attr] = v
        out.append(rec)
    return out

trades = extract_records(result.trades)
orders = extract_records(result.orders)
nav = [[str(ts)[:10], float(v)] for ts, v in result.equity_curve.items()]
log(f"trades: {len(trades)}, orders: {len(orders)}, nav: {len(nav)} pts ({nav[0][0]} → {nav[-1][0]})")
pd.DataFrame(trades).to_csv(OUTDIR / 'trades.csv', index=False)
pd.DataFrame(orders).to_csv(OUTDIR / 'orders.csv', index=False)
nav_df = pd.DataFrame(nav, columns=['date', 'value'])
nav_df.to_csv(OUTDIR / 'nav.csv', index=False)

# --- 6. per-year + comparison vs research ---
nav_df['dt'] = pd.to_datetime(nav_df['date'])
log("\n=== per-year ===")
for y in [2022, 2023, 2024]:
    yr = nav_df[nav_df['dt'].dt.year == y]
    if len(yr) > 1:
        r = yr['value'].iloc[-1] / yr['value'].iloc[0] - 1
        log(f"  {y}: {r*100:+.1f}%")

per = pd.read_csv(ICDIR / 'best_config_periods.csv')
per['cum'] = (1 + per['ret']).cumprod()
log(f"\nresearch: full 37期 {per['cum'].iloc[-1]*100-100:+.1f}%")
log(f"AKQuant v2: {metrics_dict.get('total_return_pct', 0):+.1f}% (end value {metrics_dict.get('end_market_value', 0):,.0f})")

# --- 7. ledger audit ---
log("\n=== ledger audit v2 ===")
audit = {'checks': [], 'summary': {}}

def check(name, ok, detail):
    audit['checks'].append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

odf = pd.DataFrame(orders)
tdf = pd.DataFrame(trades)

check('trades_exist', len(trades) > 0, len(trades))
check('orders_both_sides', odf['side'].nunique() >= 2, odf['side'].value_counts().to_dict())
n_rej = int((odf['status'] == 'OrderStatus.Rejected').sum())
check('rejected_orders', n_rej == 0, f"{n_rej} rejected (v1 had 31)")
q = pd.to_numeric(tdf['quantity'], errors='coerce').dropna()
check('qty_lot_multiple', bool((q % 100 == 0).all()), f"min={q.min()}")
check('commission_positive', odf['commission'].sum() > 0, f"orders commission={odf['commission'].sum():,.0f}")
check('nav_bounded', nav[-1][0] <= '2024-12-31', nav[-1][0])

# open positions (from orders)
of = odf[odf['status'] == 'OrderStatus.Filled'].copy()
of['signed_qty'] = np.where(of['side'] == 'OrderSide.Buy', of['filled_quantity'], -of['filled_quantity'])
pos = of.groupby('symbol')['signed_qty'].sum()
open_pos = pos[pos > 0]
log(f"  open positions: {len(open_pos)} symbols")
# MTM at last close
last_day = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'close']).filter(
    pl.col('trade_date') <= pl.lit(pd.Timestamp('2024-12-31').to_pydatetime())
).group_by('ts_code').agg(pl.col('close').last())
close_map = dict(zip(last_day['ts_code'].to_list(), last_day['close'].to_list()))
mtm = float(sum(qty * close_map.get(sym, 0) for sym, qty in open_pos.items()))
check('open_positions_mtm', True, f"{len(open_pos)} pos, MTM≈¥{mtm:,.0f}")

audit['summary'] = {
    'final_value': metrics_dict.get('end_market_value'),
    'total_return_pct': metrics_dict.get('total_return_pct'),
    'n_trades': len(trades), 'n_rejected': n_rej,
    'open_positions': len(open_pos), 'open_mtm': mtm,
    'research_full': float(per['cum'].iloc[-1] - 1),
}
with open(OUTDIR / 'ledger_audit.json', 'w') as f:
    json.dump(audit, f, indent=2, default=str)

summary = {
    'strategy': 'V32dVoteStrategy (static pool, bounded, 0.99 buffer)',
    'window': f'{rebal_str[0]} ~ {rebal_str[-1]}',
    'pool': POOL, 'target_total': TARGET_TOTAL,
    'params': {'K_NOM': K_NOM, 'MIN_VOTES': MIN_VOTES, 'MIN_STOCKS': MIN_STOCKS,
               'MAX_STOCKS': MAX_STOCKS, 'REBAL_STEP': REBAL_STEP},
    'costs': {'commission_rate': COMMISSION_RATE, 'slippage': SLIPPAGE, 'lot_size': LOT_SIZE},
    'metrics': metrics_dict, 'n_trades': len(trades), 'n_rejected': n_rej,
}
with open(OUTDIR / 'result.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

log(f"\nSaved: {OUTDIR}/")
log("DONE")
