"""V32c: AKQuant 真实回测 — IC/收益排名 + 投票机制策略 (静态池版)

策略 (来自 v32/v32b 验证):
  - 因子池: H1 (2022-01~2023-06) 收益 top10 + 方向固定:
      l_ami(hi), gtja_144(hi), l_size3(lo), l_size(lo), gtja_132(lo),
      l_dtvm(lo), r_tv(hi), gtja_070(lo), gtja_095(lo), talib_NATR(hi)
  - 每 20 交易日调仓: 每个因子提名自己 top-10 股票 (按方向), 票数 >=3 入选
    (>=5 保底, <=20 封顶), 等权
  - 执行: T+1 open (fill_policy=NextOpen), lot=100, commission 0.0025 + slippage 0.001
  - 窗口: 2022-01-04 ~ 2024-12-31
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
OUTDIR = Path("evidence/v32_akquant_voting_20260925")
OUTDIR.mkdir(parents=True, exist_ok=True)

W_S, W_E = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
REBAL_STEP = 20
K_NOM = 10
MIN_VOTES, MIN_STOCKS, MAX_STOCKS = 3, 5, 20
INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100

POOL = {  # static pool (H1 return top10) + fixed directions
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log("=" * 80)
log("V32c: AKQuant 真实回测 - 投票策略 (静态池)")
log("=" * 80)

# --- 1. load panel, rebal dates, picks ---
log("Loading panel...")
df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
df = df.filter(pl.col('trade_date').is_between(
    pl.lit(W_S.to_pydatetime()), pl.lit(W_E.to_pydatetime())))
all_dates = sorted(df['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
rebal_str = [str(pd.Timestamp(d))[:10] for d in rebal_dates]
log(f"window: {len(all_dates)} dates, {len(rebal_dates)} rebalances")

# --- 2. compute votes/picks per rebalance date ---
log("Computing picks (nominations + voting)...")
picks = {}
vote_detail = {}
for rd, rds in zip(rebal_dates, rebal_str):
    day = df.filter(pl.col('trade_date') == rd)
    votes = Counter()
    for fac, direction in POOL.items():
        sub = day.select(['ts_code', fac]).drop_nulls()
        if len(sub) < K_NOM:
            continue
        if direction == 'hi':
            sub = sub.with_columns(pl.col(fac).rank(descending=True).alias('rk'))
        else:
            sub = sub.with_columns(pl.col(fac).rank().alias('rk'))
        nom = sub.filter(pl.col('rk') <= K_NOM)['ts_code'].to_list()
        for c in nom:
            votes[c] += 1
    if not votes:
        continue
    sel = [c for c, v in votes.items() if v >= MIN_VOTES]
    if len(sel) < MIN_STOCKS:
        sel = [c for c, _ in votes.most_common(MIN_STOCKS)]
    if len(sel) > MAX_STOCKS:
        sel = [c for c, _ in votes.most_common(MAX_STOCKS)]
    picks[rds] = {c: float(votes[c]) for c in sel}
    vote_detail[rds] = {'n_votes_total': len(votes), 'n_selected': len(sel),
                        'max_votes': max(votes.values()), 'sel_votes': sorted([votes[c] for c in sel], reverse=True)[:5]}

log(f"picks computed: {len(picks)} / {len(rebal_str)} rebalance dates")
avg_picks = np.mean([len(p) for p in picks.values()])
log(f"avg picks: {avg_picks:.1f}")
with open(OUTDIR / 'picks.json', 'w') as f:
    json.dump(picks, f, indent=1)
with open(OUTDIR / 'vote_detail.json', 'w') as f:
    json.dump(vote_detail, f, indent=1)

# --- 3. build AKQuant data dict (all panel symbols in window + buffer) ---
log("Building AKQuant data dict...")
data_src = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'])
data_src = data_src.filter(pl.col('trade_date') >= pl.lit(pd.Timestamp("2021-12-01")))
symbols = sorted(data_src['ts_code'].unique().to_list())
log(f"symbols: {len(symbols)}")

universe = sorted({s for p in picks.values() for s in p})
log(f"picks universe: {len(universe)} symbols ever picked")

data = {}
for sym in universe:
    pdf = (data_src.filter(pl.col('ts_code') == sym)
           .sort('trade_date').to_pandas()
           .set_index('trade_date').rename_axis('date'))
    if pdf.empty:
        continue
    pdf['symbol'] = sym
    pdf = pdf.drop(columns=['ts_code'])
    pdf['volume'] = 1.0e9   # AKQuant tradability 要求
    data[sym] = pdf
log(f"data dict: {len(data)} symbols")

# --- 4. strategy ---
_DAILY_PICKS = picks
_REBAL_SET = set(picks.keys())


class V32VoteStrategy(aq.Strategy):
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
        new_set = set(p)
        if new_set == self._current:
            return
        self.rebalance_to_topn(
            scores=p, top_n=len(new_set), weight_mode='equal',
            long_only=True, liquidate_unmentioned=True,
        )
        self._current = new_set
        self.log(f"rebalance {d}: {len(new_set)} stocks")


# --- 5. run ---
log("Running AKQuant backtest...")
t0 = time.time()
result = aq.run_backtest(
    data=data,
    strategy=V32VoteStrategy,
    initial_cash=INITIAL_CASH,
    commission_rate=COMMISSION_RATE,
    slippage=SLIPPAGE,
    t_plus_one=False,
    fill_policy=aq.NextOpen(),
    lot_size=LOT_SIZE,
)
log(f"backtest done in {time.time()-t0:.1f}s")

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

log("\n=== metrics ===")
for k in ['total_return_pct', 'annualized_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
          'win_rate', 'profit_factor', 'closed_trade_count', 'end_market_value', 'total_commission']:
    if k in metrics_dict:
        log(f"  {k}: {metrics_dict[k]}")

# --- 6. extract trades/orders/nav ---
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
nav = [[str(ts)[:10], float(v)] for ts, v in result.equity_curve.items()] if hasattr(result.equity_curve, 'items') else []
if not nav and hasattr(result, 'equity_curve'):
    try:
        nav = [[str(ts)[:10], float(v)] for ts, v in result.equity_curve.items()]
    except Exception:
        pass

log(f"trades: {len(trades)}, orders: {len(orders)}, nav points: {len(nav)}")
if trades:
    log(f"trade fields: {sorted(trades[0].keys())}")
    log(f"first trade: {trades[0]}")
pd.DataFrame(trades).to_csv(OUTDIR / 'trades.csv', index=False)
pd.DataFrame(orders).to_csv(OUTDIR / 'orders.csv', index=False)
pd.DataFrame(nav, columns=['date', 'value']).to_csv(OUTDIR / 'nav.csv', index=False)

# --- 7. ledger audit ---
log("\n=== ledger audit ===")
audit = {'checks': [], 'summary': {}}

def check(name, ok, detail):
    audit['checks'].append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

check('trades_exist', len(trades) > 0, f"{len(trades)} trades")
if trades:
    tdf = pd.DataFrame(trades)
    qty_col = next((c for c in ('qty', 'quantity', 'volume', 'filled_qty') if c in tdf.columns), None)
    side_col = next((c for c in ('side', 'direction', 'action') if c in tdf.columns), None)
    price_col = next((c for c in ('price', 'fill_price', 'avg_price', 'exec_price') if c in tdf.columns), None)
    comm_col = next((c for c in ('commission', 'fee', 'fees') if c in tdf.columns), None)
    log(f"  cols: qty={qty_col}, side={side_col}, price={price_col}, commission={comm_col}")
    if qty_col:
        q = pd.to_numeric(tdf[qty_col], errors='coerce').dropna()
        check('qty_lot_multiple', bool((q % 100 == 0).all()), f"min={q.min()}, all %100==0 = {(q % 100 == 0).all()}")
    if side_col:
        check('both_sides', tdf[side_col].nunique() >= 2, f"sides={tdf[side_col].value_counts().to_dict()}")
    if comm_col:
        total_comm = pd.to_numeric(tdf[comm_col], errors='coerce').sum()
        check('commission_positive', total_comm > 0, f"sum={total_comm:,.0f}")
        audit['summary']['total_commission_from_trades'] = float(total_comm)

audit['summary']['final_value'] = metrics_dict.get('end_market_value')
audit['summary']['total_return_pct'] = metrics_dict.get('total_return_pct')
audit['summary']['n_trades'] = len(trades)
audit['summary']['n_rebalances'] = len(picks)
audit['summary']['avg_picks'] = float(avg_picks)
with open(OUTDIR / 'ledger_audit.json', 'w') as f:
    json.dump(audit, f, indent=2, default=str)

# --- 8. save result summary ---
summary = {
    'strategy': 'V32VoteStrategy (static pool, H1-selected)',
    'window': f'{rebal_str[0]} ~ {rebal_str[-1]}',
    'pool': POOL,
    'params': {'K_NOM': K_NOM, 'MIN_VOTES': MIN_VOTES, 'MIN_STOCKS': MIN_STOCKS,
               'MAX_STOCKS': MAX_STOCKS, 'REBAL_STEP': REBAL_STEP},
    'costs': {'commission_rate': COMMISSION_RATE, 'slippage': SLIPPAGE, 'lot_size': LOT_SIZE},
    'metrics': metrics_dict,
    'n_trades': len(trades),
    'n_rebalances': len(picks),
    'avg_picks': float(avg_picks),
}
with open(OUTDIR / 'result.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

log(f"\nSaved: {OUTDIR}/result.json, trades.csv, orders.csv, nav.csv, ledger_audit.json")
log("DONE")
