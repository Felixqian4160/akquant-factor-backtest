"""V32 voting 策略在多个熊市时段测试。

合同 (从 v32e_akquant_voting_v3.py 复制):
- 静态池 10 因子 + 固定方向
- 每因子提名 top-10 股票 (按方向 rank)
- 票数 >= MIN_VOTES=3, MIN_STOCKS=5, MAX_STOCKS=20
- 20 交易日调仓
- T+1 open (NextOpen), lot=100, commission 0.25% + slippage 0.10%
- target_total=0.99
- AKQuant 真实回测 + audit

跑多时段: 2011, 2015H2, 2016, 2018, 2022, 2024 (全熊市/震荡市)
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

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUTDIR = Path("evidence/v32_full_bear_history_20261006")
OUTDIR.mkdir(parents=True, exist_ok=True)

REBAL_STEP = 20
K_NOM = 10
MIN_VOTES, MIN_STOCKS, MAX_STOCKS = 3, 5, 20
INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.99

POOL = {
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}

# 测试时段 (start, end, label)
PERIODS = [
    ("2011-01-01", "2012-01-01", "2011_yin_die"),
    ("2015-06-01", "2016-06-01", "2015H2_gushai"),
    ("2016-01-01", "2017-01-01", "2016_ronge"),
    ("2018-01-01", "2019-01-01", "2018_quan_nian_xia_die"),
    ("2022-01-01", "2023-01-01", "2022_bear"),
    ("2024-04-01", "2024-12-31", "2024H2_shark"),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_picks(df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    df_win = df.filter(pl.col('trade_date').is_between(
        pl.lit(start.to_pydatetime()), pl.lit(end.to_pydatetime())))
    all_dates = sorted(df_win['trade_date'].unique().to_list())
    rebal_dates = all_dates[::REBAL_STEP]
    rebal_str = [str(pd.Timestamp(d))[:10] for d in rebal_dates]
    log(f"  {len(all_dates)} dates, {len(rebal_dates)} rebalances ({rebal_str[0]} → {rebal_str[-1]})")
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
    log(f"  picks: {len(picks)} dates, avg {np.mean([len(p) for p in picks.values()]):.1f}")
    return picks


def run_akquant(picks: dict, df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                out_dir: Path) -> dict:
    # data bounded both sides
    data_src = df.filter(
        (pl.col('trade_date') >= pl.lit((start - pd.Timedelta(days=30)).to_pydatetime())) &
        (pl.col('trade_date') <= pl.lit(end.to_pydatetime())))
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
    log(f"  data: {len(data)} symbols")

    _DAILY_PICKS = picks
    _REBAL_SET = set(picks.keys())

    class VoteStrategy(aq.Strategy):
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
            self.rebalance_weights(
                target_weights={s: TARGET_TOTAL / n for s in p},
                liquidate_unmentioned=True,
            )
            self._current = set(p)
            self.log(f"rebalance {d}: {n} stocks")

    log("  Running AKQuant...")
    t0 = time.time()
    try:
        result = aq.run_backtest(
            data=data,
            strategy=VoteStrategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COMMISSION_RATE,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=LOT_SIZE,
        )
    except Exception as e:
        log(f"  AKQuant failed: {e}")
        return {"error": str(e)}
    log(f"  AKQuant done in {time.time()-t0:.1f}s")

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

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'result.json', 'w') as f:
        json.dump({
            "period": f"{start.date()} ~ {end.date()}",
            "label": out_dir.name,
            "pool": POOL,
            "params": {
                "K_NOM": K_NOM, "MIN_VOTES": MIN_VOTES,
                "MIN_STOCKS": MIN_STOCKS, "MAX_STOCKS": MAX_STOCKS,
                "REBAL_STEP": REBAL_STEP,
            },
            "metrics": metrics_dict,
        }, f, indent=2, ensure_ascii=False)
    # trades
    try:
        orders_df = result.orders_df
        orders_df.to_csv(out_dir / 'orders.csv', index=False)
    except Exception:
        pass
    try:
        trades_df = result.trades_df
        trades_df.to_csv(out_dir / 'trades.csv', index=False)
    except Exception:
        pass
    try:
        nav_df = result.nav_series.to_frame('value').reset_index()
        nav_df.columns = ['date', 'value']
        nav_df.to_csv(out_dir / 'nav.csv', index=False)
    except Exception:
        pass
    return metrics_dict


def main():
    log("=" * 80)
    log("V32 voting 在多熊市时段验证")
    log("=" * 80)
    log(f"加载 panel + 10 因子...")
    df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
    log(f"panel loaded: {df.height:,} rows")

    summary = {}
    for start, end, label in PERIODS:
        log(f"\n=== 跑 {label} ({start} ~ {end}) ===")
        period_dir = OUTDIR / label
        period_dir.mkdir(parents=True, exist_ok=True)
        try:
            picks = build_picks(df, pd.Timestamp(start), pd.Timestamp(end))
            with open(period_dir / 'picks.json', 'w') as f:
                json.dump(picks, f, indent=1)
            m = run_akquant(picks, df, pd.Timestamp(start), pd.Timestamp(end), period_dir)
            if 'error' in m:
                summary[label] = m
                continue
            summary[label] = {
                "total_return_pct": m.get('total_return_pct', None),
                "sharpe_ratio": m.get('sharpe_ratio', None),
                "max_drawdown_pct": m.get('max_drawdown_pct', None),
                "win_rate": m.get('win_rate', None),
                "profit_factor": m.get('profit_factor', None),
                "closed_trade_count": m.get('closed_trade_count', None),
            }
            log(f"  -> +{m.get('total_return_pct', 0):.2f}% / sharpe {m.get('sharpe_ratio', 0):.3f} / MDD {m.get('max_drawdown_pct', 0):.2f}% / PF {m.get('profit_factor', 0):.3f}")
        except Exception as e:
            log(f"  period {label} failed: {e}")
            summary[label] = {"error": str(e)}

    log(f"\n=== 总结 ===")
    for label, m in summary.items():
        if 'error' in m:
            print(f"  {label}: ERROR {m['error']}")
        else:
            print(f"  {label}: +{m.get('total_return_pct', 0):.2f}% / sharpe {m.get('sharpe_ratio', 0):.3f} / MDD {m.get('max_drawdown_pct', 0):.2f}% / PF {m.get('profit_factor', 0):.3f}")

    with open(OUTDIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()