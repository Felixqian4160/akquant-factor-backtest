"""V32 voting 单变量参数 sweep 在 4 个亏损熊市时段。

合同 baseline (V32 voting_v3):
- POOL 10 因子静态
- MIN_VOTES=3, MIN_STOCKS=5, MAX_STOCKS=20, REBAL_STEP=20
- target_total=0.99, lot=100, commission=0.25%, slippage=0.10%
- T+1 open (NextOpen)

Sweep:
- MIN_VOTES ∈ {2, 3, 4, 5, 6}
- MAX_STOCKS ∈ {10, 15, 20, 25, 30, 40}
- REBAL_STEP ∈ {10, 15, 20, 30, 40}
- target_total ∈ {0.5, 0.7, 0.85, 0.99}

只改一个参数, 其他保持 baseline. 每个时段 × 每个 sweep 共 4+4+5+4 = 17 runs.
总 4 时段 × 17 = 68 个 backtests.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUTDIR = Path("evidence/v32_sweep_20261006")
OUTDIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
K_NOM = 10

POOL = {
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}

# 仅亏损的 4 个时段
LOSS_PERIODS = [
    ("2011-01-01", "2012-01-01", "2011_yin_die"),
    ("2015-06-01", "2016-06-01", "2015H2_gushai"),
    ("2018-01-01", "2019-01-01", "2018_quan_nian_xia_die"),
    ("2022-01-01", "2023-01-01", "2022_bear"),
]

# Baseline
BASELINE = {
    "MIN_VOTES": 3, "MIN_STOCKS": 5, "MAX_STOCKS": 20,
    "REBAL_STEP": 20, "TARGET_TOTAL": 0.99,
}

SWEEPS = {
    "MIN_VOTES": [2, 3, 4, 5, 6],
    "MAX_STOCKS": [10, 15, 20, 25, 30, 40],
    "REBAL_STEP": [10, 15, 20, 30, 40],
    "TARGET_TOTAL": [0.5, 0.7, 0.85, 0.99],
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_period(df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
               params: dict, label: str) -> dict:
    MIN_VOTES = params["MIN_VOTES"]
    MIN_STOCKS = params["MIN_STOCKS"]
    MAX_STOCKS = params["MAX_STOCKS"]
    REBAL_STEP = params["REBAL_STEP"]
    TARGET_TOTAL = params["TARGET_TOTAL"]

    df_win = df.filter(pl.col('trade_date').is_between(
        pl.lit(start.to_pydatetime()), pl.lit(end.to_pydatetime())))
    all_dates = sorted(df_win['trade_date'].unique().to_list())
    rebal_dates = all_dates[::REBAL_STEP]
    rebal_str = [str(pd.Timestamp(d))[:10] for d in rebal_dates]
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

    data_src = df.filter(
        (pl.col('trade_date') >= pl.lit((start - pd.Timedelta(days=30)).to_pydatetime())) &
        (pl.col('trade_date') <= pl.lit(end.to_pydatetime())))
    universe = sorted({s for p in picks.values() for s in p})
    data = {}
    for sym in universe:
        pdf = (data_src.filter(pl.col('ts_code') == sym)
               .sort('trade_date').to_pandas()
               .set_index('trade_date').rename_axis('date'))
        if pdf.empty: continue
        pdf['symbol'] = sym
        pdf = pdf.drop(columns=['ts_code'])
        pdf['volume'] = 1.0e9
        data[sym] = pdf

    _DAILY_PICKS = picks
    _REBAL_SET = set(picks.keys())

    class VoteStrategy(aq.Strategy):
        warmup = 5
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._current: set = set()
        def on_bar(self, bar): pass
        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET: return
            p = _DAILY_PICKS.get(d, {})
            if not p: return
            n = len(p)
            self.rebalance_weights(
                target_weights={s: TARGET_TOTAL / n for s in p},
                liquidate_unmentioned=True,
            )
            self._current = set(p)

    try:
        result = aq.run_backtest(
            data=data, strategy=VoteStrategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COMMISSION_RATE,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=LOT_SIZE,
        )
    except Exception as e:
        return {"error": str(e)}

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
    return metrics_dict


def main():
    log("=" * 80)
    log("V32 voting 单变量参数 sweep 在 4 个亏损熊市时段")
    log("=" * 80)
    log(f"加载 panel + 10 因子...")
    df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
    log(f"panel loaded: {df.height:,} rows")

    summary = {}
    for param_name, values in SWEEPS.items():
        log(f"\n=== Sweep {param_name} ∈ {values} ===")
        for v in values:
            params = dict(BASELINE)
            params[param_name] = v
            param_summary = {}
            for s, e, plabel in LOSS_PERIODS:
                log(f"  run {plabel} {param_name}={v}")
                t0 = time.time()
                m = run_period(df, pd.Timestamp(s), pd.Timestamp(e), params, plabel)
                if 'error' in m:
                    param_summary[plabel] = m
                    continue
                ret = m.get('total_return_pct', 0)
                sharpe = m.get('sharpe_ratio', 0)
                mdd = m.get('max_drawdown_pct', 0)
                pf = m.get('profit_factor', 0)
                win = m.get('win_rate', 0)
                param_summary[plabel] = {
                    "ret": ret, "sharpe": sharpe, "mdd": mdd,
                    "pf": pf, "win": win, "duration": time.time()-t0,
                }
                marker = "✅" if ret > 0 else "❌"
                log(f"    {marker} +{ret:.2f}% / sharpe {sharpe:.3f} / MDD -{mdd:.2f}% / PF {pf:.3f} ({time.time()-t0:.1f}s)")
            summary[f"{param_name}={v}"] = param_summary

    log("\n=== Sweep 总结 ===")
    for k, runs in summary.items():
        rets = [r.get('ret', 0) for r in runs.values() if 'ret' in r]
        if not rets: continue
        positive = sum(1 for r in rets if r > 0)
        mean = sum(rets)/len(rets)
        log(f"  {k}: {positive}/{len(rets)} 正, mean +{mean:.2f}%")

    with open(OUTDIR / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\nSummary saved: {OUTDIR/'sweep_summary.json'}")


if __name__ == "__main__":
    main()