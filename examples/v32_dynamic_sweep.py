"""V32 voting dynamic v4 参数 sweep — 找最优 vol/dd 因子.

参数 sweep (3×3 = 9 组合):
  vol_base ∈ {0.005, 0.01, 0.015}
  dd_slope ∈ {3.0, 5.0, 7.0}

每个组合 × 4 亏损时段 = 36 backtests.

公式:
  vol_factor = clip(vol_base / max(vol, vol_base), 0.2, 1.0)
  dd_factor  = clip(1 + dd * dd_slope, 0.25, 1.0)
  target     = clip(0.99 × vol_factor × dd_factor, 0.1, 0.99)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUTDIR = Path("evidence/v32_dynamic_sweep_20261006")
OUTDIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
K_NOM = 10
MIN_VOTES, MIN_STOCKS, MAX_STOCKS = 3, 5, 20
REBAL_STEP = 20

POOL = {
    'l_ami': 'hi', 'gtja_gtja_144': 'hi', 'l_size3': 'lo', 'l_size': 'lo',
    'gtja_gtja_132': 'lo', 'l_dtvm': 'lo', 'r_tv': 'hi', 'gtja_gtja_070': 'lo',
    'gtja_gtja_095': 'lo', 'talib_NATR': 'hi',
}

LOSS_PERIODS = [
    ("2011-01-01", "2012-01-01", "2011_yin_die"),
    ("2015-06-01", "2016-06-01", "2015H2_gushai"),
    ("2018-01-01", "2019-01-01", "2018_quan_nian_xia_die"),
    ("2022-01-01", "2023-01-01", "2022_bear"),
]

SWEEP_GRID = list(product([0.005, 0.01, 0.015], [3.0, 5.0, 7.0]))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dyn_target(eq_hist: list[float], vol_base: float, dd_slope: float,
               base: float = 0.99) -> float:
    if len(eq_hist) < 5:
        return base
    arr = np.array(eq_hist[-21:], dtype=np.float64)
    if arr.mean() <= 0:
        return 0.5
    ret = arr[1:] / arr[:-1] - 1.0
    vol = float(ret.std()) if len(ret) > 1 else 0.01
    vol_factor = float(np.clip(vol_base / max(vol, vol_base), 0.2, 1.0))
    peak = arr.max()
    dd = arr[-1] / peak - 1.0 if peak > 0 else 0.0
    dd_factor = float(np.clip(1.0 + dd * dd_slope, 0.25, 1.0))
    target = base * vol_factor * dd_factor
    return float(np.clip(target, 0.1, 0.99))


def run_period(df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
               label: str, vol_base: float, dd_slope: float) -> dict:
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
        if pdf.empty:
            continue
        pdf['symbol'] = sym
        pdf = pdf.drop(columns=['ts_code'])
        pdf['volume'] = 1.0e9
        data[sym] = pdf

    _DAILY_PICKS = picks
    _REBAL_SET = set(picks.keys())
    _NAV_HISTORY: list[float] = [INITIAL_CASH]
    _VB = vol_base
    _DS = dd_slope

    class DynSweepStrategy(aq.Strategy):
        warmup = 5

        def on_bar(self, bar):
            pass

        def on_after_trading(self, trading_date, timestamp):
            try:
                eq = float(self.equity)
                if eq > 0:
                    if not _NAV_HISTORY or _NAV_HISTORY[-1] != eq:
                        _NAV_HISTORY.append(eq)
                        if len(_NAV_HISTORY) > 30:
                            _NAV_HISTORY.pop(0)
            except Exception:
                pass

        def on_cross_section(self, trading_date, timestamp):
            d = str(trading_date)[:10]
            if d not in _REBAL_SET:
                return
            p = _DAILY_PICKS.get(d, {})
            if not p:
                return
            n = len(p)
            target = dyn_target(_NAV_HISTORY, _VB, _DS)
            self.rebalance_weights(
                target_weights={s: target / n for s in p},
                liquidate_unmentioned=True,
            )

    try:
        result = aq.run_backtest(
            data=data, strategy=DynSweepStrategy,
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
    log("V32 voting dynamic v4 sweep: vol_base × dd_slope")
    log("=" * 80)
    df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
    log(f"panel loaded: {df.height:,} rows")
    log(f"sweep grid: {len(SWEEP_GRID)} combos × {len(LOSS_PERIODS)} periods = {len(SWEEP_GRID)*len(LOSS_PERIODS)} runs")

    summary = {}
    for vb, ds in SWEEP_GRID:
        log(f"\n=== vol_base={vb}, dd_slope={ds} ===")
        runs = {}
        for s, e, plabel in LOSS_PERIODS:
            t0 = time.time()
            m = run_period(df, pd.Timestamp(s), pd.Timestamp(e), plabel, vb, ds)
            if 'error' in m:
                runs[plabel] = m
                continue
            runs[plabel] = {
                "ret": m.get('total_return_pct', 0),
                "sharpe": m.get('sharpe_ratio', 0),
                "mdd": m.get('max_drawdown_pct', 0),
                "pf": m.get('profit_factor', 0),
                "win": m.get('win_rate', 0),
                "n_trades": int(m.get('closed_trade_count', 0)),
                "duration": time.time()-t0,
            }
            r = runs[plabel]['ret']
            marker = "✅" if r > 0 else "❌"
            log(f"  {marker} {plabel}: +{r:.2f}% ({time.time()-t0:.1f}s)")
        summary[f"vol_base={vb}_dd_slope={ds}"] = runs
        # Mean
        rets = [r['ret'] for r in runs.values() if 'ret' in r]
        if rets:
            log(f"  mean: {sum(rets)/len(rets):+.2f}%")

    log("\n" + "=" * 80)
    log("Sweep 汇总 (按 mean_ret 排序)")
    log("=" * 80)
    rows = []
    for k, runs in summary.items():
        rets = [r.get('ret', 0) for r in runs.values() if 'ret' in r]
        if not rets:
            continue
        rows.append((k, sum(rets)/len(rets), len([r for r in rets if r > 0]),
                     *[runs.get(p, {}).get('ret', 0) for p in ["2011_yin_die","2015H2_gushai","2018_quan_nian_xia_die","2022_bear"]]))
    rows.sort(key=lambda x: -x[1])
    log(f"{'config':30s} {'mean':>8} {'n_pos':>5} {'2011':>8} {'2015H2':>8} {'2018':>8} {'2022':>8}")
    for k, m, n, r1, r2, r3, r4 in rows:
        log(f"{k:30s} {m:>+7.2f}% {n:>5} {r1:>+7.2f}% {r2:>+7.2f}% {r3:>+7.2f}% {r4:>+7.2f}%")

    with open(OUTDIR / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\nSummary saved: {OUTDIR/'sweep_summary.json'}")


if __name__ == "__main__":
    main()