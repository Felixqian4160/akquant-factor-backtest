"""V32 voting 动态仓位 (修复版) — 基于 daily NAV 累积 + 真正敏感公式.

Bug 修复:
- v1: global NAV history, 但 on_cross_section 只在 rebalance day 调用
- v2: self.equity, 同样只在 rebalance day 更新, history 只 ~13 个点
- v3: on_after_trading 累积 daily NAV, 但 vol_factor 公式 saturate 到 2.0, target 永远 ≈ 0.99

v4 修复公式:
  vol_factor = clip(0.01/vol, 0.2, 1.0)  # vol=0.01→1.0, vol=0.02→0.5
  dd_factor  = clip(1+dd*5, 0.25, 1.0)   # dd=-0.05→0.75, dd=-0.15→0.25
  target     = clip(0.99 × vol_factor × dd_factor, 0.1, 0.99)

不同 vol/dd 下 target 范围: 0.34 ~ 0.99 (真正动态).
"""
from __future__ import annotations

import json
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
OUTDIR = Path("evidence/v32_dynamic_v4_20261006")
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


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dyn_target_v4(eq_hist: list[float], base: float = 0.99) -> float:
    """eq_hist: 每日 NAV 序列 (含今日), 长度 ≥ 5.
    返回 [0.1, 0.99] target_total.
    """
    if len(eq_hist) < 5:
        return base
    arr = np.array(eq_hist[-21:], dtype=np.float64)
    if arr.mean() <= 0:
        return 0.5
    ret = arr[1:] / arr[:-1] - 1.0
    vol = float(ret.std()) if len(ret) > 1 else 0.01
    vol_factor = float(np.clip(0.01 / max(vol, 0.005), 0.2, 1.0))
    peak = arr.max()
    dd = arr[-1] / peak - 1.0 if peak > 0 else 0.0
    dd_factor = float(np.clip(1.0 + dd * 5.0, 0.25, 1.0))
    target = base * vol_factor * dd_factor
    return float(np.clip(target, 0.1, 0.99))


def run_period(df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label: str) -> dict:
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
    _TARGETS_LOG: list[tuple] = []

    class DynV4Strategy(aq.Strategy):
        warmup = 5

        def on_bar(self, bar):
            pass

        def on_after_trading(self, trading_date, timestamp):
            """每日触发 — 累积 daily NAV history."""
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
            """rebalance day 触发 — 用最新 NAV history 算 v4 dynamic target."""
            d = str(trading_date)[:10]
            if d not in _REBAL_SET:
                return
            p = _DAILY_PICKS.get(d, {})
            if not p:
                return
            n = len(p)
            target = dyn_target_v4(_NAV_HISTORY)
            self.rebalance_weights(
                target_weights={s: target / n for s in p},
                liquidate_unmentioned=True,
            )
            _TARGETS_LOG.append((d, target, _NAV_HISTORY[-1] if _NAV_HISTORY else 0, len(_NAV_HISTORY)))

    try:
        result = aq.run_backtest(
            data=data, strategy=DynV4Strategy,
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

    out_dir = OUTDIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'result.json', 'w') as f:
        json.dump({
            "period": f"{start.date()} ~ {end.date()}",
            "label": label,
            "pool": POOL,
            "params": {"MIN_VOTES": MIN_VOTES, "MIN_STOCKS": MIN_STOCKS,
                       "MAX_STOCKS": MAX_STOCKS, "REBAL_STEP": REBAL_STEP,
                       "DYNAMIC_TARGET_v4": "vol_factor=clip(0.01/vol,0.2,1), dd_factor=clip(1+dd*5,0.25,1)"},
            "metrics": metrics_dict,
            "targets_log": _TARGETS_LOG,
        }, f, indent=2, ensure_ascii=False, default=str)
    return metrics_dict


def main():
    log("=" * 80)
    log("V32 voting 动态仓位 v4 (修复公式 + on_after_trading 累积 daily NAV)")
    log("=" * 80)
    log(f"加载 panel + 10 因子...")
    df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol'] + list(POOL.keys()))
    log(f"panel loaded: {df.height:,} rows")

    summary = {}
    for s, e, label in LOSS_PERIODS:
        log(f"\n=== {label} ({s} ~ {e}) ===")
        t0 = time.time()
        m = run_period(df, pd.Timestamp(s), pd.Timestamp(e), label)
        if 'error' in m:
            summary[label] = m
            continue
        ret = m.get('total_return_pct', 0)
        sharpe = m.get('sharpe_ratio', 0)
        mdd = m.get('max_drawdown_pct', 0)
        pf = m.get('profit_factor', 0)
        win = m.get('win_rate', 0)
        n_trades = int(m.get('closed_trade_count', 0))
        summary[label] = {
            "ret": ret, "sharpe": sharpe, "mdd": mdd,
            "pf": pf, "win": win, "n_trades": n_trades,
            "duration": time.time()-t0,
        }
        marker = "✅" if ret > 0 else "❌"
        log(f"  {marker} +{ret:.2f}% / sharpe {sharpe:.3f} / MDD -{mdd:.2f}% / PF {pf:.3f} / Win {win:.1f}% ({n_trades} trades, {time.time()-t0:.1f}s)")

    log("\n=== Dynamic v4 vs Static baseline 对比 ===")
    baseline = {"2011_yin_die": -30.96, "2015H2_gushai": -25.14,
                "2018_quan_nian_xia_die": -22.87, "2022_bear": -6.36}
    for label, m in summary.items():
        if 'error' in m: continue
        b = baseline[label]
        d = m['ret'] - b
        marker = "✅" if m['ret'] > 0 else "❌"
        log(f"  {marker} {label}: {b:+.2f}% (static) → {m['ret']:+.2f}% (dynamic v4), delta {d:+.2f}%")
    log("")
    rets = [m['ret'] for m in summary.values() if 'ret' in m]
    log(f"  mean dynamic v4: {sum(rets)/len(rets):+.2f}%  vs mean static: {sum(baseline.values())/4:+.2f}%")

    with open(OUTDIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()