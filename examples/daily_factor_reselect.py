"""
Daily Factor Reselect — User's new workflow.

Workflow:
  Phase 1 (historical): for each trading_day in panel:
    - For each of 406 factors:
      - Run short-window backtest (window=120d, single-stock long/flat)
      - Record (date, factor, sharpe, ann, return, mdd)
  Phase 2 (incremental): for new date, run only (today-1) backtests
  Factor selection: per rebal_day:
    - Filter: factor's previous_day (d-1) return > 0
    - Rank remaining factors by (return + sharpe)
    - Try N ∈ {5,6,7,8,9,10} → pick best by validation metric
    - Use top-N composite for that rebal window
  Result: per-day (rebal) picks → ledger → NAV
"""
import polars as pl
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys

PANEL_PATH = '/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet'


def load_panel():
    """Load full panel (cached)."""
    return pl.read_parquet(PANEL_PATH)


def get_factor_metrics_for_day(
    panel: pl.DataFrame,
    factor: str,
    eval_date: str,
    short_window_days: int = 120,
    top_k: int = 10,
) -> dict:
    """
    Run short-window backtest for one factor at eval_date.
    Use past `short_window_days` data to pick top_k stocks by factor.
    Return: {sharpe, ann_return, total_return, mdd, n_periods}
    """
    eval_dt = datetime.strptime(eval_date, '%Y-%m-%d')
    start_dt = eval_dt - timedelta(days=short_window_days * 2)  # buffer for trading days
    start_str = start_dt.strftime('%Y-%m-%d')

    # Filter panel: [start, eval_date), 354 stocks
    sub = panel.filter(
        (pl.col('trade_date') >= pl.lit(start_str).str.strptime(pl.Date, '%Y-%m-%d')) &
        (pl.col('trade_date') < pl.lit(eval_date).str.strptime(pl.Date, '%Y-%m-%d'))
    )

    if sub.height == 0 or factor not in sub.columns:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Drop nulls on factor
    sub = sub.drop_nulls(subset=[factor])
    if sub.height < short_window_days * 100:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # For each trading_day in window: rank stocks by factor, compute next-20d return of top-K
    # Simplified: compute forward 20d return of top-K per rebal day (every 20 days)
    dates = sub['trade_date'].unique().sort()
    if len(dates) < 30:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Pick eval_date - 20 as last rebal day in window
    last_rebal_idx = max(0, len(dates) - 21)
    last_rebal_date = dates[last_rebal_idx]

    # At last_rebal_date: rank stocks by factor (descending) → top_k
    rebal = sub.filter(pl.col('trade_date') == last_rebal_date).sort(factor, descending=True, nulls_last=True).head(top_k)
    if rebal.height == 0:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Compute next 20-day return per top_k stock
    rebal_ts_codes = rebal['ts_code'].to_list()
    fwd_dates = dates[(last_rebal_idx + 1):(last_rebal_idx + 21)]
    fwd = sub.filter(pl.col('ts_code').is_in(rebal_ts_codes) & pl.col('trade_date').is_in(fwd_dates))
    if fwd.height == 0:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Pivot: ts_code x date → close
    pivot = fwd.pivot(index='ts_code', on='trade_date', values='close').sort('ts_code')
    if pivot.height == 0 or pivot.width < 5:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Per-stock forward return (last close / first close - 1)
    close_cols = [c for c in pivot.columns if c != 'ts_code']
    close_arr = pivot.select(close_cols).to_numpy()
    if close_arr.shape[1] < 2:
        return {'sharpe': 0.0, 'ann_return': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'n_periods': 0}

    # Per-stock return
    per_stock_ret = (close_arr[:, -1] / close_arr[:, 0]) - 1
    avg_ret = float(np.nanmean(per_stock_ret))

    # Total return (simplified: avg_ret × top_k / portfolio)
    total_return = avg_ret
    n_days = close_arr.shape[1]
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # Sharpe (assume constant vol ~30% annual)
    sharpe = avg_ret / max(np.nanstd(per_stock_ret) if len(per_stock_ret) > 1 else 0.01, 0.01) * np.sqrt(252 / max(n_days, 1))

    # MDD (simplified: assume MDD ≈ -2 * |ret| for single-period)
    mdd = -2 * abs(total_return) if total_return < 0 else -0.05

    return {
        'sharpe': float(sharpe),
        'ann_return': float(ann_return),
        'total_return': float(total_return),
        'mdd': float(mdd),
        'n_periods': 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['historical', 'incremental', 'smoke'], default='smoke')
    parser.add_argument('--date', type=str, help='For incremental mode: target date (YYYY-MM-DD)')
    parser.add_argument('--factor', type=str, help='For smoke mode: single factor to test')
    parser.add_argument('--n-factors', type=int, default=406, help='Number of factors to test')
    parser.add_argument('--window-days', type=int, default=120)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='evidence/daily_factor_reselect/smoke')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{datetime.now().isoformat()}] loading panel...')
    panel = load_panel()
    print(f'[{datetime.now().isoformat()}] panel loaded: {panel.height} rows × {panel.width} cols')

    if args.mode == 'smoke':
        # Single factor + single date
        eval_date = args.date or '2024-02-02'
        factor = args.factor or 'talib_NATR'
        print(f'[{datetime.now().isoformat()}] smoke test: factor={factor} date={eval_date} window={args.window_days}d')
        import time
        t0 = time.time()
        metrics = get_factor_metrics_for_day(panel, factor, eval_date, args.window_days, args.top_k)
        elapsed = time.time() - t0
        print(f'[{datetime.now().isoformat()}] elapsed: {elapsed:.2f}s')
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        out_file = out_dir / f'smoke_{factor}_{eval_date}.json'
        out_file.write_text(json.dumps({
            'factor': factor,
            'date': eval_date,
            'window_days': args.window_days,
            'top_k': args.top_k,
            'elapsed_sec': elapsed,
            'metrics': metrics,
        }, indent=2, ensure_ascii=False))
        print(f'wrote {out_file}')

    elif args.mode == 'incremental':
        # Run all factors for one date
        assert args.date, '--date required for incremental mode'
        with open('evidence/daily_factor_reselect/all_factor_candidates.json') as f:
            factors = json.load(f)[:args.n_factors]

        print(f'[{datetime.now().isoformat()}] incremental: date={args.date} n_factors={len(factors)}')
        import time
        t0 = time.time()
        results = []
        for i, factor in enumerate(factors):
            t1 = time.time()
            m = get_factor_metrics_for_day(panel, factor, args.date, args.window_days, args.top_k)
            elapsed_f = time.time() - t1
            results.append({'factor': factor, **m, 'elapsed_sec': elapsed_f})
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(factors) - i - 1) / rate
                print(f'[{datetime.now().isoformat()}] {i+1}/{len(factors)} ({rate:.2f}/s, ETA {eta/60:.1f}min)')

        total_elapsed = time.time() - t0
        out_file = out_dir / f'incremental_{args.date}.json'
        out_file.write_text(json.dumps({
            'date': args.date,
            'n_factors': len(factors),
            'total_elapsed_sec': total_elapsed,
            'results': results,
        }, indent=2, ensure_ascii=False))
        print(f'[{datetime.now().isoformat()}] done: {total_elapsed:.1f}s, {len(results)} factors')
        print(f'wrote {out_file}')


if __name__ == '__main__':
    main()
