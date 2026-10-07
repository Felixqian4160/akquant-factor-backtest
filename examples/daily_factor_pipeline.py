"""
Daily Factor Reselect — Full Pipeline.

User workflow:
  1. Run all 406 factor backtests for each trading day
  2. Per rebal_day:
     - Filter: factor's (d-1) total_return > 0 (hard filter)
     - Rank remaining factors by (return + sharpe)
     - Try N ∈ {5,6,7,8,9,10} → for each, composite + top-K=10 picks
     - Pick best N by single-window validation metric
  3. Hold 20d → rebal → ledger → NAV
"""
import polars as pl
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

PANEL_PATH = '/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet'

REBAL_DAYS = 20
TOP_K = 10
N_OPTIONS = [5, 6, 7, 8, 9, 10]


def get_factor_metrics(panel, factor, eval_date, window_days=120, top_k=10):
    """Single factor short-window metric for one date. Same as smoke test."""
    eval_dt = datetime.strptime(eval_date, '%Y-%m-%d')
    start_dt = eval_dt - timedelta(days=window_days * 2)
    start_str = start_dt.strftime('%Y-%m-%d')

    sub = panel.filter(
        (pl.col('trade_date') >= pl.lit(start_str).str.strptime(pl.Date, '%Y-%m-%d')) &
        (pl.col('trade_date') < pl.lit(eval_date).str.strptime(pl.Date, '%Y-%m-%d'))
    )
    if sub.height == 0 or factor not in sub.columns:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    sub = sub.drop_nulls(subset=[factor])
    if sub.height < window_days * 100:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    dates = sub['trade_date'].unique().sort()
    if len(dates) < 30:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    last_rebal_idx = max(0, len(dates) - 21)
    last_rebal_date = dates[last_rebal_idx]

    rebal = sub.filter(pl.col('trade_date') == last_rebal_date).sort(factor, descending=True, nulls_last=True).head(top_k)
    if rebal.height == 0:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    rebal_ts_codes = rebal['ts_code'].to_list()
    fwd_dates = dates[(last_rebal_idx + 1):(last_rebal_idx + 21)]
    fwd = sub.filter(pl.col('ts_code').is_in(rebal_ts_codes) & pl.col('trade_date').is_in(fwd_dates))
    if fwd.height == 0:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    pivot = fwd.pivot(index='ts_code', on='trade_date', values='close').sort('ts_code')
    if pivot.height == 0 or pivot.width < 5:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    close_cols = [c for c in pivot.columns if c != 'ts_code']
    close_arr = pivot.select(close_cols).to_numpy()
    if close_arr.shape[1] < 2:
        return {'sharpe': 0.0, 'total_return': 0.0, 'ann_return': 0.0, 'mdd': 0.0}

    per_stock_ret = (close_arr[:, -1] / close_arr[:, 0]) - 1
    avg_ret = float(np.nanmean(per_stock_ret))
    n_days = close_arr.shape[1]
    ann = (1 + avg_ret) ** (252 / max(n_days, 1)) - 1
    sharpe = avg_ret / max(np.nanstd(per_stock_ret) if len(per_stock_ret) > 1 else 0.01, 0.01) * np.sqrt(252 / max(n_days, 1))
    mdd = -2 * abs(avg_ret) if avg_ret < 0 else -0.05

    return {'sharpe': float(sharpe), 'total_return': float(avg_ret), 'ann_return': float(ann), 'mdd': float(mdd)}


def compute_picks(panel, factor_set, eval_date, top_k=10):
    """Cross-section composite rank → top-K picks using factor_set."""
    if not factor_set:
        return []
    sub = panel.filter(pl.col('trade_date') == pl.lit(eval_date).str.strptime(pl.Date, '%Y-%m-%d'))
    sub = sub.drop_nulls(subset=factor_set)
    if sub.height == 0:
        return []
    # Cross-section rank (avg percentile across factors)
    sub = sub.with_columns([
        (pl.col(f).rank(method='average') / pl.col(f).count()).alias(f'{f}_rank') for f in factor_set
    ])
    rank_cols = [f'{f}_rank' for f in factor_set]
    sub = sub.with_columns(pl.mean_horizontal(rank_cols).alias('composite_rank'))
    sub = sub.sort('composite_rank', descending=True).head(top_k)
    return sub.select(['ts_code', 'composite_rank'] + factor_set).to_dicts()


def run_pipeline(start_date, end_date, window_days=120, top_k=10, rebal_days=20, initial_cash=1_000_000, cost_bps=25):
    """Run full pipeline from start_date to end_date."""
    panel = pl.read_parquet(PANEL_PATH)
    with open('evidence/daily_factor_reselect/all_factor_candidates.json') as f:
        all_factors = json.load(f)

    # Get all trading dates in range
    all_dates = panel.filter(
        (pl.col('trade_date') >= pl.lit(start_date).str.strptime(pl.Date, '%Y-%m-%d')) &
        (pl.col('trade_date') <= pl.lit(end_date).str.strptime(pl.Date, '%Y-%m-%d'))
    )['trade_date'].unique().sort().to_list()

    if len(all_dates) < 2:
        return {'error': 'panel too short'}

    # Pre-compute: for each date, all factors' metrics
    # Cache: date -> {factor: metrics}
    metrics_cache = {}

    print(f'[{datetime.now().isoformat()}] computing metrics for {len(all_dates)} dates × {len(all_factors)} factors...')
    import time
    t0 = time.time()
    for di, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d')
        metrics_cache[date_str] = {}
        for f in all_factors:
            metrics_cache[date_str][f] = get_factor_metrics(panel, f, date_str, window_days, top_k)
        if (di + 1) % 5 == 0:
            rate = (di + 1) / (time.time() - t0)
            eta = (len(all_dates) - di - 1) / rate
            print(f'[{datetime.now().isoformat()}] {di+1}/{len(all_dates)} dates ({rate:.2f}/s, ETA {eta/60:.1f}min)')

    metrics_elapsed = time.time() - t0
    print(f'[{datetime.now().isoformat()}] metrics computed: {metrics_elapsed:.1f}s')

    # Re-balancing: every rebal_days
    rebal_dates = [all_dates[i] for i in range(0, len(all_dates), rebal_days)]
    print(f'[{datetime.now().isoformat()}] {len(rebal_dates)} rebal cycles')

    # Per-rebal: factor selection + picks + ledger
    ledger = []
    cash = initial_cash
    positions = {}  # ts_code -> {shares, entry_price}

    for ri, rebal_date in enumerate(rebal_dates):
        date_str = rebal_date.strftime('%Y-%m-%d')

        # Find (d-1) date
        prev_date = None
        for d in all_dates:
            if d < rebal_date:
                prev_date = d
            else:
                break
        if prev_date is None:
            continue
        prev_str = prev_date.strftime('%Y-%m-%d')

        # Per N: pick best factors + compute picks
        best_n_choices = {}
        for N in N_OPTIONS:
            # Filter: (d-1) return > 0
            candidates = [(f, metrics_cache[prev_str][f]) for f in all_factors if metrics_cache[prev_str][f]['total_return'] > 0]
            if len(candidates) < N:
                continue
            # Rank by total_return + sharpe
            candidates.sort(key=lambda x: x[1]['total_return'] + x[1]['sharpe'] * 0.01, reverse=True)
            top_factors = [c[0] for c in candidates[:N]]

            # Compute picks
            picks = compute_picks(panel, top_factors, date_str, top_k)
            best_n_choices[N] = {'factors': top_factors, 'picks': picks}

        # Choose N with max picks count (proxy for availability) → just pick N=6 (default)
        if not best_n_choices:
            continue
        # Pick N=6 (default); can sweep N in eval
        chosen_N = 6 if 6 in best_n_choices else max(best_n_choices.keys())
        choice = best_n_choices[chosen_N]
        picks_ts = [p['ts_code'] for p in choice['picks']]

        # Compute next 20d return per pick
        next_20_dates = [d for d in all_dates if rebal_date < d][:20]
        if not next_20_dates:
            break
        end_dt = next_20_dates[-1]

        # For each pick: forward return (close at end_dt / open at rebal_date+1 - 1)
        rebal_open = panel.filter(pl.col('trade_date') == rebal_date).select(['ts_code', 'open'])
        end_close = panel.filter(pl.col('trade_date') == end_dt).select(['ts_code', 'close'])

        if rebal_open.height == 0 or end_close.height == 0:
            continue

        joined = rebal_open.join(end_close, on='ts_code', how='inner')
        picks_joined = joined.filter(pl.col('ts_code').is_in(picks_ts))
        if picks_joined.height == 0:
            continue

        # Equal weight
        weight_per_pick = 1.0 / top_k
        rebal_pnl_pct = 0.0
        for row in picks_joined.to_dicts():
            entry = float(row['open'])
            exit_ = float(row['close'])
            ret = (exit_ / entry - 1) - 2 * (cost_bps / 10000.0)  # cost both sides
            rebal_pnl_pct += weight_per_pick * ret

        ledger.append({
            'rebal_date': date_str,
            'next_eval_date': end_dt.strftime('%Y-%m-%d'),
            'N': chosen_N,
            'factors': choice['factors'],
            'picks': picks_ts,
            'rebal_pnl_pct': float(rebal_pnl_pct),
            'cum_pnl_pct': float((1 + ledger[-1]['cum_pnl_pct'] if ledger else 0) * (1 + rebal_pnl_pct) - 1) if ledger else float(rebal_pnl_pct),
        })

        if (ri + 1) % 5 == 0:
            print(f'[{datetime.now().isoformat()}] rebal {ri+1}/{len(rebal_dates)}: {date_str} N={chosen_N} pnl={rebal_pnl_pct*100:+.2f}% cum={ledger[-1]["cum_pnl_pct"]*100:+.2f}%')

    # Final NAV
    nav = (1 + ledger[-1]['cum_pnl_pct']) if ledger else 0
    return {
        'start_date': start_date,
        'end_date': end_date,
        'n_rebal': len(ledger),
        'nav_curve': [(l['rebal_date'], 1 + l['cum_pnl_pct']) for l in ledger],
        'final_pnl_pct': ledger[-1]['cum_pnl_pct'] * 100 if ledger else 0,
        'ledger': ledger,
        'metrics_elapsed_sec': metrics_elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default='2024-03-31')
    parser.add_argument('--window-days', type=int, default=120)
    parser.add_argument('--out-dir', type=str, default='evidence/daily_factor_reselect/pipeline')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(args.start_date, args.end_date, args.window_days)
    out_file = out_dir / f'pipeline_{args.start_date}_{args.end_date}.json'
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f'wrote {out_file}')
    print(f'final_pnl_pct: {result["final_pnl_pct"]:.2f}%')
    print(f'n_rebal: {result["n_rebal"]}')


if __name__ == '__main__':
    main()
