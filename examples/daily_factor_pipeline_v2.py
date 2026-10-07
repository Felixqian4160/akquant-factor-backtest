"""
Daily Factor Reselect — Full Pipeline v2.

User-specified design:
  1. Each day, run all 406 factor short-window backtests
  2. Each rebal cycle: pick N ∈ {5,6,7,8,9,10} best factors
     - Hard filter: factor's previous-day return > 0
     - Rank by (return + sharpe)
  3. Composite → top-K picks (10-20 stocks)
  4. Hold rebal_period ∈ {5, 10, 20} days
  5. If factors unchanged → stocks unchanged → continue holding
     If factors changed → rebalance
  6. Compute NAV curve (1M start → rebal-by-rebal)

Logs: written to out_dir/pipeline.log (not stdout, safe from pipe break)
Metrics cache: out_dir/metrics_cache.parquet (incremental)
"""
import polars as pl
import numpy as np
import json
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time

PANEL_PATH = '/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet'

# Pre-load & cache panel in memory (avoid repeated parquet reads)
_PANEL_CACHE = None
_PANEL_DATES = None


def get_panel():
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        _PANEL_CACHE = pl.read_parquet(PANEL_PATH)
    return _PANEL_CACHE


def get_trading_dates(panel, start_date, end_date):
    """Get sorted trading dates in [start, end]."""
    return panel.filter(
        (pl.col('trade_date') >= pl.lit(start_date).str.strptime(pl.Date, '%Y-%m-%d')) &
        (pl.col('trade_date') <= pl.lit(end_date).str.strptime(pl.Date, '%Y-%m-%d'))
    )['trade_date'].unique().sort().to_list()


def compute_factor_metric(panel, factor, eval_date, window_days=60, top_k=20):
    """Single-factor short-window backtest. Returns {return, sharpe, ann}."""
    eval_dt = datetime.strptime(eval_date, '%Y-%m-%d')
    start_str = (eval_dt - timedelta(days=window_days * 2)).strftime('%Y-%m-%d')

    sub = panel.filter(
        (pl.col('trade_date') >= pl.lit(start_str).str.strptime(pl.Date, '%Y-%m-%d')) &
        (pl.col('trade_date') < pl.lit(eval_date).str.strptime(pl.Date, '%Y-%m-%d'))
    )
    if factor not in sub.columns:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}
    sub = sub.drop_nulls(subset=[factor])
    if sub.height < window_days * 50:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}

    dates = sub['trade_date'].unique().sort()
    if len(dates) < 30:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}

    last_idx = max(0, len(dates) - 21)
    last_rebal = dates[last_idx]
    rebal = sub.filter(pl.col('trade_date') == last_rebal).sort(factor, descending=True, nulls_last=True).head(top_k)
    if rebal.height == 0:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}

    ts_codes = rebal['ts_code'].to_list()
    fwd_dates = dates[(last_idx + 1):(last_idx + 21)]
    fwd = sub.filter(pl.col('ts_code').is_in(ts_codes) & pl.col('trade_date').is_in(fwd_dates))
    if fwd.height == 0:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}

    pivot = fwd.pivot(index='ts_code', on='trade_date', values='close').sort('ts_code')
    if pivot.height == 0 or pivot.width < 3:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}
    close_cols = [c for c in pivot.columns if c != 'ts_code']
    arr = pivot.select(close_cols).to_numpy()
    if arr.shape[1] < 2:
        return {'return': 0.0, 'sharpe': 0.0, 'ann': 0.0}
    per_ret = arr[:, -1] / arr[:, 0] - 1
    avg = float(np.nanmean(per_ret))
    std = float(np.nanstd(per_ret)) if len(per_ret) > 1 else 0.01
    n_d = arr.shape[1]
    sharpe = avg / max(std, 0.01) * np.sqrt(252 / max(n_d, 1))
    ann = (1 + avg) ** (252 / max(n_d, 1)) - 1
    return {'return': avg, 'sharpe': float(sharpe), 'ann': float(ann)}


def compute_all_metrics_for_date(panel, all_factors, date_str, log_file, window_days=60, top_k=20):
    """Compute all factor metrics for one date. Cache to disk."""
    cache_dir = Path(log_file).parent / 'metrics_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'metrics_{date_str}.json'

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return cached

    metrics = {}
    t0 = time.time()
    for i, f in enumerate(all_factors):
        metrics[f] = compute_factor_metric(panel, f, date_str, window_days, top_k)
        if (i + 1) % 50 == 0:
            with open(log_file, 'a') as fp:
                fp.write(f'  [{date_str}] {i+1}/{len(all_factors)} factors ({(i+1)/(time.time()-t0):.1f}/s)\n')

    cache_file.write_text(json.dumps({'date': date_str, 'metrics': metrics}, indent=2))
    return {'date': date_str, 'metrics': metrics}


def pick_factors(prev_metrics, all_factors, N):
    """Hard filter: prev day return > 0. Rank by return+sharpe. Pick top-N."""
    candidates = [(f, prev_metrics[f]) for f in all_factors if prev_metrics[f]['return'] > 0]
    if len(candidates) < N:
        return []
    candidates.sort(key=lambda x: x[1]['return'] + x[1]['sharpe'] * 0.01, reverse=True)
    return [c[0] for c in candidates[:N]]


def compute_picks(panel, factor_set, eval_date, top_k=20):
    """Cross-section composite rank → top-K picks."""
    if not factor_set:
        return [], []
    sub = panel.filter(pl.col('trade_date') == pl.lit(eval_date).str.strptime(pl.Date, '%Y-%m-%d'))
    sub = sub.drop_nulls(subset=factor_set)
    if sub.height == 0:
        return [], []
    rank_cols = []
    for f in factor_set:
        col = pl.col(f).rank(method='average') / pl.col(f).count()
        sub = sub.with_columns(col.alias(f'{f}_rank'))
        rank_cols.append(f'{f}_rank')
    sub = sub.with_columns(pl.mean_horizontal(rank_cols).alias('composite_rank'))
    sub = sub.sort('composite_rank', descending=True).head(top_k)
    picks = sub['ts_code'].to_list()
    scores = sub['composite_rank'].to_list()
    return picks, scores


def run_pipeline(start_date, end_date, rebal_days=20, window_days=60, top_k=20, n_options=None, cost_bps=25, slippage_bps=10):
    """Run full pipeline. Returns NAV curve + ledger + factor selection log."""
    if n_options is None:
        n_options = [5, 6, 7, 8, 9, 10]

    log_file = Path('evidence/daily_factor_reselect/pipeline_v2/pipeline.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(f'[{datetime.now().isoformat()}] start pipeline: {start_date} ~ {end_date} rebal={rebal_days}d window={window_days}d top_k={top_k} N_options={n_options}\n')

    with open('evidence/daily_factor_reselect/all_factor_candidates.json') as f:
        all_factors = json.load(f)
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] loading panel...\n')

    panel = get_panel()
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] panel loaded: {panel.height} rows × {panel.width} cols, {len(all_factors)} factors\n')

    all_dates = get_trading_dates(panel, start_date, end_date)
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] trading dates in range: {len(all_dates)} ({all_dates[0]} ~ {all_dates[-1]})\n')

    if len(all_dates) < 30:
        with open(log_file, 'a') as fp:
            fp.write(f'[{datetime.now().isoformat()}] ERROR: panel too short ({len(all_dates)} days, need ≥30)\n')
        return {'error': 'panel too short', 'n_dates': len(all_dates)}

    # Step 1: compute metrics for all dates in panel
    # Need: dates from (start_date - rebal_days) to end_date to allow (d-1) lookup
    extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=int(rebal_days * 2.5))).strftime('%Y-%m-%d')
    extended_dates = get_trading_dates(panel, extended_start, end_date)
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] computing metrics for {len(extended_dates)} dates (extended from {extended_start})\n')

    t_metrics_start = time.time()
    metrics_by_date = {}
    batch_size = int(os.environ.get('BATCH_SIZE', '999999'))  # default: all
    start_idx = int(os.environ.get('BATCH_START', '0'))
    end_idx = min(start_idx + batch_size, len(extended_dates))
    batch_dates = extended_dates[start_idx:end_idx]
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] computing metrics for {len(batch_dates)} dates (batch [{start_idx}:{end_idx}] of {len(extended_dates)})\n')

    for di, date in enumerate(batch_dates):
        date_str = date.strftime('%Y-%m-%d')
        m = compute_all_metrics_for_date(panel, all_factors, date_str, str(log_file), window_days, top_k)
        metrics_by_date[date_str] = m['metrics']
        if (di + 1) % 5 == 0:
            elapsed = time.time() - t_metrics_start
            rate = (di + 1) / elapsed
            eta = (len(batch_dates) - di - 1) / rate
            with open(log_file, 'a') as fp:
                fp.write(f'[{datetime.now().isoformat()}] metrics: {di+1}/{len(batch_dates)} ({rate:.1f} dates/s, ETA {eta/60:.1f}min)\n')

    metrics_elapsed = time.time() - t_metrics_start
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] metrics done: {metrics_elapsed:.1f}s\n')

    # Step 2: rebal cycles
    in_range_dates = get_trading_dates(panel, start_date, end_date)
    rebal_indices = list(range(0, len(in_range_dates), rebal_days))
    if rebal_indices[-1] != len(in_range_dates) - 1:
        rebal_indices.append(len(in_range_dates) - 1)
    rebal_dates = [in_range_dates[i] for i in rebal_indices]
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] {len(rebal_dates)} rebal cycles: {[d.strftime("%Y-%m-%d") for d in rebal_dates]}\n')

    # Step 3: per-rebal picks + NAV
    NAV = 1.0
    NAV_curve = []  # (date, nav)
    ledger = []  # per rebal entry
    prev_picks = []  # for rebalance detection

    for ri, rebal_date in enumerate(rebal_dates):
        date_str = rebal_date.strftime('%Y-%m-%d')

        # Find prev day (d-1) — must be before rebal_date in panel
        prev_str = None
        for d in extended_dates:
            if d < rebal_date:
                prev_str = d.strftime('%Y-%m-%d')
            else:
                break
        if prev_str is None or prev_str not in metrics_by_date:
            with open(log_file, 'a') as fp:
                fp.write(f'[{date_str}] SKIP: no prev day metrics\n')
            continue
        prev_metrics = metrics_by_date[prev_str]

        # For each N option, pick factors + picks
        all_picks_per_n = {}
        for N in n_options:
            factors = pick_factors(prev_metrics, all_factors, N)
            if not factors:
                continue
            picks, scores = compute_picks(panel, factors, date_str, top_k)
            if not picks:
                continue
            all_picks_per_n[N] = {'factors': factors, 'picks': picks, 'scores': scores}

        if not all_picks_per_n:
            with open(log_file, 'a') as fp:
                fp.write(f'[{date_str}] SKIP: no N produced picks\n')
            continue

        # Pick N with most picks (proxy); if ties, pick smallest N (lower turnover)
        chosen_N = max(all_picks_per_n.keys(), key=lambda n: (len(all_picks_per_n[n]['picks']), -n))
        choice = all_picks_per_n[chosen_N]

        # Detect rebalance: compare with prev rebal picks
        rebalance = (choice['picks'] != prev_picks) if prev_picks else True

        # Forward return: next rebal_days days
        end_idx = min(rebal_indices[ri] + rebal_days, len(in_range_dates) - 1)
        end_dt = in_range_dates[end_idx]
        next_rebal_str = end_dt.strftime('%Y-%m-%d')

        # Get open @ rebal_date and close @ next_rebal_str for picks
        open_p = panel.filter(pl.col('trade_date') == rebal_date).select(['ts_code', 'open'])
        close_p = panel.filter(pl.col('trade_date') == end_dt).select(['ts_code', 'close'])
        joined = open_p.join(close_p, on='ts_code', how='inner')
        picks_data = joined.filter(pl.col('ts_code').is_in(choice['picks']))
        if picks_data.height == 0:
            continue

        per_pick_ret = []
        for row in picks_data.to_dicts():
            entry = float(row['open'])
            exit_ = float(row['close'])
            ret = (exit_ / entry - 1) - 2 * (cost_bps / 10000.0)
            per_pick_ret.append(ret)

        avg_ret = float(np.mean(per_pick_ret))
        NAV *= (1 + avg_ret)
        NAV_curve.append((date_str, NAV, len(choice['picks']), chosen_N, choice['factors'][:3]))

        ledger.append({
            'rebal_date': date_str,
            'next_eval_date': next_rebal_str,
            'N': chosen_N,
            'factors': choice['factors'],
            'picks': choice['picks'],
            'scores': choice['scores'],
            'rebalance': rebalance,
            'prev_picks': prev_picks,
            'new_picks': [p for p in choice['picks'] if p not in prev_picks],
            'dropped_picks': [p for p in prev_picks if p not in choice['picks']],
            'avg_return_pct': avg_ret * 100,
            'nav': NAV,
        })

        prev_picks = choice['picks']
        with open(log_file, 'a') as fp:
            fp.write(f'[{date_str}] N={chosen_N} picks={len(choice["picks"])} factors={choice["factors"][:3]}... ret={avg_ret*100:+.2f}% NAV={NAV:.4f} {"REBAL" if rebalance else "HOLD"}\n')

    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] pipeline done: {len(ledger)} rebal cycles, final NAV={NAV:.4f} ({(NAV-1)*100:+.2f}%)\n')

    # Output
    result = {
        'config': {
            'start_date': start_date,
            'end_date': end_date,
            'rebal_days': rebal_days,
            'window_days': window_days,
            'top_k': top_k,
            'n_options': n_options,
            'cost_bps': cost_bps,
            'slippage_bps': slippage_bps,
        },
        'metrics_elapsed_sec': metrics_elapsed,
        'n_rebal_cycles': len(ledger),
        'final_nav': NAV,
        'final_return_pct': (NAV - 1) * 100,
        'nav_curve': NAV_curve,
        'ledger': ledger,
    }
    out_file = log_file.parent / f'pipeline_{start_date}_{end_date}_rebal{rebal_days}.json'
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    with open(log_file, 'a') as fp:
        fp.write(f'[{datetime.now().isoformat()}] wrote {out_file}\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default='2024-03-31')
    parser.add_argument('--rebal-days', type=int, default=20)
    parser.add_argument('--window-days', type=int, default=60)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()

    result = run_pipeline(args.start_date, args.end_date, args.rebal_days, args.window_days, args.top_k)
    print(f'final_nav={result["final_nav"]:.4f} (return={result["final_return_pct"]:+.2f}%)')
    print(f'n_rebal={result["n_rebal_cycles"]}')


if __name__ == '__main__':
    main()
