"""
Plot NAV curve vs HS300 benchmark.

Reads evidence/daily_factor_reselect/pipeline_v2/pipeline_2010-01-01_2025-12-31_rebal20.json
       + evidence/daily_factor_reselect/full_2010_2025/hs300_buyhold_nav.json
Outputs evidence/daily_factor_reselect/full_2010_2025/nav_vs_hs300.png
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import sys


def main():
    pipeline_json = Path('evidence/daily_factor_reselect/pipeline_v2/pipeline_2010-01-01_2025-12-31_rebal20.json')
    benchmark_json = Path('evidence/daily_factor_reselect/full_2010_2025/hs300_buyhold_nav.json')
    out_png = Path('evidence/daily_factor_reselect/full_2010_2025/nav_vs_hs300.png')
    out_csv = Path('evidence/daily_factor_reselect/full_2010_2025/nav_vs_hs300.csv')

    if not pipeline_json.exists():
        print(f'ERROR: {pipeline_json} not found. Wait for driver to complete.')
        return
    if not benchmark_json.exists():
        print(f'ERROR: {benchmark_json} not found.')
        return

    # Load
    with open(pipeline_json) as f:
        p = json.load(f)
    with open(benchmark_json) as f:
        b = json.load(f)

    print(f'pipeline: n_rebal={p["n_rebal_cycles"]}, final_nav={p["final_nav"]:.4f}, final_return={p["final_return_pct"]:+.2f}%')
    print(f'benchmark: {b["benchmark_name"]}, buy_hold_return={b["buy_hold_return_pct"]:+.2f}%')

    # Strategy NAV (only at rebal dates)
    strat_dates = []
    strat_nav = []
    for date_str, nav, _, _, _ in p['nav_curve']:
        strat_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        strat_nav.append(nav)

    # Benchmark NAV (daily)
    bench_dates = []
    bench_nav = []
    for date_str, nav in b['nav_curve']:
        bench_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        bench_nav.append(nav)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(bench_dates, bench_nav, label=f'HS300 Buy-Hold (+{b["buy_hold_return_pct"]:.2f}%)',
            color='gray', linewidth=1.2, alpha=0.7)
    ax.plot(strat_dates, strat_nav, label=f'Daily Factor Reselect (+{p["final_return_pct"]:.2f}%)',
            color='crimson', linewidth=2.5, marker='o', markersize=3)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV (Start = 1.00)', fontsize=12)
    ax.set_title('Daily Factor Reselect vs HS300 — 2010-01-04 to 2025-12-31',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'wrote {out_png}')

    # CSV
    out_csv.write_text('date,strategy_nav,benchmark_nav\n')
    bench_map = dict(b['nav_curve'])
    for date_str, nav, _, _, _ in p['nav_curve']:
        bench_v = bench_map.get(date_str, '')
        out_csv.write_text(f'{date_str},{nav},{bench_v}\n')
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
