"""V19 bear specialist 在 2022 sweep 参数 — 找让 2022 转正的 config.

Sweep grid:
- bear_lookback_sessions ∈ {20, 30, 45, 60}
- min_votes ∈ {2, 3, 4}
- min_factor_mean_return ∈ {0.0, 0.003, 0.005}

Total: 4*3*3 = 36 runs, 每个 ~5-10s = ~5 min.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

OUT_BASE = Path("evidence/v19_sweep_2022_20261006")
OUT_BASE.mkdir(parents=True, exist_ok=True)

SWEEP_GRID = list(product(
    [20, 30, 45, 60],   # bear_lookback_sessions
    [2, 3, 4],          # min_votes
    [0.0, 0.003, 0.005],  # min_factor_mean_return
))


def run_one(lookback: int, min_votes: int, min_ret: float) -> dict:
    job_id = f"lb{lookback}_mv{min_votes}_mr{int(min_ret*1000):03d}"
    cmd = [
        '/usr/bin/python3.12', '-u',
        'examples/bear_factor_return_vote_v19.py',
        '--start-date', '2022-01-01',
        '--end-date', '2023-01-01',
        '--job-id', job_id,
        '--out-base', str(OUT_BASE),
        '--bear-lookback-sessions', str(lookback),
        '--min-votes', str(min_votes),
        '--min-factor-mean-return', str(min_ret),
    ]
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
    env['PATH'] = '/usr/bin:/bin:/usr/local/bin'
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd='/media/felix/f/quant/akquant-factor-backtest', timeout=120)
    fp = OUT_BASE / job_id / 'result.json'
    if not fp.exists():
        return {"error": "no result"}
    obj = json.loads(fp.read_text())
    return {
        "ret": obj.get("final_return_pct", 0),
        "sharpe": obj.get("metrics", {}).get("sharpe_ratio", 0),
        "mdd": obj.get("metrics", {}).get("max_drawdown_pct", 0),
        "pf": obj.get("metrics", {}).get("profit_factor", 0),
        "win": obj.get("metrics", {}).get("win_rate", 0),
        "n_trades": obj.get("n_trades", 0),
    }


def main():
    print("=" * 80)
    print(f"V19 sweep in 2022: {len(SWEEP_GRID)} configs")
    print("=" * 80)
    summary = {}
    t0 = time.time()
    for i, (lb, mv, mr) in enumerate(SWEEP_GRID):
        print(f"\n[{i+1}/{len(SWEEP_GRID)}] lookback={lb}, min_votes={mv}, min_ret={mr}")
        t_run = time.time()
        try:
            m = run_one(lb, mv, mr)
            if 'error' in m:
                summary[(lb, mv, mr)] = m
                continue
            summary[(lb, mv, mr)] = m
            marker = "✅" if m['ret'] > 0 else "❌"
            print(f"  {marker} ret={m['ret']:+.2f}%, sharpe={m['sharpe']:.3f}, MDD={m['mdd']:.2f}%, PF={m['pf']:.3f}, win={m['win']:.1f}%, {m['n_trades']} trades ({time.time()-t_run:.1f}s)")
        except subprocess.TimeoutExpired:
            summary[(lb, mv, mr)] = {"error": "timeout"}
            print(f"  TIMEOUT")

    # Sort by ret
    print("\n" + "=" * 80)
    print("Sweep 汇总 (按 ret 排序)")
    print("=" * 80)
    rows = [(k, v) for k, v in summary.items() if 'ret' in v]
    rows.sort(key=lambda x: -x[1]['ret'])
    print(f"{'config':30s} {'ret':>8} {'sharpe':>7} {'MDD':>7} {'PF':>5} {'Win':>5} {'Trades':>7}")
    for (lb, mv, mr), m in rows:
        marker = "✅" if m['ret'] > 0 else "❌"
        print(f"  {marker} lb={lb} mv={mv} mr={mr:.3f}  {m['ret']:>+7.2f}% {m['sharpe']:>7.3f} {m['mdd']:>6.2f}% {m['pf']:>5.3f} {m['win']:>4.1f}% {m['n_trades']:>7}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    # Save
    serial = {f"lb{k[0]}_mv{k[1]}_mr{k[2]}": v for k, v in summary.items()}
    with open(OUT_BASE / 'sweep_summary.json', 'w') as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()