"""V19 bear specialist (best variant) 跨熊市时段跑。

合同 (best variant from earlier sweep):
- BEAR_LOOKBACK_SESSIONS=30
- MIN_VOTES=3
- MIN_FACTOR_MEAN_RETURN=0.005
- bear_router: idx_ret_60d <= -5% AND idx_ret_20d <= 0
- bear_rebalance_sessions=20
- T+1 raw open -> T+21 raw open, 0.5% round-trip cost
- top-10 factors, top-10 stocks per factor, 5-10 stocks per rebal

复用 bear_factor_return_vote_v19.py main() 逻辑。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PANEL = Path("/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
             "v10_2_mainwave_features_v2_talib_20260924_021633/"
             "wavehunter_mainwave_features_v2.parquet")
OUT_BASE = Path("evidence/v19_best_cross_bear_20261006")
OUT_BASE.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("2011-01-01", "2012-01-01", "2011_yin_die"),
    ("2015-06-01", "2016-06-01", "2015H2_gushai"),
    ("2018-01-01", "2019-01-01", "2018_quan_nian_xia_die"),
    ("2022-01-01", "2023-01-01", "2022_bear"),
]


def run_one(start: str, end: str, label: str) -> dict:
    job_id = f"v19best_{label}"
    cmd = [
        "/usr/bin/python3.12", "-u",
        "examples/bear_factor_return_vote_v19.py",
        "--start-date", start,
        "--end-date", end,
        "--job-id", job_id,
        "--out-base", str(OUT_BASE),
        "--bear-lookback-sessions", "30",
        "--min-votes", "3",
        "--min-factor-mean-return", "0.005",
    ]
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
    env['PATH'] = '/usr/bin:/bin:/usr/local/bin'
    print(f"[run] {label}: {start} ~ {end}")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd='/media/felix/f/quant/akquant-factor-backtest', timeout=300)
    out = r.stdout
    for line in out.split('\n'):
        if 'final_return_pct' in line or 'done' in line.lower() or 'done ' in line or 'metrics' in line.lower():
            print(f"  {line.strip()[:200]}")
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
    # Read result.json
    result_path = OUT_BASE / job_id / "result.json"
    if not result_path.exists():
        return {"error": "no result.json", "returncode": r.returncode}
    r_obj = json.loads(result_path.read_text())
    return {
        "ret": r_obj.get("final_return_pct", 0),
        "metrics": r_obj.get("metrics", {}),
        "n_trades": r_obj.get("n_trades", 0),
        "regime_counts": r_obj.get("regime_counts", {}),
        "n_bear_rebal_dates": r_obj.get("n_bear_rebalance_dates", 0),
        "n_pick_dates": r_obj.get("n_pick_dates", 0),
    }


def main():
    print("=" * 80)
    print("V19 bear specialist (best variant: lookback=30, MIN_VOTES=3, min_ret=0.005)")
    print("跨熊市时段验证")
    print("=" * 80)
    summary = {}
    for s, e, label in PERIODS:
        print(f"\n=== {label} ({s} ~ {e}) ===")
        try:
            m = run_one(s, e, label)
            if 'error' in m:
                summary[label] = m
                continue
            ret = m['ret']
            sharpe = m['metrics'].get('sharpe_ratio', 0)
            mdd = m['metrics'].get('max_drawdown_pct', 0)
            pf = m['metrics'].get('profit_factor', 0)
            win = m['metrics'].get('win_rate', 0)
            n_trades = m['n_trades']
            regime_counts = m['regime_counts']
            marker = "✅" if ret > 0 else "❌"
            print(f"  {marker} {label}: +{ret:.2f}% / sharpe {sharpe:.3f} / MDD -{mdd:.2f}% / PF {pf:.3f} / Win {win:.1f}% / {n_trades} trades")
            print(f"     regime_counts={regime_counts} / n_pick_dates={m['n_pick_dates']}")
            summary[label] = {
                "ret": ret, "sharpe": sharpe, "mdd": mdd, "pf": pf, "win": win,
                "n_trades": n_trades,
                "regime_counts": regime_counts,
            }
        except subprocess.TimeoutExpired:
            summary[label] = {"error": "timeout 300s"}
            print(f"  TIMEOUT")

    print("\n=== V19 best cross-bear 汇总 ===")
    baseline_v32 = {"2011_yin_die": -30.96, "2015H2_gushai": -25.14,
                    "2018_quan_nian_xia_die": -22.87, "2022_bear": -6.36}
    print(f"{'period':25} {'V32 voting':>12} {'V19 best':>10} {'delta':>10}")
    for label, m in summary.items():
        if 'error' in m: continue
        v32 = baseline_v32[label]
        delta = m['ret'] - v32
        marker = "✅" if m['ret'] > 0 else "❌"
        print(f"  {marker} {label:23} {v32:>+10.2f}% {m['ret']:>+8.2f}% {delta:>+8.2f}%")

    with open(OUT_BASE / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()