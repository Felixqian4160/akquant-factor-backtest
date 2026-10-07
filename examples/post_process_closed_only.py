"""Post-process v3 sweep_results.json: compute closed_only_return_pct and re-rank.

The v3 sweep uses akquant, whose orders fill T+1 (next bar). The strategy's
force-close on the last bar / on_stop NEVER fills (no next bar after backtest
end), so result.metrics always shows open_position_count=1 and a large
unrealized_pnl. The 'total_return_pct' is therefore biased upward by the
unrealized mark-to-market PnL.

To get a closed-only return (what production would actually realize), we
compute:
    closed_only_pnl      = total_pnl - unrealized_pnl
    closed_only_return   = closed_only_pnl / initial_market_value * 100
This is exactly what a true end-of-period close would realize.

This script does NOT re-run any backtest. It reads sweep_results.json (raw
field present), adds `closed_only_return_pct`, re-dedups, and writes a
new ranking suite into the SAME evidence dir.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

EVIDENCE_DIR = Path("/media/felix/f/quant/akquant-factor-backtest/evidence/sweep_v3_full_20260929_220211")
sweep_json = EVIDENCE_DIR / "sweep_results.json"
raw_data = json.loads(sweep_json.read_text())
raw_results = raw_data["raw"]

print(f"raw records: {len(raw_results)}")

# Compute closed_only_return_pct for every record.
for r in raw_results:
    if "error" in r:
        r["closed_only_return_pct"] = float("nan")
        continue
    total_pnl = r.get("total_pnl", 0)
    upnl = r.get("unrealized_pnl", 0)
    initial = r.get("initial_market_value", 1) or 1
    closed_only_pnl = total_pnl - upnl
    r["closed_only_return_pct"] = round(closed_only_pnl / initial * 100, 4)

ok = sum(1 for r in raw_results if "error" not in r)
err = sum(1 for r in raw_results if "error" in r)
print(f"  ok: {ok}, err: {err}")

# Sanity: how many are positive under closed-only?
positive = sum(1 for r in raw_results if "error" not in r and r.get("closed_only_return_pct", 0) > 0)
print(f"  closed_only positive: {positive}/{ok}")

# Dedup using closed_only_return_pct + key risk metrics.
fp_groups = defaultdict(list)
for r in raw_results:
    if "error" in r:
        continue
    key = (
        r["rebal_days"],
        round(r.get("closed_only_return_pct", 0), 4),
        round(r.get("sharpe_ratio", 0), 4),
        round(r.get("max_drawdown_pct", 0), 4),
        int(r.get("closed_trade_count", 0)),
    )
    fp_groups[key].append(r["factor"])

dedup = []
dropped = []
for key, fgroup in fp_groups.items():
    alpha_pref = [f for f in fgroup if f.startswith("alpha_alpha")]
    keep = alpha_pref[0] if alpha_pref else fgroup[0]
    r_match = next(
        (r for r in raw_results
         if "error" not in r and r["factor"] == keep and r["rebal_days"] == key[0]),
        None,
    )
    if r_match:
        dedup.append(r_match)
    if len(fgroup) > 1:
        dropped.append((keep, [f for f in fgroup if f != keep]))

print(f"dedup kept: {len(dedup)}, dropped groups: {len(dropped)}")

# Audit: how many still have open_position > 0 (akmeans metric invariants — akquant order T+1 issue).
still_open = sum(1 for r in dedup if r.get("open_position_count", 0) > 0)
print(f"audit: {still_open}/{len(dedup)} have open_position>0 (expected: all {len(dedup)}; akquant order T+1)")

# Write back augmented raw + dedup into same json.
out_json = {
    "raw": raw_results,
    "dedup": dedup,
    "dropped": dropped,
    "force_close_audit_note": (
        "All records show open_position_count=1 because akquant 0.3.64 orders "
        "fill T+1 (next bar); the strategy's force-close on the last bar and "
        "on_stop callback NEVER fill (no next bar). The 'closed_only_return_pct' "
        "field is the true closed-only return (= total_pnl - unrealized_pnl) / "
        "initial_market_value * 100, which is what production would realize at "
        "end-of-period close."
    ),
}
sweep_json.write_text(json.dumps(out_json, indent=2, default=str))
print(f"updated {sweep_json}")

# Re-write ranking CSVs using closed_only_return_pct as primary key.
ranking_keys = [
    ("closed_only_return_pct", "closed_only_return"),
    ("annualized_return", "annualized_return"),
    ("sharpe_ratio", "sharpe"),
    ("win_rate", "win_rate"),
]
top_n = 20
for key, label in ranking_keys:
    ranked = sorted(dedup, key=lambda r: r.get(key, float("-inf")), reverse=True)[:top_n]
    csv_path = EVIDENCE_DIR / f"top_by_{label}.csv"
    with csv_path.open("w") as fh:
        fh.write("rank,factor,rebal_days,closed_only_return_pct,total_return_pct,annualized_return,"
                 "sharpe_ratio,win_rate,max_drawdown_pct,closed_trade_count,"
                 "open_position_count,unrealized_pnl\n")
        for rank, r in enumerate(ranked, 1):
            row = [
                str(rank), r["factor"], str(r["rebal_days"]),
                f'{r.get("closed_only_return_pct", 0):.4f}',
                f'{r.get("total_return_pct", 0):.4f}',
                f'{r.get("annualized_return", 0):.6f}',
                f'{r.get("sharpe_ratio", 0):.6f}',
                f'{r.get("win_rate", 0):.4f}',
                f'{r.get("max_drawdown_pct", 0):.4f}',
                str(int(r.get("closed_trade_count", 0))),
                str(int(r.get("open_position_count", 0))),
                f'{r.get("unrealized_pnl", 0):.2f}',
            ]
            fh.write(",".join(row) + "\n")
    print(f"  wrote {csv_path.name}")

# Top by closed_only_return
top_csv = EVIDENCE_DIR / "top_by_closed_only_return.csv"
print()
print("=== Top 10 by closed_only_return_pct ===")
with top_csv.open() as fh:
    for i, line in enumerate(fh):
        if i == 0:
            print(line.rstrip())
        elif i <= 10:
            print(line.rstrip())
