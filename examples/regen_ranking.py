"""Regenerate ranking.md from an existing sweep_results.json.

Reads: evidence/sweep_*/sweep_results.json
Writes: same dir/ranking.md (overwrite)
"""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: regen_ranking.py <sweep_dir>")
    sys.exit(2)

sweep_dir = Path(sys.argv[1])
results = json.load(open(sweep_dir / "sweep_results.json"))
valid = [r for r in results if "error" not in r]
valid.sort(key=lambda r: r.get("sharpe_ratio", float("-inf")), reverse=True)
top_n = 20
top = valid[:top_n]

md = sweep_dir / "ranking.md"
with md.open("w") as fh:
    fh.write(f"# AKQuant factor sweep ranking (regenerated)\n\n")
    fh.write(f"- factors swept: {len(results)} (ok={len(valid)})\n")
    fh.write(f"- strategy: factor > 5d MA -> 99% long; else flat\n")
    fh.write(f"- commission: 25bps per side; slippage: 0\n\n")
    fh.write(f"## Top {top_n} by Sharpe\n\n")
    fh.write("| rank | factor | sharpe | sortino | PF | win% | return% | MDD% | trades |\n")
    fh.write("|---:|:---|---:|---:|---:|---:|---:|---:|---:|\n")
    for rank, r in enumerate(top, 1):
        sharpe = r.get("sharpe_ratio", 0)
        sortino = r.get("sortino_ratio", 0)
        pf = r.get("profit_factor", 0)
        wr = r.get("win_rate", 0)
        ret = r.get("total_return_pct", 0)
        mdd = r.get("max_drawdown_pct", 0)
        trades = int(r.get("closed_trade_count", 0))
        fh.write(
            f"| {rank} | `{r['factor']}` | {sharpe:+.3f} | {sortino:+.3f} | "
            f"{pf:.3f} | {wr:.2f} | {ret:+.2f} | {mdd:.2f} | {trades} |\n"
        )

    # Bottom 20 (worst) — sanity check
    bottom = valid[-20:]
    fh.write(f"\n## Bottom 20 by Sharpe (sanity check)\n\n")
    fh.write("| rank | factor | sharpe | sortino | PF | win% | return% | MDD% | trades |\n")
    fh.write("|---:|:---|---:|---:|---:|---:|---:|---:|---:|\n")
    n = len(valid)
    for rank, r in enumerate(bottom, n - 19):
        sharpe = r.get("sharpe_ratio", 0)
        sortino = r.get("sortino_ratio", 0)
        pf = r.get("profit_factor", 0)
        wr = r.get("win_rate", 0)
        ret = r.get("total_return_pct", 0)
        mdd = r.get("max_drawdown_pct", 0)
        trades = int(r.get("closed_trade_count", 0))
        fh.write(
            f"| {rank} | `{r['factor']}` | {sharpe:+.3f} | {sortino:+.3f} | "
            f"{pf:.3f} | {wr:.2f} | {ret:+.2f} | {mdd:.2f} | {trades} |\n"
        )

    # Quick distribution stats
    sharpes = [r.get("sharpe_ratio", 0) for r in valid]
    returns = [r.get("total_return_pct", 0) for r in valid]
    pfs = [r.get("profit_factor", 0) for r in valid]
    fh.write(f"\n## Distribution (over {len(valid)} valid factors)\n\n")
    fh.write(f"- Sharpe: min={min(sharpes):+.3f}  median={sorted(sharpes)[len(sharpes)//2]:+.3f}  max={max(sharpes):+.3f}\n")
    fh.write(f"- Return%: min={min(returns):+.2f}  median={sorted(returns)[len(returns)//2]:+.2f}  max={max(returns):+.2f}\n")
    fh.write(f"- PF: min={min(pfs):.3f}  median={sorted(pfs)[len(pfs)//2]:.3f}  max={max(pfs):.3f}\n")
    fh.write(f"- factors with positive total_return_pct: {sum(1 for r in returns if r > 0)}/{len(returns)}\n")

print(f"regenerated: {md}")
