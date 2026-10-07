"""Comprehensive 10-seed audit pipeline for ZigZag Simple IC strategy.

Reads evidence/sweep/zigzag_simple_ic_20261006/akquant/V14_{offset}/
ledger_audit.json + trades.csv + orders.csv + nav.csv + result.json.

Outputs:
  evidence/sweep/zigzag_simple_ic_20261006/audit/{offset}/...
  evidence/sweep/zigzag_simple_ic_20261006/audit/cross_seed_summary.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/cross_seed_summary.md
  evidence/sweep/zigzag_simple_ic_20261006/audit/period_split.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/yearly_breakdown.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/regime_purity.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/factor_rotation.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/holdings_overlap.csv
  evidence/sweep/zigzag_simple_ic_20261006/audit/trade_attribution.csv
"""
from __future__ import annotations

import csv
import json
import pathlib
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
SWEEP_DIR = ROOT / "evidence" / "sweep" / "zigzag_simple_ic_20261006"
AKQUANT_DIR = SWEEP_DIR / "akquant"
PICKS_DIR = SWEEP_DIR  # contains V14_{offset}/picks.json, picks_meta.json
AUDIT_DIR = SWEEP_DIR / "audit"
ROUTER_MAP_PATH = ROOT / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
PANEL_PATH = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
PANEL_YEARS = 15.99


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def per_offset_audit(off: int, audit_root: pathlib.Path) -> dict:
    """Pull all per-offset artifacts and aggregate."""
    src = AKQUANT_DIR / f"V14_{off}"
    obj = json.loads((src / "result.json").read_text())
    m = obj["metrics"]
    audit = obj["audit"]
    trades = read_csv(src / "trades.csv")
    orders = read_csv(src / "orders.csv")
    nav = read_csv(src / "nav.csv")
    picks = json.loads((PICKS_DIR / f"V14_{off}" / "picks.json").read_text())
    meta = json.loads((PICKS_DIR / f"V14_{off}" / "picks_meta.json").read_text())

    summary = {
        "offset": off,
        "total_return_pct": m["total_return_pct"],
        "annualized_return_pct": m["annualized_return"] * 100,
        "sharpe": m["sharpe_ratio"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "profit_factor": m["profit_factor"],
        "win_rate": m["win_rate"],
        "closed_trade_count": int(m["closed_trade_count"]),
        "open_position_count": int(m["open_position_count"]),
        "avg_trade_bars": round(m["avg_trade_bars"], 2),
        "total_commission": m["total_commission"],
        "n_rejected": audit["summary"]["n_rejected"],
        "n_picks_dates": len(picks),
        "n_bull_dates": sum(1 for v in meta.values() if v["regime"] == "bull_neutral"),
        "n_bear_dates": sum(1 for v in meta.values() if v["regime"] == "bear"),
    }

    # Save per-offset audit dir
    per_off_dir = audit_root / f"V14_{off}"
    per_off_dir.mkdir(parents=True, exist_ok=True)
    (per_off_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, trades, orders, nav, picks, meta


def period_split_metrics(nav_dates: list[str], nav_values: list[float],
                         splits: list[tuple[str, str]]) -> list[dict]:
    """Compute period-segment returns from a NAV series."""
    out = []
    for label, start, end in splits:
        s_idx = None
        e_idx = None
        for i, d in enumerate(nav_dates):
            if s_idx is None and d >= start:
                s_idx = i
            if d <= end:
                e_idx = i
        if s_idx is None or e_idx is None or e_idx <= s_idx:
            out.append({"period": label, "start": start, "end": end,
                        "start_value": None, "end_value": None, "ret_pct": None})
            continue
        v0 = nav_values[s_idx]
        v1 = nav_values[e_idx]
        out.append({"period": label, "start": start, "end": end,
                    "start_value": v0, "end_value": v1,
                    "ret_pct": round((v1 / v0 - 1) * 100, 3)})
    return out


def yearly_breakdown(nav_dates: list[str], nav_values: list[float]) -> list[dict]:
    """Per-year start/end NAV and return."""
    by_year: dict[int, tuple[int, int]] = {}
    for i, d in enumerate(nav_dates):
        y = int(d[:4])
        if y not in by_year:
            by_year[y] = (i, i)
        else:
            lo, _ = by_year[y]
            by_year[y] = (lo, i)
    rows = []
    for y in sorted(by_year.keys()):
        s_idx, e_idx = by_year[y]
        v0 = nav_values[s_idx]
        v1 = nav_values[e_idx]
        rows.append({"year": y, "start_value": v0, "end_value": v1,
                     "ret_pct": round((v1 / v0 - 1) * 100, 3)})
    return rows


def regime_purity(meta: dict, router: dict[str, str]) -> dict:
    """Compare meta regime per rebal date vs router ground truth."""
    correct = 0
    total = 0
    for d, v in meta.items():
        truth = router.get(d, "bull_neutral")
        pred = v["regime"]
        total += 1
        if truth == pred:
            correct += 1
    return {"n_rebal_dates": total, "agreement": correct,
            "agreement_pct": round(correct / max(total, 1) * 100, 2)}


def factor_rotation(meta: dict) -> list[dict]:
    """Per-bear-factor frequency across offsets."""
    factor_rebal_dates: dict[str, set[str]] = defaultdict(set)
    for d, v in meta.items():
        if v["regime"] == "bear":
            sel_meta = v  # contains n_active_factors
    # No per-bear-factor list in meta; use n_active_factors proxy.
    rows = []
    by_date = sorted(meta.keys())
    for d in by_date:
        v = meta[d]
        rows.append({"rebal_date": d, "regime": v["regime"],
                    "n_active_factors": v["n_active_factors"],
                    "n_picks": v["n_picks"],
                    "max_votes": v["max_votes"],
                    "n_at_or_above_threshold": v["n_at_or_above_threshold"]})
    return rows


def holdings_overlap(picks: dict[str, dict[str, int]]) -> list[dict]:
    """Jaccard overlap between consecutive rebal baskets."""
    dates = sorted(picks.keys())
    out = []
    for i in range(1, len(dates)):
        s1 = set(picks[dates[i-1]].keys())
        s2 = set(picks[dates[i]].keys())
        if not s1 and not s2:
            j = None
        else:
            j = round(len(s1 & s2) / max(len(s1 | s2), 1), 3)
        out.append({"date_prev": dates[i-1], "date_curr": dates[i],
                    "n_prev": len(s1), "n_curr": len(s2),
                    "n_common": len(s1 & s2), "jaccard": j})
    return out


def trade_attribution(trades: list[dict], router: dict[str, str]) -> dict:
    """Trade-by-trade PnL attribution to regime at entry."""
    by_regime: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "comm": 0.0, "wins": 0})
    by_year: dict[int, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "comm": 0.0, "wins": 0})
    for t in trades:
        et = datetime.fromtimestamp(int(t["entry_time"]) / 1e9).date().isoformat()
        regime = router.get(et, "bull_neutral")
        pnl = float(t["net_pnl"])
        comm = float(t["commission"])
        year = int(et[:4])
        by_regime[regime]["n"] += 1
        by_regime[regime]["pnl"] += pnl
        by_regime[regime]["comm"] += comm
        if pnl > 0:
            by_regime[regime]["wins"] += 1
        by_year[year]["n"] += 1
        by_year[year]["pnl"] += pnl
        by_year[year]["comm"] += comm
        if pnl > 0:
            by_year[year]["wins"] += 1
    return {
        "by_regime": {r: dict(v) for r, v in by_regime.items()},
        "by_year": {str(y): dict(v) for y, v in sorted(by_year.items())},
    }


def main():
    audit_root = AUDIT_DIR
    audit_root.mkdir(parents=True, exist_ok=True)
    router = json.loads(ROUTER_MAP_PATH.read_text())

    summary_rows = []
    all_trades = {}
    all_orders = {}
    all_nav = {}
    all_picks = {}
    all_meta = {}
    all_attrib = {}

    for off in OFFSETS:
        summary, trades, orders, nav, picks, meta = per_offset_audit(off, audit_root)
        summary_rows.append(summary)
        all_trades[off] = trades
        all_orders[off] = orders
        all_nav[off] = nav
        all_picks[off] = picks
        all_meta[off] = meta
        all_attrib[off] = trade_attribution(trades, router)

    # === Cross-seed summary CSV ===
    keys = ["offset", "total_return_pct", "annualized_return_pct", "sharpe",
            "max_drawdown_pct", "profit_factor", "win_rate",
            "closed_trade_count", "open_position_count", "avg_trade_bars",
            "total_commission", "n_rejected", "n_picks_dates",
            "n_bull_dates", "n_bear_dates"]
    with (audit_root / "cross_seed_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in keys})

    # === Period split metrics ===
    splits = [
        ("2010-2015", "2010-01-04", "2015-12-31"),
        ("2015-2020", "2016-01-01", "2020-12-31"),
        ("2020-2025", "2021-01-01", "2025-12-31"),
        ("2010-2025_OOS_2023_2025", "2010-01-04", "2022-12-31"),
        ("OOS_2023_2025", "2023-01-01", "2025-12-31"),
    ]
    period_rows = []
    for off in OFFSETS:
        nav_rows = all_nav[off]
        nav_dates = [r["date"] for r in nav_rows]
        nav_values = [float(r["value"]) for r in nav_rows]
        for label, s, e in splits:
            for ps in period_split_metrics(nav_dates, nav_values, [(label, s, e)]):
                period_rows.append({"offset": off, **ps})
    pd.DataFrame(period_rows).to_csv(audit_root / "period_split.csv", index=False)

    # === Yearly breakdown ===
    yearly_rows = []
    for off in OFFSETS:
        nav_rows = all_nav[off]
        nav_dates = [r["date"] for r in nav_rows]
        nav_values = [float(r["value"]) for r in nav_rows]
        for y in yearly_breakdown(nav_dates, nav_values):
            yearly_rows.append({"offset": off, **y})
    pd.DataFrame(yearly_rows).to_csv(audit_root / "yearly_breakdown.csv", index=False)

    # === Regime purity per offset ===
    purity_rows = []
    for off in OFFSETS:
        p = regime_purity(all_meta[off], router)
        purity_rows.append({"offset": off, **p})
    pd.DataFrame(purity_rows).to_csv(audit_root / "regime_purity.csv", index=False)

    # === Factor rotation proxy (per-rebal-date picks metadata) ===
    rot_rows = []
    for off in OFFSETS:
        for r in factor_rotation(all_meta[off]):
            rot_rows.append({"offset": off, **r})
    pd.DataFrame(rot_rows).to_csv(audit_root / "factor_rotation.csv", index=False)

    # === Holdings overlap (Jaccard) ===
    overlap_rows = []
    for off in OFFSETS:
        for r in holdings_overlap(all_picks[off]):
            overlap_rows.append({"offset": off, **r})
    pd.DataFrame(overlap_rows).to_csv(audit_root / "holdings_overlap.csv", index=False)

    # === Trade attribution per regime / per year ===
    attrib_rows_byreg = []
    for off in OFFSETS:
        for regime, v in all_attrib[off]["by_regime"].items():
            attrib_rows_byreg.append({"offset": off, "regime": regime,
                                     "n": v["n"], "pnl": round(v["pnl"], 2),
                                     "comm": round(v["comm"], 2),
                                     "wins": v["wins"]})
    pd.DataFrame(attrib_rows_byreg).to_csv(audit_root / "trade_attribution_by_regime.csv", index=False)

    attrib_rows_byyear = []
    for off in OFFSETS:
        for year, v in all_attrib[off]["by_year"].items():
            attrib_rows_byyear.append({"offset": off, "year": year,
                                       "n": v["n"], "pnl": round(v["pnl"], 2),
                                       "comm": round(v["comm"], 2),
                                       "wins": v["wins"]})
    pd.DataFrame(attrib_rows_byyear).to_csv(audit_root / "trade_attribution_by_year.csv", index=False)

    # === Aggregate "trade ledger" (first 100 + last 100 of V14_0) ===
    sample_trades = all_trades[OFFSETS[0]]
    (audit_root / "trade_ledger_sample_V14_0_first_50.csv").write_text(
        "\n".join([",".join(sample_trades[0].keys())] + [
            ",".join([str(r.get(k, "")) for k in sample_trades[0].keys()])
            for r in sample_trades[:50]
        ])
    )

    print(f"Wrote audit files to {audit_root}/")
    print(f"  cross_seed_summary.csv: {len(summary_rows)} rows")
    print(f"  period_split.csv: {len(period_rows)} rows")
    print(f"  yearly_breakdown.csv: {len(yearly_rows)} rows")
    print(f"  regime_purity.csv: {len(purity_rows)} rows")
    print(f"  factor_rotation.csv: {len(rot_rows)} rows")
    print(f"  holdings_overlap.csv: {len(overlap_rows)} rows")
    print(f"  trade_attribution_by_regime.csv: {len(attrib_rows_byreg)} rows")
    print(f"  trade_attribution_by_year.csv: {len(attrib_rows_byyear)} rows")
    return summary_rows


if __name__ == "__main__":
    rows = main()
    print()
    print("Mean annualized return: %.2f%%" % (sum(r["annualized_return_pct"] for r in rows) / len(rows)))
    print("Mean Sharpe: %.3f" % (sum(r["sharpe"] for r in rows) / len(rows)))
    print("Mean MDD: %.1f%%" % (sum(r["max_drawdown_pct"] for r in rows) / len(rows)))
    print("Mean PF: %.3f" % (sum(r["profit_factor"] for r in rows) / len(rows)))
    print("Positive offsets: %d/%d" % (sum(1 for r in rows if r["total_return_pct"] > 0), len(rows)))
