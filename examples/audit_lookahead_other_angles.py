"""Lookahead audit — other angles (2026-10-10): capacity, survivorship, router lag, costs.

Angles (for the v34-lb20 factor-return voting audit):
  A. Tradeability / capacity — order notional vs stock ADTV, by year (archived + causal runs)
  B. Universe / survivorship — per-symbol first/last trade dates; exits vs entries
  C. Router confirmation lag — ZigZag pivots vs when they could be CONFIRMED in real time
  D. Cost completeness — A-share stamp duty missing from the runner; quantify drag

Outputs: evidence/audit_lookahead_20261010/other_angles.json (+ printed tables)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
ARCH_TRADES = AKQ / "evidence" / "v34_lb20_rerun_20261010" / "sims" / "V14_0" / "trades.csv"
CAUS_TRADES = AKQ / "evidence" / "audit_lookahead_20261010" / "sims" / "V14_0" / "trades.csv"
OUT = AKQ / "evidence" / "audit_lookahead_20261010"
PIVOTS = pathlib.Path("/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
                      "hs300_index_pivots_clean_20260919/hs300_index_pivots.json")

results: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def ns_to_cst_date(series):
    return (pd.to_datetime(series.astype("int64"), unit="ns", utc=True)
            .dt.tz_convert("Asia/Shanghai").dt.date)


# ─────────────── shared: panel daily turnover ───────────────
log("loading panel turnover columns...")
pan = (
    pl.read_parquet(PANEL, columns=["ts_code", "trade_date", "amount", "vol", "close"])
    .with_columns(pl.col("trade_date").dt.date().alias("d"))
    .select(["ts_code", "d", "amount", "vol", "close"])
    .to_pandas()
)
pan["adtv_yuan"] = pan["amount"] * 1000.0          # v34 amount unit = 千元
ratio = (pan["adtv_yuan"] / (pan["vol"] * 100.0 * pan["close"]))
log(f"  unit sanity: median(amount*1000 / vol*100*close) = {ratio.median():.4f}")
pan["d"] = pd.to_datetime(pan["d"]).dt.date
pan = pan[["ts_code", "d", "adtv_yuan"]]

# ─────────────── A. capacity ───────────────
log("\n=== A. capacity / tradeability (order notional vs ADTV) ===")
cap = {}
for name, path in (("archived", ARCH_TRADES), ("causal", CAUS_TRADES)):
    tr = pd.read_csv(path)
    tr["d"] = ns_to_cst_date(tr["entry_time"])
    tr["notional"] = tr["quantity"] * tr["entry_price"]
    j = tr.merge(pan, left_on=["symbol", "d"], right_on=["ts_code", "d"], how="left")
    j["part"] = j["notional"] / j["adtv_yuan"]
    j["year"] = pd.to_datetime(j["d"]).dt.year
    per_year = (j.groupby("year")["part"]
                .agg(["median", lambda s: s.quantile(0.9), "count"])
                .rename(columns={"<lambda_0>": "p90"}))
    cap[name] = {
        "n_trades": int(len(j)),
        "median_part": float(j["part"].median()),
        "p90_part": float(j["part"].quantile(0.9)),
        "frac_gt_10pct_adv": float((j["part"] > 0.10).mean()),
        "frac_gt_100pct_adv": float((j["part"] > 1.00).mean()),
        "per_year_median": {int(y): float(v) for y, v in per_year["median"].items()},
        "per_year_p90": {int(y): float(v) for y, v in per_year["p90"].items()},
    }
    log(f"[{name}] median participation={cap[name]['median_part']:.2%}, "
        f"p90={cap[name]['p90_part']:.2%}, >10%ADV share={cap[name]['frac_gt_10pct_adv']:.1%}, "
        f">100%ADV share={cap[name]['frac_gt_100pct_adv']:.2%}")
    yrs = sorted(cap[name]["per_year_median"])
    log("  per-year median participation: " +
        " ".join(f"{y}:{cap[name]['per_year_median'][y]:.1%}" for y in yrs[::4]))
results["capacity"] = cap

# ─────────────── B. survivorship ───────────────
log("\n=== B. universe / survivorship ===")
sp = (
    pl.read_parquet(PANEL, columns=["ts_code", "trade_date"])
    .group_by("ts_code")
    .agg(pl.col("trade_date").min().alias("first"), pl.col("trade_date").max().alias("last"),
         pl.len().alias("n"))
)
sp = sp.with_columns(pl.col("first").dt.date(), pl.col("last").dt.date())
n_total = sp.height
n_exit = sp.filter(pl.col("last") < pl.date(2026, 8, 1)).height
exits = sp.filter(pl.col("last") < pl.date(2026, 8, 1)).sort("last")
log(f"  symbols total={n_total}; with last<2026-08 (exits/delisted/removed)={n_exit}")
if n_exit:
    log("  exits: " + ", ".join(f"{r['ts_code']}@{r['last']}" for r in exits.head(15).iter_rows(named=True)))
n_late = sp.filter(pl.col("first") > pl.date(2010, 7, 1)).height
log(f"  symbols entering after 2010-07 (later IPOs)={n_late}; in 2010: {n_total - n_late}")
results["survivorship"] = {"symbols_total": n_total, "symbols_exited": n_exit,
                           "exits_sample": [f"{r['ts_code']}@{r['last']}" for r in exits.head(20).iter_rows(named=True)],
                           "symbols_late_entry": n_late}

# ─────────────── C. router confirmation lag ───────────────
log("\n=== C. router confirmation lag (ZigZag pivots) ===")
piv = json.loads(PIVOTS.read_text())["pivots"]
ix = (
    pl.read_parquet(PANEL, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
    .filter(pl.col("idx_close").is_not_null())
    .with_columns(pl.col("trade_date").dt.date().alias("d"))
    .select(["d", "idx_close"]).to_pandas()
)
ix["d"] = pd.to_datetime(ix["d"]).dt.date
dates = list(ix["d"]); idxv = ix["idx_close"].to_numpy(dtype=float)
dpos = {d: i for i, d in enumerate(dates)}
TH = 0.10
lag_rows = []
for p in piv:
    pd_ = pd.Timestamp(p["date"]).date()
    if pd_ not in dpos:
        continue
    i0 = dpos[pd_]
    px0 = p["price"]
    if p["type"] == "peak":
        cond = idxv[i0:] <= px0 * (1 - TH)
    else:
        cond = idxv[i0:] >= px0 * (1 + TH)
    hits = np.flatnonzero(cond)
    if len(hits) == 0:
        lag_rows.append({"date": p["date"], "type": p["type"], "lag_days": None})
        continue
    i1 = i0 + int(hits[0])
    lag_rows.append({"date": p["date"], "type": p["type"], "lag_days": int(i1 - i0)})
lags = [r["lag_days"] for r in lag_rows if r["lag_days"] is not None]
if lags:
    log(f"  pivots={len(lag_rows)}, median lag={int(np.median(lags))} td, max={max(lags)} td")
else:
    log(f"  pivots={len(lag_rows)}, no confirmation found")
for r in sorted([r for r in lag_rows if r["lag_days"]], key=lambda x: -(x["lag_days"] or 0))[:6]:
    log(f"    {r['date']} {r['type']:6s} confirm lag={r['lag_days']} td")
# affected rebalance dates (standard 10-offset grid): date inside [pivot, confirmation)
all_dates = [str(d)[:10] for d in pl.scan_parquet(PANEL).select(pl.col("trade_date")).unique()
             .sort("trade_date").collect()["trade_date"].to_list()]
reb = set()
for off in range(0, 20, 2):
    reb.update(all_dates[off::20])
reb = {d for d in reb if "2010-01-01" <= d <= "2025-12-31"}
affected = []
for r in lag_rows:
    if not r["lag_days"]:
        continue
    pd_ = pd.Timestamp(r["date"]).date()
    i0 = dpos[pd_]
    window = {str(x) for x in dates[i0:i0 + r["lag_days"]]}
    hit = reb & window
    affected.extend(sorted(hit))
affected = sorted(set(affected))
log(f"  rebalance dates inside unconfirmed-pivot windows (10-offset grid): {len(affected)}")
results["router"] = {"n_pivots": len(lag_rows), "median_lag_td": int(np.median(lags)),
                     "max_lag_td": int(max(lags)), "lag_rows": lag_rows,
                     "affected_rebalances": affected, "n_affected": len(affected)}

# ─────────────── D. cost completeness (stamp duty) ───────────────
log("\n=== D. stamp duty (missing from runner) ===")
stamp = {}
for name, path in (("archived", ARCH_TRADES), ("causal", CAUS_TRADES)):
    tr = pd.read_csv(path)
    tr["exit_d"] = ns_to_cst_date(tr["exit_time"])
    tr["exit_notional"] = tr["quantity"] * tr["exit_price"]
    tr["rate"] = np.where(pd.to_datetime(tr["exit_d"]) < pd.Timestamp("2023-08-28"), 0.001, 0.0005)
    tr["stamp"] = tr["exit_notional"] * tr["rate"]
    total_stamp = float(tr["stamp"].sum())
    avg_nav = float(tr["entry_portfolio_value"].mean())
    years = (pd.to_datetime(tr["exit_d"]).max() - pd.to_datetime(tr["exit_d"]).min()).days / 365.25
    drag = total_stamp / avg_nav / years
    stamp[name] = {"total_stamp_yuan": total_stamp, "avg_nav": avg_nav, "years": years,
                   "annual_drag_pct_of_nav": drag}
    log(f"[{name}] total stamp=¥{total_stamp/1e6:.1f}M, avg NAV=¥{avg_nav/1e6:.0f}M, "
        f"annual drag={drag:.2%}/yr")
results["stamp"] = stamp

(OUT / "other_angles.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
log(f"\nsaved {OUT/'other_angles.json'}")
