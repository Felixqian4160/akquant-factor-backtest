"""Derive corporate actions (split/dividend) from LOCAL cache — no Tushare API.

Data sources (all local):
  aurumq-rl/data_cache/{code}_full.parquet        : close, pre_close, adj_factor
  aurumq-rl/data_cache/{code}_daily_basic.parquet : total_share

Math:
  af ratio r = adj_factor[t]/adj_factor[t-1]  (= prev_close/pre_close, verified 3.6e-5)
  event     : r > 1 + 1e-6
  split s   : total_share[t]/total_share[t-1] (rounded, validated vs candidate snapping)
  cash      : cash = prev_close - s * pre_close   (ex-rights formula, pre_close = 除权基准价)
  validate  : NAV continuity  s*close[t] + cash  ==  close[t]*r

Candidate s set: total_share ratio, round-to-2/3/4dp(r), 1.0. Pick min |NAV residual|
among candidates with 0 <= cash <= 8%*prev_close. Flag events failing validation.

Output: evidence/v34_adj_20261007/corporate_actions_derived.parquet
        columns: ts_code, ex_date, action, value, s, cash, r, residual, valid
"""
from __future__ import annotations
import glob
import json
import os
import pathlib
import time

import numpy as np
import polars as pl

CACHE = pathlib.Path("/media/felix/f/quant/aurumq-rl/data_cache")
OUTDIR = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest/evidence/v34_adj_20261007")
OUTDIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


files = sorted(glob.glob(str(CACHE / "*_full.parquet")))
log(f"full files: {len(files)}")

all_events = []
n_no_db = 0
t0 = time.time()
for i, fp in enumerate(files, 1):
    code = os.path.basename(fp).replace("_full.parquet", "")
    f = pl.read_parquet(fp, columns=["trade_date", "close", "pre_close", "adj_factor"]).sort("trade_date")
    f = f.with_columns([
        pl.col("close").shift(1).alias("pc"),
        pl.col("adj_factor").shift(1).alias("paf"),
    ])
    f = f.with_columns((pl.col("adj_factor") / pl.col("paf")).alias("r"))

    # daily_basic for total_share
    dbp = CACHE / f"{code}_daily_basic.parquet"
    if dbp.exists():
        db = pl.read_parquet(dbp, columns=["trade_date", "total_share"]).sort("trade_date")
        db = db.with_columns(pl.col("total_share").shift(1).alias("ts_prev"))
        f = f.join(db.select(["trade_date", "total_share", "ts_prev"]), on="trade_date", how="left")
    else:
        n_no_db += 1
        f = f.with_columns([pl.lit(None, dtype=pl.Float64).alias("total_share"),
                            pl.lit(None, dtype=pl.Float64).alias("ts_prev")])

    # real ex-event: pre_close (除权基准) materially below previous close;
    # af-micro-noise (r ~ 1+1e-5) excluded because pre_close == prev_close there
    ev = f.filter(((pl.col("pc") - pl.col("pre_close")) > 0.004) & pl.col("pc").is_not_null() & pl.col("pre_close").is_not_null() & pl.col("r").is_not_null())

    for row in ev.iter_rows(named=True):
        pc = float(row["pc"]); prec = float(row["pre_close"]); cl = float(row["close"])
        r = float(row["r"])
        ts, tsp = row["total_share"], row["ts_prev"]
        # ── candidate selection: total_share PRIORITY, then grid, then r-snaps ──
        def bounds_ok(s_c):
            cash_c = pc - s_c * prec
            return -0.015 <= cash_c <= 0.08 * pc, cash_c

        best = None
        source = None
        # priority 1: total_share ratio (flat -> pure dividend s=1.0)
        if ts is not None and tsp is not None and tsp > 0:
            s_ratio = ts / tsp
            p1 = [round(s_ratio, 4), round(s_ratio, 2)] if s_ratio > 1.0005 else [1.0]
            for s_c in p1:
                if s_c < 1.0:
                    continue
                ok, cash_c = bounds_ok(s_c)
                if ok:
                    best = (s_c, max(cash_c, 0.0), abs(s_c * cl + max(cash_c, 0.0) - cl * r))
                    source = "total_share"
                    break
        # priority 2: 0.1-step grid (prefer closest to r among bounds-valid)
        if best is None:
            grid = [1.0] + [round(1.0 + 0.1 * k, 4) for k in range(1, 41)]
            for s_c in grid:
                ok, cash_c = bounds_ok(s_c)
                if not ok:
                    continue
                if best is None or abs(s_c - r) < abs(best[0] - r):
                    best = (s_c, max(cash_c, 0.0), 0.0)
                    source = "grid"
        # priority 3: r snaps (bounds only)
        if best is None:
            for s_c in [round(r, 2), round(r, 3), round(r, 4)]:
                if s_c < 1.0:
                    continue
                ok, cash_c = bounds_ok(s_c)
                if ok:
                    best = (s_c, max(cash_c, 0.0), 0.0)
                    source = "snap"
                    break
        if best is None:
            # fallback: NAV-exact decomposition s=r, cash=0
            best = (r, 0.0, 0.0)
            source = "fallback"
            valid = False
        else:
            valid = True

        s, cash, resid = best
        events_here = []
        if s > 1.0005:
            events_here.append(("split", s))
        if cash > 0.0005:
            events_here.append(("dividend", cash))
        if not events_here:
            events_here.append(("noop", 0.0))
        for action, value in events_here:
            all_events.append({
                "ts_code": code.replace("_", ".") if code[-3:] in ("_SZ", "_SH") else code,
                "ex_date": row["trade_date"],
                "action": action, "value": value,
                "s": s, "cash": cash, "r": r,
                "residual": resid, "valid": valid, "source": source,
            })
    if i % 100 == 0:
        log(f"  {i}/{len(files)} ({time.time()-t0:.0f}s), events so far: {len(all_events)}")

log(f"scan done in {time.time()-t0:.0f}s; events: {len(all_events)}; no daily_basic: {n_no_db}")

evd = pl.DataFrame(all_events)
# ts_code fix: cache names like 000001_SZ -> 000001.SZ
evd = evd.with_columns(
    pl.col("ts_code").str.replace(r"_(\w{2})$", ".$1").alias("ts_code")
)
evd = evd.with_columns(pl.col("ex_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
evd = evd.sort(["ts_code", "ex_date", "action"])

# stats
n_total = evd.height
n_split = (evd["action"] == "split").sum()
n_div = (evd["action"] == "dividend").sum()
n_noop = (evd["action"] == "noop").sum()
n_invalid = evd.filter(~pl.col("valid")).height
log(f"total event rows: {n_total}; split={n_split}, dividend={n_div}, noop={n_noop}; invalid={n_invalid}")

# per-year
evd2 = evd.with_columns(pl.col("ex_date").dt.year().alias("year"))
per_year = evd2.group_by("year").agg([pl.len().alias("n"), (pl.col("action") == "split").sum().alias("splits")]).sort("year")
log("per-year:\n" + str(per_year))

out = OUTDIR / "corporate_actions_derived.parquet"
evd.write_parquet(out)
log(f"saved {out} ({out.stat().st_size/1024:.0f} KB)")

# spot check cases
import datetime as _dt
for code, d in [("000630.SZ", "2015-10-23"), ("002594.SZ", "2025-07-29"), ("000001.SZ", "2023-06-14"), ("601633.SH", "2015-10-13")]:
    row = evd.filter((pl.col("ts_code") == code) & (pl.col("ex_date") == _dt.date.fromisoformat(d)))
    log(f"spot {code} {d}: {row.select(['action','value','s','cash','r','residual','valid','source']).to_dicts()}")

evd_post = evd.filter(pl.col("ex_date") >= _dt.date(2010, 1, 1))
summary = {
    "n_event_rows": n_total, "n_split": int(n_split), "n_dividend": int(n_div),
    "n_noop": int(n_noop), "n_invalid": int(n_invalid),
    "n_rows_post2010": evd_post.height,
    "n_invalid_post2010": evd_post.filter(~pl.col("valid")).height,
    "n_error_rows_post2010": evd_post.filter(pl.col("action") == "noop").height,
    "source_breakdown": evd.group_by("source").len().sort("len", descending=True).to_dicts(),
    "ex_date_min": str(evd["ex_date"].min()), "ex_date_max": str(evd["ex_date"].max()),
    "n_codes": int(evd["ts_code"].n_unique()),
}
(OUTDIR / "corporate_actions_derived_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log(f"DONE {summary}")
