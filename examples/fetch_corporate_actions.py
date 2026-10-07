"""Fetch Tushare dividend/split data for the full HS300 universe (354 stocks).

Output: evidence/v34_adj_20261007/corporate_actions.parquet
Columns: ts_code, ex_date, action ('split'|'dividend'), value
  - split:    value = 1 + stk_div  (share multiplier)
  - dividend: value = cash_div_tax (cash per pre-event share, after tax)

Events filtered: div_proc == '实施', ex_date in [2010-01-01, 2026-12-31].
"""
from __future__ import annotations
import pathlib
import time

import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
OUTDIR = AKQ / "evidence" / "v34_adj_20261007"
TOKEN = pathlib.Path("/media/felix/f/quant/aurumq-rl/.qbot_token").read_text().strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


universe = sorted(pl.read_parquet(V34, columns=["ts_code"])["ts_code"].unique().to_list())
log(f"universe: {len(universe)} stocks")

import tushare as ts
pro = ts.pro_api(TOKEN)

rows = []
errs = []
t0 = time.time()
for i, code in enumerate(universe, 1):
    for attempt in range(4):
        try:
            df = pro.dividend(ts_code=code)
            if df is not None and len(df):
                rows.append(df)
            break
        except Exception as e:
            msg = str(e)
            if attempt == 3:
                errs.append((code, msg[:100]))
            else:
                time.sleep(3 + attempt * 3)
    time.sleep(0.35)
    if i % 40 == 0:
        log(f"  {i}/{len(universe)} ({(time.time()-t0):.0f}s), rows={sum(len(r) for r in rows)}")

log(f"fetched {len(rows)} tables, errors={len(errs)}; elapsed {time.time()-t0:.0f}s")
for e in errs[:10]:
    log(f"  ERR {e[0]}: {e[1]}")
if not rows:
    raise SystemExit("no data fetched")

full = pd.concat(rows, ignore_index=True)
log(f"raw rows: {len(full)}")

# normalize
full["ex_date"] = pd.to_datetime(full["ex_date"], format="%Y%m%d", errors="coerce")
full = full[(full["div_proc"] == "实施") & full["ex_date"].notna()]
full = full[(full["ex_date"] >= "2010-01-01") & (full["ex_date"] <= "2026-12-31")]
log(f"after filter: {len(full)} rows")

events = []
for _, r in full.iterrows():
    code = r["ts_code"]
    exd = r["ex_date"].date()
    stk = float(r["stk_div"]) if pd.notna(r["stk_div"]) else 0.0
    cdt = float(r["cash_div_tax"]) if pd.notna(r["cash_div_tax"]) else 0.0
    cd = float(r["cash_div"]) if pd.notna(r["cash_div"]) else 0.0
    cash = cdt if cdt > 0 else cd
    if cash > 0:
        # dividend on PRE-event shares
        events.append({"ts_code": code, "ex_date": exd, "action": "dividend", "value": cash,
                       "seq": 0})
    if stk > 0:
        events.append({"ts_code": code, "ex_date": exd, "action": "split", "value": 1.0 + stk,
                       "seq": 1})

ev = pd.DataFrame(events).sort_values(["ts_code", "ex_date", "seq"]).reset_index(drop=True)
log(f"events total: {len(ev)}; split={int((ev['action']=='split').sum())}, dividend={int((ev['action']=='dividend').sum())}")

ev_pl = pl.from_pandas(ev[["ts_code", "ex_date", "action", "value", "seq"]])
ev_pl = ev_pl.with_columns(pl.col("ex_date").cast(pl.Date))
out = OUTDIR / "corporate_actions.parquet"
ev_pl.write_parquet(out)
log(f"saved {out} ({out.stat().st_size/1024:.0f} KB)")

# summary
summ = {
    "universe": len(universe),
    "stocks_with_events": int(ev["ts_code"].nunique()),
    "n_events": len(ev),
    "n_split": int((ev["action"] == "split").sum()),
    "n_dividend": int((ev["action"] == "dividend").sum()),
    "ex_date_min": str(ev["ex_date"].min()),
    "ex_date_max": str(ev["ex_date"].max()),
    "errors": [e[0] for e in errs],
}
import json
(OUTDIR / "corporate_actions_summary.json").write_text(json.dumps(summ, indent=2))
log(f"DONE: {summ}")
