"""Adjustment Impact Study — Layer A: factor VALUE impact on cross-sectional voting.

For representative factors with formulas verified against the panel:
  1. talib_MOM  (close - close[10], diff-type)
  2. talib_ROC  (= (close/close[n]-1)*100, ratio-type — test n)
  3. gtja_gtja_002 (intraday shape — INVARIANT control)

For each factor, compute raw version vs adjusted version, then per rebal date:
  - Spearman rank correlation across stocks
  - top-10 overlap (Jaccard count)
"""
import json, pathlib, time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
ADJ = ROOT / "evidence" / "audit_v33_20261007" / "adj_study" / "adj_prices_354.parquet"
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
OUT = ROOT / "evidence" / "audit_v33_20261007" / "adj_study"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── load ──
prices = pl.read_parquet(ADJ).with_columns(pl.col("trade_date").cast(pl.Datetime("ms"))).sort(["ts_code", "trade_date"])
panel_f = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "talib_MOM", "talib_ROC",
                                          "gtja_gtja_002", "high", "low", "close", "open"])
panel_f = panel_f.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
log(f"prices: {prices.height:,}, panel factors: {panel_f.height:,}")

# ── Step 0: verify ROC period from panel ──
log("=== verify talib_ROC period ===")
chk = prices.join(panel_f.select(["trade_date", "ts_code", "talib_ROC"]), on=["trade_date", "ts_code"], how="inner")
for n in [10, 12, 20]:
    cand = chk.with_columns(((pl.col("close") / pl.col("close").shift(n).over("ts_code") - 1.0) * 100).alias("_cand"))
    valid = cand.filter(pl.col("talib_ROC").is_not_null() & pl.col("_cand").is_not_null())
    diff = (valid["talib_ROC"] - valid["_cand"]).abs().max()
    log(f"  ROC n={n}: max_diff={diff:.6f}")

# ── Step 1: compute raw & adjusted versions ──
log("=== compute factor versions ===")
N_MOM = 10  # verified
N_ROC = 10  # will confirm from above; use best
# find best n for ROC
best_n, best_d = None, 1e9
for n in [10, 12, 20]:
    cand = chk.with_columns(((pl.col("close") / pl.col("close").shift(n).over("ts_code") - 1.0) * 100).alias("_cand"))
    valid = cand.filter(pl.col("talib_ROC").is_not_null() & pl.col("_cand").is_not_null())
    d = (valid["talib_ROC"] - valid["_cand"]).abs().max()
    if d < best_d: best_n, best_d = n, d
N_ROC = best_n
log(f"  ROC period = {N_ROC} (max_diff {best_d:.6f})")

f = prices.with_columns([
    (pl.col("close") - pl.col("close").shift(N_MOM).over("ts_code")).alias("mom_raw"),
    (pl.col("adj_close") - pl.col("adj_close").shift(N_MOM).over("ts_code")).alias("mom_adj"),
    ((pl.col("close") / pl.col("close").shift(N_ROC).over("ts_code") - 1.0) * 100).alias("roc_raw"),
    ((pl.col("adj_close") / pl.col("adj_close").shift(N_ROC).over("ts_code") - 1.0) * 100).alias("roc_adj"),
]).join(
    panel_f.select(["trade_date", "ts_code", "gtja_gtja_002", "high", "low"]),
    on=["trade_date", "ts_code"], how="inner"
).with_columns(
    # gtja_002 on adjusted OHLC: since adj factor is constant within a day, same-day shape
    # is invariant; compute both from panel raw as control check
    pl.col("gtja_gtja_002").alias("gtja_raw"),
)

# window + rebal dates
f = f.filter((pl.col("trade_date") >= pl.datetime(2010, 1, 1)) &
             (pl.col("trade_date") <= pl.datetime(2025, 12, 31)))

all_dates = f["trade_date"].unique().sort().to_list()
rebal_dates = all_dates[0::20]
log(f"dates: {len(all_dates)}, rebal dates: {len(rebal_dates)}")

# ── Step 2: per-date comparison ──
log("=== per-date rank comparison ===")
from scipy.stats import spearmanr
results = {"mom": [], "roc": [], "gtja": []}
t0 = time.time()

def top10_overlap(a, b):
    da = pd.Series(a).dropna()
    db = pd.Series(b).dropna()
    common = da.index.intersection(db.index)
    if len(common) < 20:
        return None, None
    da, db = da[common], db[common]
    # spearman
    rho = spearmanr(da.values, db.values).statistic
    # top-10 overlap
    ta = set(da.nlargest(10).index)
    tb = set(db.nlargest(10).index)
    return rho, len(ta & tb)

for i, d in enumerate(rebal_dates, 1):
    day = f.filter(pl.col("trade_date") == d).to_pandas().set_index("ts_code")
    day = day.replace([np.inf, -np.inf], np.nan)
    for key, cola, colb in [("mom", "mom_raw", "mom_adj"), ("roc", "roc_raw", "roc_adj")]:
        rho, ov = top10_overlap(day[cola], day[colb])
        if rho is not None:
            results[key].append({"date": str(d)[:10], "spearman": rho, "top10_overlap": ov})
    # gtja control: same column compared to itself == 1.0 sanity; use it as invariance demo vs a tiny pert
    # (cannot build adj version w/o OHLC adj pre-factor; panel same-day shape is invariant by theory)
    if i % 40 == 0:
        log(f"  {i}/{len(rebal_dates)} dates, {time.time()-t0:.0f}s")

# ── Step 3: aggregate ──
summary = {}
for key, label in [("mom", "talib_MOM (close-close[10], diff-type)"),
                   ("roc", "talib_ROC ((close/close[10]-1)*100, ratio-type)")]:
    arr = results[key]
    if not arr:
        continue
    rhos = np.array([x["spearman"] for x in arr])
    ovs = np.array([x["top10_overlap"] for x in arr])
    s = {
        "label": label,
        "n_dates": len(arr),
        "spearman_mean": float(rhos.mean()),
        "spearman_median": float(np.median(rhos)),
        "spearman_p05": float(np.percentile(rhos, 5)),
        "top10_overlap_mean": float(ovs.mean()),
        "top10_overlap_median": float(np.median(ovs)),
        "pct_dates_overlap_lt_8": float((ovs < 8).mean() * 100),
        "pct_dates_overlap_eq_10": float((ovs == 10).mean() * 100),
    }
    summary[key] = s
    log(f"\n{label}:")
    log(f"  Spearman raw-vs-adj: mean={s['spearman_mean']:.4f}, median={s['spearman_median']:.4f}, p05={s['spearman_p05']:.4f}")
    log(f"  top-10 overlap: mean={s['top10_overlap_mean']:.2f}/10, median={s['top10_overlap_median']:.0f}/10")
    log(f"  dates with overlap<8/10: {s['pct_dates_overlap_lt_8']:.1f}%")
    log(f"  dates with perfect overlap: {s['pct_dates_overlap_eq_10']:.1f}%")

(OUT / "impact_A_factors.json").write_text(json.dumps(summary, indent=2, default=str))
pd.DataFrame(results["mom"]).to_csv(OUT / "mom_rank_comparison.csv", index=False)
pd.DataFrame(results["roc"]).to_csv(OUT / "roc_rank_comparison.csv", index=False)
log("\nsaved impact_A_factors.json + per-date CSVs")
log("DONE")
