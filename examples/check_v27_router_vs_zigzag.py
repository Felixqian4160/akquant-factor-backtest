"""Compare V27_BALANCE router with ZigZag ground truth (annual + per-day).

Outputs:
  evidence/v27_balance_router_check/v27_router_daily.csv (every trade day)
  evidence/v27_balance_router_check/yearly_summary.csv (per year)
  evidence/v27_balance_router_check/router_vs_truth.png (timeline chart)
"""
import polars as pl
import pandas as pd
import numpy as np
import json
import pathlib
import matplotlib.pyplot as plt
from datetime import date, timedelta

PANEL = pathlib.Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
PIVOTS = pathlib.Path("/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/hs300_index_pivots_clean_20260919/hs300_index_pivots.json")
OUT_DIR = pathlib.Path("evidence/v27_balance_router_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. ZigZag ground truth
data = json.loads(PIVOTS.read_text())
truth = {}
for leg in data["legs"]:
    if leg["duration_days"] < 100:
        continue
    start = date.fromisoformat(leg["start"])
    end = date.fromisoformat(leg["end"])
    kind = "bear" if leg["kind"] == "down" else "bull"
    d = start
    while d <= end:
        truth[d.isoformat()] = kind
        d += timedelta(days=1)
print(f"ZigZag truth: {len(truth)} dates, "
      f"{sum(1 for v in truth.values() if v=='bear')} bear / "
      f"{sum(1 for v in truth.values() if v=='bull')} bull")

# 2. V27 router per day
daily = pl.scan_parquet(PANEL).select(["trade_date", "idx_close"]).filter(
    pl.col("idx_close").is_not_null()
).unique("trade_date").sort("trade_date").collect().to_pandas()

daily["ma5"] = daily["idx_close"].rolling(5).mean()
daily["ma10"] = daily["idx_close"].rolling(10).mean()
daily["ma20"] = daily["idx_close"].rolling(20).mean()
daily["ma60"] = daily["idx_close"].rolling(60).mean()
daily["ma120"] = daily["idx_close"].rolling(120).mean()
daily["ret10"] = daily["idx_close"].pct_change(10)
daily["ret20"] = daily["idx_close"].pct_change(20)
daily["ret40"] = daily["idx_close"].pct_change(40)
daily["ret60"] = daily["idx_close"].pct_change(60)
daily["dist_ma20"] = (daily["idx_close"] - daily["ma20"]) / daily["ma20"]
daily["dist_ma60"] = (daily["idx_close"] - daily["ma60"]) / daily["ma60"]
daily["slope_ma60"] = daily["ma60"].diff()
daily["high_60d"] = daily["idx_close"].rolling(60).max()
daily["high_120d"] = daily["high_60d"].rolling(120).max()
daily["dd_60d"] = daily["idx_close"] / daily["high_60d"] - 1.0
daily["dd_120d"] = daily["idx_close"] / daily["high_120d"] - 1.0

specs = [
    daily["idx_close"] < daily["ma5"],
    daily["idx_close"] < daily["ma10"],
    daily["idx_close"] < daily["ma20"],
    daily["idx_close"] < daily["ma60"],
    daily["idx_close"] < daily["ma120"],
    daily["ma20"] < daily["ma60"],
    daily["ma60"] < daily["ma120"],
    daily["ret10"] < 0,
    daily["ret20"] < 0,
    daily["ret40"] < 0,
    daily["ret60"] < 0,
    daily["dist_ma20"] < -0.02,
    daily["dist_ma60"] < -0.05,
    daily["slope_ma60"] < 0,
    daily["dd_60d"] < -0.05,
    daily["dd_120d"] < -0.10,
]
sig_count = sum(s.fillna(False).astype(int) for s in specs)

# V27_BALANCE: vote=2, lookback=5, cum=5
daily["v27_v2_flag"] = (sig_count >= 2).astype(int)
daily["v27_cum"] = daily["v27_v2_flag"].rolling(5, min_periods=1).sum()
daily["is_bear_v27balance"] = (daily["v27_cum"] >= 5).astype(bool)

# V27 original (for comparison): vote=2, lookback=10, cum=4
daily["v27_cum_orig"] = daily["v27_v2_flag"].rolling(10, min_periods=1).sum()
daily["is_bear_v27original"] = (daily["v27_cum_orig"] >= 4).astype(bool)

print(f"V27_BALANCE bear dates: {daily['is_bear_v27balance'].sum()} / {len(daily)} "
      f"({daily['is_bear_v27balance'].mean()*100:.1f}%)")
print(f"V27_original bear dates: {daily['is_bear_v27original'].sum()} / {len(daily)} "
      f"({daily['is_bear_v27original'].mean()*100:.1f}%)")

# Map truth
daily["date_str"] = pd.to_datetime(daily["trade_date"]).dt.date.astype(str)
daily["truth"] = daily["date_str"].map(truth)

# Yearly summary
daily["year"] = pd.to_datetime(daily["trade_date"]).dt.year
yearly = daily.groupby("year").agg(
    n=("is_bear_v27balance", "size"),
    n_bear_v27b=("is_bear_v27balance", "sum"),
    n_bear_v27o=("is_bear_v27original", "sum"),
    n_truth_bear=("truth", lambda s: int((s == "bear").sum())),
    n_truth_bull=("truth", lambda s: int((s == "bull").sum())),
).reset_index()
yearly["pct_v27b"] = (yearly["n_bear_v27b"] / yearly["n"] * 100).round(1)
yearly["pct_v27o"] = (yearly["n_bear_v27o"] / yearly["n"] * 100).round(1)
yearly["pct_truth"] = (yearly["n_truth_bear"] / (yearly["n_truth_bear"] + yearly["n_truth_bull"]).replace(0, 1) * 100).round(1)

print()
print("Year | dates | bear_v27B | bear_v27O | truth_bear")
for _, r in yearly.iterrows():
    print(f"  {int(r['year'])} | {int(r['n']):>3} | "
          f"{int(r['n_bear_v27b']):>3} ({r['pct_v27b']:>5.1f}%) | "
          f"{int(r['n_bear_v27o']):>3} ({r['pct_v27o']:>5.1f}%) | "
          f"{int(r['n_truth_bear']):>3} ({r['pct_truth']:>5.1f}%)")

# Save CSV
daily[["trade_date", "date_str", "idx_close",
       "v27_v2_flag", "v27_cum", "is_bear_v27balance",
       "v27_cum_orig", "is_bear_v27original",
       "truth"]].to_csv(OUT_DIR / "v27_router_daily.csv", index=False)
yearly.to_csv(OUT_DIR / "yearly_summary.csv", index=False)

# Confusion matrix (V27_BALANCE)
common_dates = set(daily["date_str"]) & set(truth.keys())
truth_bear = {d for d in common_dates if truth[d] == "bear"}
v27b_bear = set(daily.loc[daily["is_bear_v27balance"], "date_str"]) & common_dates

tp = len(v27b_bear & truth_bear)
fp = len(v27b_bear - truth_bear)
tn = len(common_dates - truth_bear - v27b_bear)
fn = len(truth_bear - v27b_bear)
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"\nV27_BALANCE confusion matrix:")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"  Precision={prec:.3f} Recall={rec:.3f} Specificity={spec:.3f} F1={f1:.3f}")

# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

# Top: idx_close with bear bands shaded
ax1.plot(daily["date_str"], daily["idx_close"], color="#333", lw=0.8, label="HS300 idx_close")

# Build continuous bear bands for V27_BALANCE
v27b_dates = daily.loc[daily["is_bear_v27balance"], "date_str"].tolist()
v27b_starts, v27b_ends = [], []
prev = None
for d in sorted(v27b_dates):
    if prev is None:
        v27b_starts.append(d); prev = d
    elif (pd.Timestamp(d) - pd.Timestamp(prev)).days > 1:
        v27b_ends.append(prev); v27b_starts.append(d); prev = d
    else:
        prev = d
if prev is not None:
    v27b_ends.append(prev)

for s, e in zip(v27b_starts, v27b_ends):
    ax1.axvspan(s, e, color="#d32f2f", alpha=0.20, lw=0)

# Same for V27_original
v27o_dates = daily.loc[daily["is_bear_v27original"], "date_str"].tolist()
v27o_starts, v27o_ends = [], []
prev = None
for d in sorted(v27o_dates):
    if prev is None:
        v27o_starts.append(d); prev = d
    elif (pd.Timestamp(d) - pd.Timestamp(prev)).days > 1:
        v27o_ends.append(prev); v27o_starts.append(d); prev = d
    else:
        prev = d
if prev is not None:
    v27o_ends.append(prev)
for s, e in zip(v27o_starts, v27o_ends):
    ax1.axvspan(s, e, color="#1976d2", alpha=0.10, lw=0)

# ZigZag ground truth bear bands
truth_bear_dates = [d for d in sorted(truth.keys()) if truth[d] == "bear"]
if truth_bear_dates:
    t_starts, t_ends = [], []
    prev = None
    for d in truth_bear_dates:
        if prev is None:
            t_starts.append(d); prev = d
        elif (pd.Timestamp(d) - pd.Timestamp(prev)).days > 1:
            t_ends.append(prev); t_starts.append(d); prev = d
        else:
            prev = d
    if prev is not None:
        t_ends.append(prev)
    for s, e in zip(t_starts, t_ends):
        ax1.axvspan(s, e, color="#000", alpha=0.10, lw=0)

# Legend proxies
import matplotlib.patches as mp
legend_elems = [
    plt.Line2D([0],[0], color="#333", lw=1, label="HS300 idx_close"),
    mp.Patch(color="#d32f2f", alpha=0.20, label="V27_BALANCE bear (2/5/5)"),
    mp.Patch(color="#1976d2", alpha=0.10, label="V27_original bear (2/10/4)"),
    mp.Patch(color="#000", alpha=0.10, label="ZigZag ground truth bear"),
]
ax1.legend(handles=legend_elems, loc="upper left", fontsize=9)
ax1.set_ylabel("idx_close")
ax1.set_title("V27 router vs ZigZag ground truth (HS300 2010-2025)")
ax1.grid(True, alpha=.22)

# Bottom: 16-signal vote count vs V27 cum
ax2.fill_between(daily["date_str"], 0, sig_count,
                 color="#aaa", alpha=0.4, label="16-signal vote count")
ax2.plot(daily["date_str"], daily["v27_cum"], color="#d32f2f", lw=0.8,
         label="V27_BALANCE cum(5d) - threshold=5")
ax2.plot(daily["date_str"], daily["v27_cum_orig"], color="#1976d2", lw=0.8,
         alpha=0.7, label="V27_original cum(10d) - threshold=4")
ax2.axhline(5, color="#d32f2f", ls=":", lw=0.8)
ax2.axhline(4, color="#1976d2", ls=":", lw=0.8)
ax2.axhline(2, color="#aaa", ls=":", lw=0.8, label="vote threshold=2")
ax2.set_ylabel("vote count")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=.22)

# Reduce x-tick density
step = max(1, len(daily) // 12)
xticks = daily["date_str"][::step]
ax2.set_xticks(xticks)
ax2.tick_params(axis="x", rotation=35)

out = OUT_DIR / "router_vs_truth.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {out}")
print(f"Daily CSV: {OUT_DIR/'v27_router_daily.csv'}")
print(f"Yearly CSV: {OUT_DIR/'yearly_summary.csv'}")
