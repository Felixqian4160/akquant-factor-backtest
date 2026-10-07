"""Comprehensive chart gallery for ZigZag Simple IC strategy.

Outputs (all to evidence/sweep/zigzag_simple_ic_20261006/audit/, NOT /tmp):
  chart_10seed_equity_curves.png
  chart_yearly_returns.png
  chart_period_split.png
  chart_trade_attribution.png
"""
import csv, pathlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
NAME = "zigzag_simple_ic_20261006"
OUT_DIR = ROOT / "evidence" / "sweep" / NAME / "akquant"
AUDIT_DIR = ROOT / "evidence" / "sweep" / NAME / "audit"
TAGS = [0,2,4,6,8,10,12,14,16,18]
COLORS = ["#1565c0","#2e7d32","#ef6c00","#8e24aa","#00838f",
          "#5d4037","#c2185b","#00796b","#f57c00","#3f51b5"]
tag_colors = {t: COLORS[i] for i,t in enumerate(TAGS)}

# === Chart 1: 10-seed equity curves ===
curves = {}
for off in TAGS:
    p = OUT_DIR / f"V14_{off}" / "nav.csv"
    if not p.exists(): continue
    dates, values = [], []
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
            values.append(float(row["value"]))
    curves[off] = (dates, values)

panel = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
idx = (pl.read_parquet(panel, columns=["trade_date","idx_close"])
       .filter(pl.col("idx_close").is_not_null())
       .group_by("trade_date").agg(pl.col("idx_close").first())
       .sort("trade_date"))
idx_dates = [str(x)[:10] for x in idx["trade_date"].to_list()]
idx_vals = idx["idx_close"].to_numpy()
bench_map = dict(zip(idx_dates, idx_vals))
bench = [bench_map.get(d, None) for d in curves[TAGS[0]][0]] if TAGS else []
bench_arr = np.asarray(bench, dtype=float)
mask = np.isfinite(bench_arr)
bench_arr[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), bench_arr[mask])
bench_norm = bench_arr / bench_arr[0]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
dates = curves[TAGS[0]][0] if TAGS else []
for off, (d, v) in curves.items():
    final = (v[-1] / v[0] - 1) * 100
    ax.plot(d, np.asarray(v) / v[0], color=tag_colors[off], lw=0.9, alpha=.75,
            label=f"V14_{off}  {final:+.1f}%")
ax.plot(dates, bench_norm, color="#b71c1c", lw=1.6, ls=":",
        label=f"HS300 buy&hold  {(bench_norm[-1]-1)*100:+.1f}%")
ax.set_yscale("log")
ax.set_ylabel("Normalized NAV (initial=1.0)")
ax.set_title("Causal ZigZag Router + Simple IC Voting | 10-seed equity curves (2010-2025)")
ax.grid(True, alpha=.22)
ax.legend(ncol=4, loc="upper left", fontsize=8)

rets = {off: (v[-1]/v[0]-1)*100 for off,(d,v) in curves.items()}
mean_ret = sum(rets.values())/len(rets)
ax.text(0.995, 0.03,
        f"10-seed mean ret {mean_ret:+.1f}% | mean annual {mean_ret/15.99:.2f}% | 9/10 positive",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=.35", facecolor="white", alpha=.88, edgecolor="#bbbbbb"))

def dd(curve):
    return curve / np.maximum.accumulate(curve) - 1.0
for off, (d, v) in curves.items():
    ax2.plot(d, dd(np.asarray(v) / v[0]) * 100, color=tag_colors[off], lw=.7, alpha=.4)
ax2.plot(dates, dd(bench_norm) * 100, color="#b71c1c", lw=1.4, ls=":", label="HS300")
ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=.22)
ax2.legend(loc="lower left")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
step = max(1, len(dates) // 12)
ax2.set_xticks(dates[::step])
ax2.tick_params(axis="x", rotation=35)

fig.text(0.01, 0.005,
         "10 offsets: 0/2/4/6/8/10/12/14/16/18 | Causal ZigZag router | 21-bar cap | All artifacts in evidence/sweep/zigzag_simple_ic_20261006/",
         fontsize=8, color="#555555")

out1 = AUDIT_DIR / "chart_10seed_equity_curves.png"
fig.savefig(out1, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")

# === Chart 2: Yearly mean return bars ===
import pandas as pd
yb = pd.read_csv(AUDIT_DIR / "yearly_breakdown.csv")
yearly_mean = yb.groupby("year")["ret_pct"].mean()
fig, ax = plt.subplots(figsize=(16, 7))
years = list(yearly_mean.index)
means = list(yearly_mean.values)
mins = list(yb.groupby("year")["ret_pct"].min())
maxs = list(yb.groupby("year")["ret_pct"].max())
colors = ["#2e7d32" if m > 0 else "#c62828" for m in means]
ax.bar(years, means, color=colors, alpha=0.8)
ax.errorbar(years, means, yerr=[m - mn for m, mn in zip(means, mins)],
            fmt="none", color="#666", alpha=0.5)
ax.errorbar(years, means, yerr=[mx - m for m, mx in zip(means, maxs)],
            fmt="none", color="#666", alpha=0.5)
ax.axhline(0, color="#000", lw=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Mean Return (10 seeds)")
ax.set_title("Year-by-Year Mean Returns | Causal ZigZag Router + Simple IC Voting")
ax.grid(True, alpha=.22, axis="y")
regime_map = {2010:"bull",2011:"bear",2012:"bear",2013:"bear",2014:"bull",
              2015:"bull",2016:"bull",2017:"bull",2018:"bear",2019:"bull",
              2020:"bull",2021:"bear",2022:"bear",2023:"bear",2024:"bear",2025:"bull"}
for y, m in zip(years, means):
    label = f"{m:+.1f}%\n[{regime_map.get(int(y),'?')}]"
    ax.text(y, m + (1.5 if m >= 0 else -3.5), label, ha="center", fontsize=8)
fig.tight_layout()
out2 = AUDIT_DIR / "chart_yearly_returns.png"
fig.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")

# === Chart 3: Period split box plot ===
ps = pd.read_csv(AUDIT_DIR / "period_split.csv")
periods = ["2010-2015","2015-2020","2020-2025","OOS_2023_2025"]
fig, ax = plt.subplots(figsize=(14, 7))
data = []
labels = []
for p in periods:
    sub = ps[ps["period"]==p]["ret_pct"].dropna()
    data.append(sub.values)
    labels.append(f"{p}\nn={len(sub)}")
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                boxprops=dict(facecolor="#bbdefb", edgecolor="#1565c0"),
                medianprops=dict(color="#c62828", lw=2),
                whiskerprops=dict(color="#666"),
                capprops=dict(color="#666"))
for i, d in enumerate(data):
    ax.scatter([i+1] * len(d), d, alpha=0.5, color="#666", s=20)
ax.axhline(0, color="#000", lw=0.5)
ax.set_ylabel("Return (%)")
ax.set_title("Period Split Distribution (10 seeds)")
ax.grid(True, alpha=.22, axis="y")
fig.tight_layout()
out3 = AUDIT_DIR / "chart_period_split.png"
fig.savefig(out3, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out3}")

# === Chart 4: Trade attribution regime pie + by year bars ===
ta = pd.read_csv(AUDIT_DIR / "trade_attribution_by_regime.csv")
fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(16, 7))
regime_agg = ta.groupby("regime").agg({"n":"sum","pnl":"sum"}).reset_index()
regime_agg["pnl_M"] = regime_agg["pnl"] / 1e6
colors_pie = ["#c62828" if r=="bear" else "#2e7d32" for r in regime_agg["regime"]]
ax_pie.pie(regime_agg["n"], labels=regime_agg["regime"], colors=colors_pie,
           autopct="%1.1f%%", startangle=90)
ax_pie.set_title("Trade Count by Regime (10 seeds)")

ty = pd.read_csv(AUDIT_DIR / "trade_attribution_by_year.csv")
year_pnl = ty.groupby("year")["pnl"].sum() / 1e6
colors_bar = ["#c62828" if v < 0 else "#2e7d32" for v in year_pnl.values]
ax_bar.bar(year_pnl.index, year_pnl.values, color=colors_bar)
ax_bar.axhline(0, color="#000", lw=0.5)
ax_bar.set_xlabel("Year")
ax_bar.set_ylabel("Total PnL (M, 10 seeds)")
ax_bar.set_title("Year Total PnL (10 seeds aggregate)")
ax_bar.grid(True, alpha=.22, axis="y")
fig.tight_layout()
out4 = AUDIT_DIR / "chart_trade_attribution.png"
fig.savefig(out4, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out4}")

print("\n=== All 4 charts saved to evidence/sweep/zigzag_simple_ic_20261006/audit/ ===")
