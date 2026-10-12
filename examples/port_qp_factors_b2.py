"""QuantsPlaybook 批次2 移植: CGO 处置效应 (N=100 成本衰减) + STR 凸显性 — 含初筛与归因审计.

CGO: RP_t = Σ w_i·avg_i / Σ w_i, w_i = τ_i·Π_{j=i+1..t}(1-τ_j)  (N=100 窗口)
     fact = adj_close / RP - 1;  avg = VWAP_adj = amount*10/vol*adj_factor (fallback: (H+L+C)/3)
STR: sigma = |r-bench|/(|r|+|bench|+0.1); rank ↓ per day; w = δ^rank/mean; STR = rolling20 cov(w, r), δ=0.7
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
DEDUP = PANEL
OUT = AKQ / "evidence" / "v35_new_factors_20261012"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


cols = pl.read_parquet_schema(PANEL)
HAS_AF = "adj_factor" in cols
sel = ["trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
       "vol", "amount", "turnover_rate", "idx_close"]
if HAS_AF:
    sel.append("adj_factor")
df = pl.read_parquet(PANEL, columns=sel).to_pandas()
df["trade_date"] = pd.to_datetime(df["trade_date"])


def wide(col):
    return df.pivot(index="trade_date", columns="ts_code", values=col).sort_index()


C = wide("adj_close")
T, S = C.shape
H, L = wide("adj_high"), wide("adj_low")
V, A, TR = wide("vol"), wide("amount"), wide("turnover_rate")
IDX = wide("idx_close")
log(f"panel {T}x{S}, adj_factor={HAS_AF}")

# ---- VWAP_adj ----
with np.errstate(all="ignore"):
    vwap_raw = 10.0 * A / V          # 元 (amount千元, vol手)
if HAS_AF:
    AF = wide("adj_factor")
    AVG = vwap_raw * AF
else:
    AVG = (H + L + C) / 3.0
AVG = AVG.where(np.isfinite(AVG) & (AVG > 0))
AVG = AVG.fillna((H + L + C) / 3.0)

# ---- CGO (N=100) ----
t0 = time.time()
N = 100
cgo = np.full((T, S), np.nan)
Cn, An, Tn = C.to_numpy(), AVG.to_numpy(), (TR / 100.0).to_numpy()
for j in range(S):
    c, a, tau = Cn[:, j], An[:, j], Tn[:, j]
    if not (np.isfinite(c).sum() > N and np.isfinite(a).sum() > N):
        continue
    cw = sliding_window_view(c, N)
    aw = sliding_window_view(a, N)
    tw = sliding_window_view(tau, N)
    okw = np.isfinite(cw).all(1) & np.isfinite(aw).all(1) & np.isfinite(tw).all(1)
    tau2 = np.clip(np.nan_to_num(tw, nan=0.0), 0.0, 0.9)
    rev = np.cumprod((1.0 - tau2)[:, ::-1], axis=1)[:, ::-1]   # R[:,i]=Π_{j>=i}(1-τ)
    decay = np.empty_like(rev)
    decay[:, :-1] = rev[:, 1:]
    decay[:, -1] = 1.0
    w = tau2 * decay
    den = w.sum(1)
    with np.errstate(all="ignore"):
        rp = (w * np.nan_to_num(aw, nan=0.0)).sum(1) / den
        val = cw[:, -1] / rp - 1.0
    val = np.where(okw & (den > 0) & np.isfinite(rp) & (rp > 0), val, np.nan)
    cgo[N - 1:, j] = val
CGO = pd.DataFrame(cgo, index=C.index, columns=C.columns)
log(f"CGO done {time.time()-t0:.0f}s coverage {np.isfinite(cgo).mean():.1%}")

# ---- STR ----
t0 = time.time()
r = C.pct_change()
bench = IDX.mean(axis=1).pct_change()   # 指数收益 (Series)
b = bench.to_numpy()
sig = np.abs(r.to_numpy() - b[:, None]) / (np.abs(r.to_numpy()) + np.abs(b)[:, None] + 0.1)
SIG = pd.DataFrame(sig, index=C.index, columns=C.columns)
SIG[r.isna().to_numpy()] = np.nan
rk = SIG.rank(axis=1, ascending=False)
wgt = np.power(0.7, rk)
wgt = wgt.div(wgt.mean(axis=1), axis=0)
STR = wgt.rolling(20).cov(r)
STR[r.isna().to_numpy()] = np.nan
log(f"STR done {time.time()-t0:.0f}s coverage {np.isfinite(STR.to_numpy()).mean():.1%}")

results = {"qp_cgo": CGO, "qp_str": STR}

# ---- 初筛 ----
O = wide("adj_open")
fwd = (C.shift(-21) / O.shift(-1) - 1.0 - 0.005).to_numpy()
screen_rows = []
for name, w in results.items():
    f = w.to_numpy()
    sp = np.full(T, np.nan)
    for t in range(T):
        v, rr = f[t], fwd[t]
        m = np.isfinite(v) & np.isfinite(rr)
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        vals, rs = v[idx], rr[idx]
        order = np.argsort(-vals)
        sp[t] = rs[order[:10]].mean() - rs[order[-10:]].mean()
    sps = pd.Series(sp, index=C.index).dropna()
    m1 = sps.loc[:"2017-12-31"]
    m2 = sps.loc["2018-01-01":]

    def stat(x):
        return (x.mean(), x.mean() / x.std() * np.sqrt(len(x)), (x > 0).mean())
    a = stat(sps)
    b1 = stat(m1)
    b2 = stat(m2)
    screen_rows.append({"factor": name, "mean_bps": a[0] * 1e4, "t": a[1], "hit": a[2],
                        "mean_1017_bps": b1[0] * 1e4, "t_1017": b1[1],
                        "mean_1825_bps": b2[0] * 1e4, "t_1825": b2[1]})
    log(f"SCREEN {name}: {a[0]*1e4:7.1f}bps t={a[1]:6.2f} hit={a[2]:.0%} | 10-17 {b1[0]*1e4:6.1f} t={b1[1]:5.2f} | 18-25 {b2[0]*1e4:6.1f} t={b2[1]:5.2f}")
pd.DataFrame(screen_rows).to_csv(OUT / "screen_b2.csv", index=False)
log("screen saved")

# ---- 保存 ----
frames = [w.stack().rename(n) for n, w in results.items()]
new = pd.concat(frames, axis=1).reset_index()
new.columns = ["trade_date", "ts_code"] + list(results.keys())
pl.from_pandas(new).write_parquet(OUT / "new_factors_b2.parquet")
log(f"saved new_factors_b2.parquet ({len(new)} rows)")

# ---- 归因审计 (vs 413 库) ----
col_info = pl.read_parquet_schema(DEDUP)
SKIP = {"trade_date", "ts_code", "symbol", "open", "high", "low", "close", "vol", "amount",
        "pct_chg", "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
        "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60", "turnover_rate",
        "circ_cap", "cap", "volume", "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
lib_factors = [c for c in col_info.keys() if c not in SKIP and not c.startswith("v10_1_")]
new_factor_names = list(results.keys())
sample_dates = pd.to_datetime(np.sort(new["trade_date"].unique())[::20])
samp = pl.Series("d", sample_dates).cast(pl.Datetime("ms"))
lib_df = (pl.scan_parquet(DEDUP).filter(pl.col("trade_date").is_in(samp))
          .select(["trade_date", "ts_code"] + lib_factors).collect().to_pandas())
lib_df["trade_date"] = pd.to_datetime(lib_df["trade_date"])
acc = {n: [] for n in new_factor_names}
for d in sample_dates:
    a = new[new["trade_date"] == d]
    bdf = lib_df[lib_df["trade_date"] == d]
    if len(a) < 30 or len(bdf) < 30:
        continue
    m = a.merge(bdf, on="ts_code")
    if len(m) < 30:
        continue
    R = m[new_factor_names + lib_factors].rank()
    Cm = R.corr()
    for n in new_factor_names:
        acc[n].append(Cm.loc[n, lib_factors])
print()
for n in new_factor_names:
    if not acc[n]:
        continue
    dfa = pd.DataFrame(acc[n])
    mc = dfa.abs().mean(axis=0).sort_values(ascending=False)
    top5 = mc.head(5)
    txt = ", ".join(f"{f}={dfa[f].mean():+.2f}" for f in top5.index)
    log(f"AUDIT {n}: max|corr|={top5.iloc[0]:.3f} | {txt}")
