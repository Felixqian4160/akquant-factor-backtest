"""QuantsPlaybook 因子移植批次1: shadow(4) + ma_conv(4) + amplitude(2), 附内置效能初筛.

合同:
  - 价格用 adj_* (与 v34 因子库口径一致); vol/amount/turnover_rate 原值
  - 全部 causal (标准化用 shift(1) 滚动均值; 无未来函数)
  - 输出: evidence/v35_new_factors_20261012/new_factors_b1.parquet (long)
  - 初筛: decile-spread 口径与 v34 矩阵一致: fwd[t]=adj_close[t+21]/adj_open[t+1]-1-0.005
用法: port_qp_factors_b1.py [--screen-only]
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
OUT = AKQ / "evidence" / "v35_new_factors_20261012"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    df = pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "adj_open", "adj_high",
                                         "adj_low", "adj_close", "vol", "amount", "turnover_rate"])
    pdf = df.to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])

    def wide(col):
        return pdf.pivot(index="trade_date", columns="ts_code", values=col).sort_index()

    O, H, L, C = wide("adj_open"), wide("adj_high"), wide("adj_low"), wide("adj_close")
    V, A, TR = wide("vol"), wide("amount"), wide("turnover_rate")
    T, S = C.shape
    log(f"panel: {T} x {S}")

    results = {}

    # ---- 1) 上下影线 (东吴证券): 标准化 = shadow / 滚动5日均值.shift(1) ----
    def norm5(x):
        return x / x.rolling(5).mean().shift(1)

    results["qp_shadow_up"] = norm5(H - np.maximum(C, O))     # 上影线
    results["qp_shadow_lo"] = norm5(np.minimum(C, O) - L)     # 下影线
    results["qp_wshadow_up"] = norm5(H - C)                   # 威廉上影线
    results["qp_wshadow_lo"] = norm5(C - L)                   # 威廉下影线

    # ---- 2) 均线收敛与发散 (开源91): -log(1+std([raw, MA5..MA120], ddof=1)) ----
    PERIODS = [5, 10, 20, 60, 120]

    def conv(x):
        arr = x.to_numpy()
        stack = np.stack([arr] + [x.rolling(w).mean().to_numpy() for w in PERIODS], axis=2)
        with np.errstate(all="ignore"):
            std = np.nanstd(stack, axis=2, ddof=1)
        out = -np.log1p(np.where(std >= 0, std, np.nan))
        out[np.isnan(arr)] = np.nan
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    results["qp_pcf"] = conv(C)
    results["qp_vcf"] = conv(V)
    results["qp_acf"] = conv(A)
    results["qp_trcf"] = conv(TR)

    # ---- 3) 振幅因子隐藏结构: af=H/L-1; 在"收盘处于近N日高分位"日取 af 均值 ----
    def amp_hidden(lamb: float, group: str, N: int = 20):
        af = (H / L - 1.0).to_numpy()
        c = C.to_numpy()
        out = np.full((T, S), np.nan)
        for j in range(S):
            cj, afj = c[:, j], af[:, j]
            if np.all(np.isnan(cj)):
                continue
            cw = sliding_window_view(cj, N)          # (W, N)
            afw = sliding_window_view(afj, N)
            # 窗口内逐日分位秩: ranks[w,i] = fraction(cw[w,j] <= cw[w,i])
            with np.errstate(invalid="ignore"):
                ranks = (cw[:, None, :] <= cw[:, :, None]).mean(axis=2)  # (W, N)
                if group == "high":
                    cond = ranks >= lamb
                else:
                    cond = ranks < lamb
            ok = np.isfinite(afw).all(axis=1) & np.isfinite(ranks).all(axis=1)
            val = np.where(cond, np.nan_to_num(afw, nan=0.0), 0.0).mean(axis=1)
            val = np.where(ok, val, np.nan)
            out[N - 1:, j] = val
        return pd.DataFrame(out, index=C.index, columns=C.columns)

    results["qp_af_high"] = amp_hidden(0.5, "high")
    results["qp_af_low"] = amp_hidden(0.5, "low")

    # ---- 覆盖率 ----
    for name, w in results.items():
        cov = float(np.isfinite(w.to_numpy()).mean())
        log(f"{name:16s} coverage {cov:.1%}")

    # ---- 初筛: decile-spread (与 v34 矩阵同口径) ----
    fwd = (C.shift(-21) / O.shift(-1) - 1.0 - 0.005).to_numpy()
    dt_index = C.index
    screen_rows = []
    for name, w in results.items():
        f = w.to_numpy()
        sp = np.full(T, np.nan)
        for t in range(T):
            v = f[t]
            r = fwd[t]
            m = np.isfinite(v) & np.isfinite(r)
            if m.sum() < 30:
                continue
            idx = np.where(m)[0]
            vals = v[idx]
            rr = r[idx]
            order = np.argsort(-vals)
            top = rr[order[:10]]
            bot = rr[order[-10:]]
            sp[t] = top.mean() - bot.mean()
        sps = pd.Series(sp, index=dt_index)
        sub1 = sps.loc[:"2017-12-31"]
        sub2 = sps.loc["2018-01-01":]
        def stat(x):
            x = x.dropna()
            if len(x) < 30:
                return (np.nan, np.nan, np.nan)
            return (x.mean(), x.mean() / x.std() * np.sqrt(len(x)), (x > 0).mean())
        m_all, t_all, h_all = stat(sps)
        m1, t1, _ = stat(sub1)
        m2, t2, _ = stat(sub2)
        screen_rows.append({"factor": name, "mean_bps": m_all * 1e4, "t": t_all, "hit": h_all,
                            "mean_1017_bps": m1 * 1e4, "t_1017": t1, "mean_1825_bps": m2 * 1e4, "t_1825": t2})
        log(f"SCREEN {name:16s} spread {m_all*1e4:6.1f}bps t={t_all:5.2f} hit={h_all:.0%} "
            f"| 10-17 {m1*1e4:6.1f}bps t={t1:5.2f} | 18-25 {m2*1e4:6.1f}bps t={t2:5.2f}")
    pd.DataFrame(screen_rows).to_csv(OUT / "screen_b1.csv", index=False)

    # ---- 保存 ----
    frames = []
    for name, w in results.items():
        s = w.stack().rename(name)
        frames.append(s)
    new = pd.concat(frames, axis=1).reset_index()
    new.columns = ["trade_date", "ts_code"] + list(results.keys())
    out_p = OUT / "new_factors_b1.parquet"
    pl.from_pandas(new).write_parquet(out_p)
    log(f"saved {out_p} ({len(new)} rows)")
    log(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
