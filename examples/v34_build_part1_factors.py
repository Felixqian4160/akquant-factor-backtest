"""V34 panel build — Part 1: compute all technical factors on ADJUSTED (hfq) prices.

Rebuilds on adjusted prices:
  - GTJA191 registry  (191 factors)
  - ALPHA101 registry (107 = 101 std + 6 custom)
  - TALIB             (73 indicators, per-stock)
  - ACADEMIC          (12 price-sensitive: l_ami, r_tv, r_beta, p_m1..p_m24, p_mchg, p_52w, p_mdr)
  - GITHUB17          (17 factors)

Output: evidence/v34_adj_20261007/new_factors_adj.parquet
"""
from __future__ import annotations
import ast
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
AURUMQ = pathlib.Path("/media/felix/f/quant/aurumq-rl")
sys.path.insert(0, str(AURUMQ / "quant_workflow" / "src"))

V33 = AKQ / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
ADJ = AKQ / "evidence" / "audit_v33_20261007" / "adj_study" / "adj_prices_354.parquet"
OUTDIR = AKQ / "evidence" / "v34_adj_20261007"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "new_factors_adj.parquet"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ════════════════════════════════════════════════════════════════
# Step 1: load raw panel + adj_factor, build adjusted enriched frame
# ════════════════════════════════════════════════════════════════
log("loading v33 raw cols + adj_factor...")
t0 = time.time()
raw_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "cap", "circ_cap",
            "bps", "grossprofit_margin", "netprofit_yoy"]
df = pl.read_parquet(V33, columns=raw_cols)
log(f"  v33 raw: {df.shape} ({time.time()-t0:.0f}s)")

adj = pl.read_parquet(ADJ, columns=["ts_code", "trade_date", "adj_factor"])
adj = adj.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
df = df.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))

df = df.join(adj, on=["ts_code", "trade_date"], how="left")
n_missing_af = df.filter(pl.col("adj_factor").is_null()).height
log(f"  adj_factor missing rows: {n_missing_af}")
if n_missing_af > 0:
    # fill within stock (island boundaries), then check again
    df = df.sort(["ts_code", "trade_date"]).with_columns(
        pl.col("adj_factor").forward_fill().backward_fill().over("ts_code")
    )
    n2 = df.filter(pl.col("adj_factor").is_null()).height
    log(f"  after fill: {n2} missing")
    if n2 > 0:
        raise RuntimeError("adj_factor still missing")
df = df.sort(["ts_code", "trade_date"])
df = df.with_columns([
    (pl.col("open") * pl.col("adj_factor")).alias("adj_open"),
    (pl.col("high") * pl.col("adj_factor")).alias("adj_high"),
    (pl.col("low") * pl.col("adj_factor")).alias("adj_low"),
    (pl.col("close") * pl.col("adj_factor")).alias("adj_close"),
])
log(f"  adjusted OHLC done")

# Enriched panel (mirrors build_factor_panel.py add_derived_columns, with adj prices)
log("building enriched adjusted panel...")
en = df.with_columns([
    pl.col("ts_code").alias("stock_code"),
]).with_columns([
    pl.col("adj_open").alias("open"),
    pl.col("adj_high").alias("high"),
    pl.col("adj_low").alias("low"),
    pl.col("adj_close").alias("close"),
])
# volume normalization: v33 has vol (and maybe volume). Ensure both present, same value as vol.
en = en.with_columns([
    pl.col("vol").alias("volume"),
    (pl.col("amount") * 10.0 / pl.col("vol").clip(lower_bound=1) * pl.col("adj_factor")).alias("vwap"),
])
en = en.with_columns([
    (pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1.0).fill_null(0.0).alias("returns"),
])
for w in [5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180]:
    en = en.with_columns(
        pl.col("vol").rolling_mean(window_size=w, min_periods=1).over("stock_code").alias(f"adv{w}")
    )
# industry (from Tushare stock_basic cache; for IndNeutralize-style factors)
_ind_path = pathlib.Path("/media/felix/f/quant/aurumq-rl/data_cache/stock_basic_industry.parquet")
if _ind_path.exists():
    _ind = pl.read_parquet(_ind_path).rename({"ts_code": "stock_code"})
    if _ind["stock_code"].dtype == pl.Categorical:
        _ind = _ind.with_columns(pl.col("stock_code").cast(pl.Utf8))
    en = en.join(_ind, on="stock_code", how="left")
    en = en.with_columns(pl.col("industry").alias("sub_industry"))
    n_ind = en["industry"].is_null().sum()
    log(f"  industry merged: nulls={n_ind}")

log(f"  enriched: {en.shape}, cols: {len(en.columns)}")

# ════════════════════════════════════════════════════════════════
# Step 2: GTJA191 + ALPHA101 registries
# ════════════════════════════════════════════════════════════════
log("computing GTJA191 + ALPHA101 on adjusted panel...")
import aurumq_rl.factors.alpha101  # noqa
import aurumq_rl.factors.gtja191  # noqa
from aurumq_rl.factors.registry import ALPHA101_REGISTRY, GTJA191_REGISTRY, sanitize_factor_series

t_reg = time.time()
all_series = []
fails = []
for name, entry in sorted(GTJA191_REGISTRY.items()):
    try:
        s = sanitize_factor_series(entry.impl(en))
        all_series.append(s.alias(f"gtja_{name}"))
    except Exception as e:
        fails.append((f"gtja_{name}", str(e)[:80]))
for name, entry in sorted(ALPHA101_REGISTRY.items()):
    try:
        s = sanitize_factor_series(entry.impl(en))
        all_series.append(s.alias(f"alpha_{name}"))
    except Exception as e:
        fails.append((f"alpha_{name}", str(e)[:80]))
log(f"  computed {len(all_series)} series in {time.time()-t_reg:.0f}s, fails={len(fails)}")
for f in fails[:10]:
    log(f"    FAIL {f[0]}: {f[1]}")

reg_df = en.select(["ts_code", "trade_date"]).with_columns(all_series)
log(f"  registry frame: {reg_df.shape}")

# free memory
del en
import gc; gc.collect()

# ════════════════════════════════════════════════════════════════
# Step 3: TALIB on adjusted OHLC
# ════════════════════════════════════════════════════════════════
log("computing TALIB on adjusted OHLC...")
import akquant.talib as tq

# Extract INDICATORS list from the original rebuild script via AST
src_path = AKQ / "examples" / "rebuild_panel_with_talib.py"
tree = ast.parse(src_path.read_text())
indicators = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for tgt in node.targets:
            if getattr(tgt, "id", None) == "INDICATORS":
                indicators = eval(compile(ast.Expression(node.value), "<ast>", "eval"), {"tq": tq})
    elif isinstance(node, ast.AnnAssign):
        if getattr(node.target, "id", None) == "INDICATORS":
            indicators = eval(compile(ast.Expression(node.value), "<ast>", "eval"), {"tq": tq})
if indicators is None:
    raise RuntimeError("could not extract INDICATORS")
log(f"  indicators: {len(indicators)}")

src_ex = df.select(["ts_code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close", "vol"])
stocks = src_ex["ts_code"].unique().to_list()
log(f"  stocks: {len(stocks)}")

t_tal = time.time()
pieces = []
for i, code in enumerate(stocks, 1):
    g = src_ex.filter(pl.col("ts_code") == code).sort("trade_date").to_pandas().reset_index(drop=True)
    close = g["adj_close"].astype(float); high = g["adj_high"].astype(float)
    low = g["adj_low"].astype(float); open_ = g["adj_open"].astype(float)
    vol = g["vol"].astype(float)
    new_cols = {}
    for name, fn, kind in indicators:
        try:
            out = fn(close, high, low, open_, vol)
            if isinstance(out, tuple):
                for j, sub in enumerate(out):
                    col = f"talib_{name}_{j}"
                    new_cols[col] = sub.to_numpy() if isinstance(sub, pd.Series) else np.asarray(sub, dtype=np.float64)
            elif isinstance(out, pd.Series):
                new_cols[f"talib_{name}"] = out.to_numpy()
            elif isinstance(out, pd.DataFrame):
                for c in out.columns:
                    new_cols[f"talib_{name}_{c}"] = out[c].to_numpy()
            else:
                new_cols[f"talib_{name}"] = np.asarray(out, dtype=np.float64)
        except Exception:
            new_cols[f"talib_{name}"] = np.full(len(g), np.nan)
    piece = pd.concat([g[["ts_code", "trade_date"]],
                       pd.DataFrame(new_cols)], axis=1)
    pieces.append(pl.from_pandas(piece))
    if i % 90 == 0:
        log(f"    {i}/{len(stocks)} ({time.time()-t_tal:.0f}s)")
tal_df = pl.concat(pieces, how="vertical").with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
log(f"  talib done: {tal_df.shape} ({time.time()-t_tal:.0f}s)")

# ════════════════════════════════════════════════════════════════
# Step 4: ACADEMIC price-sensitive (12) on adjusted prices
# ════════════════════════════════════════════════════════════════
log("computing academic-12 on adjusted...")
ac = df.select(["ts_code", "trade_date", "adj_close", "amount", "circ_cap"]).sort(["ts_code", "trade_date"])
ac = ac.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ts_code") - 1.0).alias("adj_ret")
)
# market return from equal-weight adj close
mkt = (ac.group_by("trade_date").agg(pl.col("adj_close").mean().alias("_mc")).sort("trade_date")
       .with_columns((pl.col("_mc") / pl.col("_mc").shift(1) - 1.0).alias("mkt_ret")))
ac = ac.join(mkt.select(["trade_date", "mkt_ret"]), on="trade_date", how="left")

ac = ac.with_columns([
    (pl.col("adj_ret").abs() / pl.max_horizontal(pl.col("amount"), 1)).rolling_mean(21).over("ts_code").alias("l_ami"),
    pl.col("adj_ret").rolling_std(60).over("ts_code").alias("r_tv"),
    pl.col("adj_close").shift(21).over("ts_code").pct_change(21).alias("p_m1"),
    pl.col("adj_close").shift(21).over("ts_code").pct_change(63).alias("p_m3"),
    pl.col("adj_close").shift(21).over("ts_code").pct_change(126).alias("p_m6"),
    pl.col("adj_close").shift(21).over("ts_code").pct_change(231).alias("p_m11"),
    pl.col("adj_close").shift(21).over("ts_code").pct_change(504).alias("p_m24"),
    (pl.col("adj_close").pct_change(126).shift(21).over("ts_code")
     - pl.col("adj_close").pct_change(252).shift(21).over("ts_code")).alias("p_mchg"),
    (pl.col("adj_close") / pl.col("adj_close").rolling_max(252).over("ts_code")).alias("p_52w"),
    pl.col("adj_ret").rolling_max(21).over("ts_code").alias("p_mdr"),
])
# r_beta: rolling 60d beta of adj_ret vs mkt_ret (pandas for rolling cov/var)
_b = ac.select(["ts_code", "trade_date", "adj_ret", "mkt_ret"]).to_pandas().sort_values(["ts_code", "trade_date"])
_b["r_beta"] = _b.groupby("ts_code").apply(
    lambda g: g["adj_ret"].rolling(60).cov(g["mkt_ret"]) / g["mkt_ret"].rolling(60).var().replace(0, np.nan),
    include_groups=False,
).reset_index(level=0, drop=True).values
_b_pl = pl.from_pandas(_b[["ts_code", "trade_date", "r_beta"]])
if _b_pl["trade_date"].dtype != pl.Datetime("ms"):
    _b_pl = _b_pl.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
ac = ac.join(_b_pl, on=["ts_code", "trade_date"], how="left")

ac_done = ac.select(["ts_code", "trade_date", "l_ami", "r_tv", "r_beta", "p_m1", "p_m3", "p_m6",
                     "p_m11", "p_m24", "p_mchg", "p_52w", "p_mdr"])
log(f"  academic-12 done: {ac_done.shape}")

# ════════════════════════════════════════════════════════════════
# Step 5: GITHUB17 on adjusted prices
# ════════════════════════════════════════════════════════════════
log("computing github17 on adjusted...")
gh_src = df.select(["ts_code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close",
                    "vol", "amount", "bps", "grossprofit_margin", "netprofit_yoy"]).sort(["ts_code", "trade_date"])
gh = gh_src.with_columns([
    pl.col("adj_open").alias("open"), pl.col("adj_high").alias("high"),
    pl.col("adj_low").alias("low"), pl.col("adj_close").alias("close"),
])
# market cols from adj close
_mkt = (gh.group_by("trade_date").agg(pl.col("close").mean().alias("_mc")).sort("trade_date")
        .with_columns([
            (pl.col("_mc") / pl.col("_mc").shift(1) - 1.0).alias("_mkt_ret"),
            (pl.col("_mc").shift(21) / pl.col("_mc").shift(126) - 1.0).alias("_mkt_ret6"),
        ]))
gh = gh.join(_mkt.select(["trade_date", "_mkt_ret", "_mkt_ret6"]), on="trade_date", how="left")

CLOSE = pl.col("close")
def f_winner_ratio(d):
    parts = [((CLOSE.shift(i).over("ts_code")) < CLOSE).cast(pl.Float64).fill_null(0.0) for i in range(1, 31)]
    return d.select(pl.when(CLOSE.is_null()).then(None).otherwise(pl.sum_horizontal(parts) / 30.0).alias("winner_ratio"))["winner_ratio"]

def f_efficiency_ratio(d):
    num = CLOSE.diff(20).abs().over("ts_code")
    den = CLOSE.diff(1).abs().rolling_sum(20).over("ts_code")
    return d.select((num / (den + 1e-12)).alias("efficiency_ratio"))["efficiency_ratio"]

def f_fractal_dimension(d):
    w = d.with_columns((pl.col("high").rolling_max(20).over("ts_code") - pl.col("low").rolling_min(20).over("ts_code")).alias("_h_range"))
    w = w.with_columns(((1.0/20)**2 + (CLOSE.diff(1).over("ts_code") / (pl.col("_h_range") + 1e-9))**2).sqrt().alias("_leg"))
    w = w.with_columns(pl.col("_leg").rolling_sum(20).over("ts_code").alias("_L"))
    w = w.with_columns((1 + (pl.col("_L").log() + np.log(2)) / np.log(40.0)).alias("fractal_dimension"))
    return w["fractal_dimension"]

def f_alpha191_040(d):
    w = d.with_columns([
        (pl.col("vol") * (CLOSE > CLOSE.shift(1).over("ts_code")).cast(pl.Float64)).alias("_up"),
        (pl.col("vol") * (CLOSE <= CLOSE.shift(1).over("ts_code")).cast(pl.Float64)).alias("_dn"),
    ])
    w = w.with_columns([
        pl.col("_up").rolling_sum(26).over("ts_code").alias("_ups"),
        pl.col("_dn").rolling_sum(26).over("ts_code").alias("_dns"),
    ])
    return w.with_columns((pl.col("_ups") / (pl.col("_dns") + 1e-9)).alias("alpha191_040"))["alpha191_040"]

def f_alpha191_095(d):
    return d.select(pl.col("amount").rolling_std(20).over("ts_code").alias("alpha191_095"))["alpha191_095"]

def f_mom12m_jt(d):
    return d.select((CLOSE.shift(21).over("ts_code") / CLOSE.shift(252).over("ts_code") - 1.0).alias("mom12m_jt"))["mom12m_jt"]

def f_maxret_bcw(d):
    w = d.with_columns((CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    return w.with_columns(pl.col("_ret_lag").rolling_max(21).over("ts_code").alias("maxret_bcw"))["maxret_bcw"]

def f_accruals_sloan(d):
    return d.select(((pl.col("bps") / pl.col("bps").shift(252).over("ts_code") - 1.0) * 100.0 - pl.col("netprofit_yoy")).alias("accruals_sloan"))["accruals_sloan"]

def f_idiovola(d):
    w = d.with_columns((CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).abs().alias("_absret"))
    w = w.with_columns(pl.col("_absret").shift(1).over("ts_code").alias("_absret_lag"))
    return w.with_columns(pl.col("_absret_lag").rolling_std(60).over("ts_code").alias("idiovola_clmx"))["idiovola_clmx"]

def f_gp_novymarx(d):
    return d.select(pl.col("grossprofit_margin").alias("gp_novymarx"))["gp_novymarx"]

def f_overnight_intraday(d):
    return d.select((CLOSE / pl.col("open") - pl.col("open") / CLOSE.shift(1).over("ts_code")).alias("overnight_intraday_spread"))["overnight_intraday_spread"]

def f_skew21(d):
    w = d.with_columns((CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    return w.with_columns(pl.col("_ret_lag").rolling_skew(21).over("ts_code").alias("skew21_lottery"))["skew21_lottery"]

def f_pvcorr_21(d):
    w = d.with_columns([
        (CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).alias("_ret"),
        (pl.col("vol") / pl.col("vol").shift(1).over("ts_code") - 1.0).alias("_volchg"),
    ])
    w = w.with_columns([
        pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"),
        pl.col("_volchg").shift(1).over("ts_code").alias("_volchg_lag"),
    ])
    w = w.with_columns((pl.col("_ret_lag") * pl.col("_volchg_lag")).alias("_xy"))
    w = w.with_columns([
        pl.col("_xy").rolling_mean(21).over("ts_code").alias("_exy"),
        pl.col("_ret_lag").rolling_mean(21).over("ts_code").alias("_ex"),
        pl.col("_volchg_lag").rolling_mean(21).over("ts_code").alias("_ey"),
        pl.col("_ret_lag").rolling_std(21).over("ts_code").alias("_sx"),
        pl.col("_volchg_lag").rolling_std(21).over("ts_code").alias("_sy"),
    ])
    return w.with_columns(((pl.col("_exy") - pl.col("_ex") * pl.col("_ey")) / (pl.col("_sx") * pl.col("_sy") + 1e-12)).alias("pvcorr_21"))["pvcorr_21"]

def f_kurt21(d):
    w = d.with_columns((CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    return w.with_columns(pl.col("_ret_lag").rolling_kurtosis(21).over("ts_code").alias("kurt21_returns"))["kurt21_returns"]

def f_coskew60(d):
    w = d.with_columns([
        (CLOSE / CLOSE.shift(1).over("ts_code") - 1.0).alias("_ret"),
        (pl.col("_mkt_ret") ** 2).alias("_m2"),
    ])
    w = w.with_columns((pl.col("_ret") * pl.col("_m2")).alias("_rm2"))
    w = w.with_columns([
        pl.col("_rm2").rolling_mean(60).over("ts_code").alias("_erm2"),
        pl.col("_m2").rolling_mean(60).over("ts_code").alias("_em2"),
    ])
    return w.with_columns((pl.col("_erm2") / (pl.col("_em2") + 1e-9)).alias("coskew60"))["coskew60"]

def f_hl_52w(d):
    return d.select(((CLOSE - pl.col("low").rolling_min(252).over("ts_code"))
                     / (pl.col("high").rolling_max(252).over("ts_code") - pl.col("low").rolling_min(252).over("ts_code") + 1e-9)).alias("hl_52w_disposition"))["hl_52w_disposition"]

def f_resmom_6m(d):
    return d.select(((CLOSE.shift(21).over("ts_code") / CLOSE.shift(126).over("ts_code") - 1.0) - pl.col("_mkt_ret6")).alias("resmom_6m"))["resmom_6m"]

GH = {
    "winner_ratio": f_winner_ratio, "efficiency_ratio": f_efficiency_ratio,
    "fractal_dimension": f_fractal_dimension, "alpha191_040": f_alpha191_040,
    "alpha191_095": f_alpha191_095, "mom12m_jt": f_mom12m_jt,
    "maxret_bcw": f_maxret_bcw, "accruals_sloan": f_accruals_sloan,
    "idiovola_clmx": f_idiovola, "gp_novymarx": f_gp_novymarx,
    "overnight_intraday_spread": f_overnight_intraday, "skew21_lottery": f_skew21,
    "pvcorr_21": f_pvcorr_21, "kurt21_returns": f_kurt21, "coskew60": f_coskew60,
    "hl_52w_disposition": f_hl_52w, "resmom_6m": f_resmom_6m,
}
gh_series = []
for fid, fn in GH.items():
    s = fn(gh)
    s = pl.when(s.is_infinite() | s.is_nan()).then(None).otherwise(s.clip(-1e6, 1e6)).alias(fid)
    gh_series.append(gh.select(s)[fid].alias(fid))
gh_df = gh.select(["ts_code", "trade_date"]).with_columns(gh_series)
log(f"  github17 done: {gh_df.shape}")

# ════════════════════════════════════════════════════════════════
# Step 6: assemble + save new factors
# ════════════════════════════════════════════════════════════════
log("assembling new factors frame...")
base_keys = df.select(["ts_code", "trade_date"])
out = base_keys.join(reg_df, on=["ts_code", "trade_date"], how="left")
out = out.join(tal_df, on=["ts_code", "trade_date"], how="left")
out = out.join(ac_done, on=["ts_code", "trade_date"], how="left")
out = out.join(gh_df, on=["ts_code", "trade_date"], how="left")
log(f"  combined: {out.shape}")

out.write_parquet(OUT)
log(f"saved {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")

summary = {
    "registry_series": len(all_series),
    "registry_fails": fails[:20],
    "talib_cols": len([c for c in tal_df.columns if c.startswith('talib_')]),
    "academic12": 12,
    "github17": 17,
    "total_new_cols": out.shape[1] - 2,
    "rows": out.height,
}
(OUTDIR / "part1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log("DONE part 1")
