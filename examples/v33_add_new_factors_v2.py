"""V33 panel v2 — 集成 17 个新因子 (Expr-style, 修复 v1 的 Series.over bug)

v1 → v2 修复:
  - v1 用了 `df["close"].pct_change().over(...)` — Series.over 不存在 (polars 1.43)
  - v1 winner_ratio 用了裸 `.rolling(30)` 链式 — 无效 API
  - v2 全部改为 Expr-style: `pl.col(...).shift/rolling_*.over("ts_code")`
  - pvcorr_21 手工协方差分解 (polars 无 rolling_corr)
  - coskew60/resmom_6m 通过 replace_strict 预计算市场收益 (避免 join 乱序)

输入: V32 panel (441 cols) → 输出: V33 panel (458 cols)
自检: 列数/行数/行序还原/null 分布/独立 pandas 复算
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PANEL_IN = Path('data/wavehunter_hs300_v32_complete_20261003.parquet')
PANEL_OUT = Path('data/wavehunter_hs300_v33_with_new_factors_20261003.parquet')
EVIDENCE = Path('evidence/v33_new_factors_20261003')


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================================================
# 17 factor functions — Expr-style, all validated API patterns
# ============================================================

def factor_winner_ratio(df: pl.DataFrame) -> pl.Series:
    """WINNER 获利盘比例 (wukan1986/ta_cn 简化版).
    过去 30 天收盘价低于今天的比例 (0~1). hi: 套牢盘少 (反弹抛压小)."""
    close = pl.col("close")
    parts = [
        ((close.shift(i).over("ts_code")) < close).cast(pl.Float64).fill_null(0.0)
        for i in range(1, 31)
    ]
    expr = (
        pl.when(close.is_null()).then(None)
        .otherwise(pl.sum_horizontal(parts) / 30.0)
        .alias("winner_ratio")
    )
    return df.select(expr)["winner_ratio"]


def factor_efficiency_ratio(df: pl.DataFrame) -> pl.Series:
    """Kaufman 价格效率系数 = |净变化| / Σ|逐日变化| (20d). hi: 趋势强."""
    num = pl.col("close").diff(20).abs().over("ts_code")
    den = pl.col("close").diff(1).abs().rolling_sum(20).over("ts_code")
    return df.select((num / (den + 1e-12)).alias("efficiency_ratio"))["efficiency_ratio"]


def factor_fractal_dimension(df: pl.DataFrame) -> pl.Series:
    """分形维度 (路径长度法, wukan1986/ta_cn). lo: 趋势明确."""
    w = df.with_columns(
        (pl.col("high").rolling_max(20).over("ts_code")
         - pl.col("low").rolling_min(20).over("ts_code")).alias("_h_range"))
    w = w.with_columns(
        ((1.0 / 20) ** 2 + (pl.col("close").diff(1).over("ts_code") / (pl.col("_h_range") + 1e-9)) ** 2)
        .sqrt().alias("_leg"))
    w = w.with_columns(pl.col("_leg").rolling_sum(20).over("ts_code").alias("_L"))
    w = w.with_columns(
        (1 + (pl.col("_L").log() + np.log(2)) / np.log(40.0)).alias("fractal_dimension"))
    return w["fractal_dimension"]


def factor_alpha191_040(df: pl.DataFrame) -> pl.Series:
    """alpha191_040: 26d 上涨/下跌成交量比. hi: 上涨放量."""
    w = df.with_columns([
        (pl.col("vol") * (pl.col("close") > pl.col("close").shift(1).over("ts_code"))
         .cast(pl.Float64)).alias("_up"),
        (pl.col("vol") * (pl.col("close") <= pl.col("close").shift(1).over("ts_code"))
         .cast(pl.Float64)).alias("_dn"),
    ])
    w = w.with_columns([
        pl.col("_up").rolling_sum(26).over("ts_code").alias("_ups"),
        pl.col("_dn").rolling_sum(26).over("ts_code").alias("_dns"),
    ])
    w = w.with_columns((pl.col("_ups") / (pl.col("_dns") + 1e-9)).alias("alpha191_040"))
    return w["alpha191_040"]


def factor_alpha191_095(df: pl.DataFrame) -> pl.Series:
    """alpha191_095: 成交额 20d 标准差 (异动)."""
    return df.select(pl.col("amount").rolling_std(20).over("ts_code").alias("alpha191_095"))["alpha191_095"]


def factor_mom12m_jt(df: pl.DataFrame) -> pl.Series:
    """12-1 月动量 (Jegadeesh-Titman 1993): close[t-21]/close[t-252] - 1."""
    expr = (pl.col("close").shift(21).over("ts_code")
            / pl.col("close").shift(252).over("ts_code") - 1.0).alias("mom12m_jt")
    return df.select(expr)["mom12m_jt"]


def factor_maxret_bcw(df: pl.DataFrame) -> pl.Series:
    """21d 最大日收益, lag 1 (Bali-Cakici-Whitelaw 2011). 反向."""
    w = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    w = w.with_columns(pl.col("_ret_lag").rolling_max(21).over("ts_code").alias("maxret_bcw"))
    return w["maxret_bcw"]


def factor_accruals_sloan(df: pl.DataFrame) -> pl.Series:
    """应计项目代理 (Sloan 1996): bps 同比变化(%) - netprofit_yoy(%). 反向.
    注意: netprofit_yoy 是百分数 (12.17 = +12.17%), bps_chg 需 ×100 统一单位."""
    expr = ((pl.col("bps") / pl.col("bps").shift(252).over("ts_code") - 1.0) * 100.0
            - pl.col("netprofit_yoy")).alias("accruals_sloan")
    return df.select(expr)["accruals_sloan"]


def factor_idiovola(df: pl.DataFrame) -> pl.Series:
    """特质波动率代理 (Campbell et al. 2001): |日收益| 60d std, lag 1. 反向."""
    w = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).abs().alias("_absret"))
    w = w.with_columns(pl.col("_absret").shift(1).over("ts_code").alias("_absret_lag"))
    w = w.with_columns(pl.col("_absret_lag").rolling_std(60).over("ts_code").alias("idiovola_clmx"))
    return w["idiovola_clmx"]


def factor_gp_novymarx(df: pl.DataFrame) -> pl.Series:
    """毛利率 (Novy-Marx 2013). 正向."""
    return df.select(pl.col("grossprofit_margin").alias("gp_novymarx"))["gp_novymarx"]


def factor_overnight_intraday(df: pl.DataFrame) -> pl.Series:
    """日内-隔夜收益差 (Lou-Polk-Skouras 2019): close/open - open/prev_close."""
    expr = (pl.col("close") / pl.col("open")
            - pl.col("open") / pl.col("close").shift(1).over("ts_code")).alias("overnight_intraday_spread")
    return df.select(expr)["overnight_intraday_spread"]


def factor_skew21(df: pl.DataFrame) -> pl.Series:
    """21d 收益偏度, lag 1 (Bali-Brown-Murray 2017 彩票代理). 反向."""
    w = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    w = w.with_columns(pl.col("_ret_lag").rolling_skew(21).over("ts_code").alias("skew21_lottery"))
    return w["skew21_lottery"]


def factor_pvcorr_21(df: pl.DataFrame) -> pl.Series:
    """价量相关 21d (Chordia-Subrahmanian 2010). 手工协方差分解 (polars 无 rolling_corr)."""
    w = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret"),
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
    w = w.with_columns(
        ((pl.col("_exy") - pl.col("_ex") * pl.col("_ey"))
         / (pl.col("_sx") * pl.col("_sy") + 1e-12)).alias("pvcorr_21"))
    return w["pvcorr_21"]


def factor_kurt21(df: pl.DataFrame) -> pl.Series:
    """21d 收益峰度, lag 1 (Fang-Lai 1997). 反向."""
    w = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret"))
    w = w.with_columns(pl.col("_ret").shift(1).over("ts_code").alias("_ret_lag"))
    w = w.with_columns(pl.col("_ret_lag").rolling_kurtosis(21).over("ts_code").alias("kurt21_returns"))
    return w["kurt21_returns"]


def factor_coskew60(df: pl.DataFrame) -> pl.Series:
    """协同偏度 60d (Kraus-Litzenberger 1976). 需 df 已含 _mkt_ret 列."""
    w = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("_ret"),
        (pl.col("_mkt_ret") ** 2).alias("_m2"),
    ])
    w = w.with_columns((pl.col("_ret") * pl.col("_m2")).alias("_rm2"))
    w = w.with_columns([
        pl.col("_rm2").rolling_mean(60).over("ts_code").alias("_erm2"),
        pl.col("_m2").rolling_mean(60).over("ts_code").alias("_em2"),
    ])
    w = w.with_columns((pl.col("_erm2") / (pl.col("_em2") + 1e-9)).alias("coskew60"))
    return w["coskew60"]


def factor_hl_52w_disposition(df: pl.DataFrame) -> pl.Series:
    """52 周高低位置 (Grinblatt-Han 2005). 反向."""
    expr = (
        (pl.col("close") - pl.col("low").rolling_min(252).over("ts_code"))
        / (pl.col("high").rolling_max(252).over("ts_code")
           - pl.col("low").rolling_min(252).over("ts_code") + 1e-9)
    ).alias("hl_52w_disposition")
    return df.select(expr)["hl_52w_disposition"]


def factor_resmom_6m(df: pl.DataFrame) -> pl.Series:
    """6m 残差动量 (Blitz-Huij-Martens 2011). 需 df 已含 _mkt_ret6 列."""
    expr = (
        (pl.col("close").shift(21).over("ts_code") / pl.col("close").shift(126).over("ts_code") - 1.0)
        - pl.col("_mkt_ret6")
    ).alias("resmom_6m")
    return df.select(expr)["resmom_6m"]


NEW_FACTOR_REGISTRY = {
    # Task 1 (5)
    "winner_ratio": factor_winner_ratio,
    "efficiency_ratio": factor_efficiency_ratio,
    "fractal_dimension": factor_fractal_dimension,
    "alpha191_040": factor_alpha191_040,
    "alpha191_095": factor_alpha191_095,
    # Task 3 (12)
    "mom12m_jt": factor_mom12m_jt,
    "maxret_bcw": factor_maxret_bcw,
    "accruals_sloan": factor_accruals_sloan,
    "idiovola_clmx": factor_idiovola,
    "gp_novymarx": factor_gp_novymarx,
    "overnight_intraday_spread": factor_overnight_intraday,
    "skew21_lottery": factor_skew21,
    "pvcorr_21": factor_pvcorr_21,
    "kurt21_returns": factor_kurt21,
    "coskew60": factor_coskew60,
    "hl_52w_disposition": factor_hl_52w_disposition,
    "resmom_6m": factor_resmom_6m,
}


def add_market_cols(df: pl.DataFrame) -> pl.DataFrame:
    """预计算等权市场收益 (replace_strict, 字符串日期键, 无 join 乱序风险)."""
    mkt = (df.group_by("trade_date").agg(pl.col("close").mean().alias("_mc")).sort("trade_date")
           .with_columns([
               (pl.col("_mc") / pl.col("_mc").shift(1) - 1.0).alias("_mkt_ret"),
               (pl.col("_mc").shift(21) / pl.col("_mc").shift(126) - 1.0).alias("_mkt_ret6"),
           ])
           .with_columns(pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("_dstr")))
    rmap = dict(zip(mkt["_dstr"].to_list(), mkt["_mkt_ret"].to_list()))
    r6map = dict(zip(mkt["_dstr"].to_list(), mkt["_mkt_ret6"].to_list()))
    return df.with_columns([
        pl.col("trade_date").dt.strftime("%Y-%m-%d").replace_strict(rmap, default=None).alias("_mkt_ret"),
        pl.col("trade_date").dt.strftime("%Y-%m-%d").replace_strict(r6map, default=None).alias("_mkt_ret6"),
    ])


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 80)
    log("V33 panel v2 — 集成 17 个新因子")
    log("=" * 80)

    # 1. 读 V32, 保行序
    log(f"reading {PANEL_IN.name}...")
    v32 = pl.read_parquet(PANEL_IN)
    log(f"V32: {v32.shape}")
    v32 = v32.with_row_index("_ri")
    first_before = v32.head(3).select(["ts_code", "trade_date"]).to_dicts()

    # 2. 排序 + 市场列
    log("sorting + market cols...")
    v32s = v32.sort(["ts_code", "trade_date"])
    v32s = add_market_cols(v32s)

    # 3. 17 因子
    log("\n=== 计算 17 个新因子 ===")
    new_exprs = []
    for fid, fn in NEW_FACTOR_REGISTRY.items():
        t0 = time.time()
        s = fn(v32s)
        new_exprs.append(s)
        log(f"  {fid:28s} {time.time()-t0:5.1f}s  nulls={s.null_count()}")

    v33s = v32s.with_columns(new_exprs)
    log(f"\nv33 sorted shape: {v33s.shape}")

    # 4. sanitize: inf/nan → null, clip ±1e6
    log("sanitizing (inf/nan → null, clip)...")
    v33s = v33s.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None)
        .otherwise(pl.col(c).clip(-1e6, 1e6)).alias(c)
        for c in NEW_FACTOR_REGISTRY
    ])

    # 5. 还原原行序
    log("restoring original row order...")
    v33 = v33s.sort("_ri").drop("_ri")
    first_after = v33.head(3).select(["ts_code", "trade_date"]).to_dicts()
    assert first_before == first_after, "row order restore FAILED"
    log("✅ 行序还原一致")

    # 6. 自检
    log("\n=== 自检 ===")
    assert v33.shape[1] == 441 + 17, f"expected 458 cols, got {v33.shape[1]}"
    log(f"✅ 列数 = {v33.shape[1]} (441 + 17)")
    for fid in NEW_FACTOR_REGISTRY:
        assert fid in v33.columns, f"missing: {fid}"
    log("✅ 17 新因子列全部就位")
    assert v33.shape[0] == 1_298_335
    log(f"✅ 行数 = {v33.shape[0]:,}")

    # null 分布
    log("\n=== 新因子 null 分布 ===")
    null_stats = {}
    for fid in NEW_FACTOR_REGISTRY:
        n_null = v33[fid].null_count()
        pct = n_null / v33.shape[0] * 100
        null_stats[fid] = pct
        log(f"  {fid:28s}: null {n_null:8,d} ({pct:5.1f}%)")

    # 7. 独立 pandas 复算 (单股全序列, 3 个因子)
    log("\n=== 独立 pandas 复算对照 (000001.SZ 全序列, 最后 200 行) ===")
    import pandas as pd
    one = (v33.filter(pl.col("ts_code") == "000001.SZ").sort("trade_date")
           .select(["trade_date", "high", "low", "close", "amount",
                    "efficiency_ratio", "alpha191_095", "hl_52w_disposition", "winner_ratio"])
           .to_pandas())
    checks = {
        "efficiency_ratio": (
            one["close"].diff(20).abs()
            / (one["close"].diff().abs().rolling(20).sum() + 1e-12)
        ).values,
        "alpha191_095": one["amount"].rolling(20).std().values,
        "hl_52w_disposition": (
            (one["close"] - one["low"].rolling(252).min())
            / (one["high"].rolling(252).max() - one["low"].rolling(252).min() + 1e-9)
        ).values,
        "winner_ratio": np.array([
            np.mean([(one["close"].iloc[t - i] < one["close"].iloc[t]) for i in range(1, 31)])
            if t >= 30 else np.nan for t in range(len(one))
        ]),
    }
    verify = {}
    for fid, pd_vals in checks.items():
        pl_vals = one[fid].values
        tail = slice(-200, None)
        pd_t, pl_t = pd_vals[tail], pl_vals[tail]
        mask = ~(pd.isna(pd_t) | pd.isna(pl_t))
        if mask.sum() > 0:
            rel = np.abs(pd_t[mask].astype(float) - pl_t[mask].astype(float)) / (np.abs(pd_t[mask].astype(float)) + 1e-9)
            max_err = float(rel.max())
            verify[fid] = max_err
            log(f"  {fid:22s}: max rel err = {max_err:.2e} (n={int(mask.sum())})")
        else:
            log(f"  {fid:22s}: no valid overlap")

    # 8. 写盘
    log(f"\nwriting {PANEL_OUT.name}...")
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    v33.write_parquet(PANEL_OUT, compression="zstd")
    size_mb = PANEL_OUT.stat().st_size / 1e6
    log(f"✅ saved: {PANEL_OUT} ({size_mb:.1f} MB)")

    # 9. 报告
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report = f"""# V33 Panel 新因子集成报告 (v2)

**日期**: 2026-10-03
**升级路径**: V32 (441 cols) + 17 新因子 = **V33 ({v33.shape[1]} cols)**
**脚本**: examples/v33_add_new_factors_v2.py (Expr-style, v1 的 Series.over bug 已修)

## 新增 17 因子

| # | 因子 | 来源 | 块 |
|---:|---|---|---|
| 1 | winner_ratio | wukan1986/ta_cn | Task1 |
| 2 | efficiency_ratio | wukan1986/ta_cn | Task1 |
| 3 | fractal_dimension | wukan1986/ta_cn | Task1 |
| 4 | alpha191_040 | Daic115 | Task1 |
| 5 | alpha191_095 | Daic115 | Task1 |
| 6 | mom12m_jt | Jegadeesh-Titman 1993 | Task3 |
| 7 | maxret_bcw | Bali-Cakici-Whitelaw 2011 | Task3 |
| 8 | accruals_sloan | Sloan 1996 | Task3 |
| 9 | idiovola_clmx | Campbell-Lettau 2001 | Task3 |
| 10 | gp_novymarx | Novy-Marx 2013 | Task3 |
| 11 | overnight_intraday_spread | Lou-Polk-Skouras 2019 | Task3 |
| 12 | skew21_lottery | Bali-Brown-Murray 2017 | Task3 |
| 13 | pvcorr_21 | Chordia-Subrahmanian 2010 | Task3 |
| 14 | kurt21_returns | Fang-Lai 1997 | Task3 |
| 15 | coskew60 | Kraus-Litzenberger 1976 | Task3 |
| 16 | hl_52w_disposition | Grinblatt-Han 2005 | Task3 |
| 17 | resmom_6m | Blitz-Huij-Martens 2011 | Task3 |

## null 分布

| 因子 | null% |
|---|---:|
"""
    for fid, pct in null_stats.items():
        report += f"| {fid} | {pct:.1f}% |\n"
    report += "\n## 独立 pandas 复算对照 (max rel err)\n\n| 因子 | max rel err |\n|---|---:|\n"
    for fid, err in verify.items():
        report += f"| {fid} | {err:.2e} |\n"
    report += """
## 已知限制

1. winner_ratio 是简化版 (30d percentile 近似, 非完整 chip 累积)
2. coskew60/resmom_6m 用等权市场均值 (非市值加权)
3. accruals_sloan 用 bps 变化代理 (完整版需 WC accruals 分解)
4. 全程零远程依赖 (只读本地 parquet)
"""
    with open(EVIDENCE / "new_factors_selfcheck.md", "w") as f:
        f.write(report)
    log(f"✅ 自检报告: {EVIDENCE}/new_factors_selfcheck.md")
    log("\nDONE")


if __name__ == "__main__":
    main()
