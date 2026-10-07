"""V37 picks: V14-style rolling-IC voting + bear router (idx_ret_60d / idx_ret_20d).

合同:
- Panel: V33, 410 选股因子
- 2010-01-04 ~ 2025-12-31
- 因果 IC: 60 日 Spearman rolling, lag = 21
- 调仓: 20 个交易日
- 每日 top-10 |IC| 因子 (TOP_K)
- 每因子提名 top-10 股票, 票数 >= 2, 股票 5-10 只
- Bear router: idx_ret_60d <= -5% AND idx_ret_20d <= 0 -> bear, AKQuant 平仓
- 复用 V34 IC artifact (evidence/v34_ic_voting_2010_2025_correct/ic_scores.csv)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
OUT_ROOT = Path("evidence/v37_v14_causal_v27balance_20261006")

START = "2010-01-01"
END = "2025-12-31"
IC_START = "2009-08-01"
REBAL_STEP = 20
FWD_DAYS = 20
ENTRY_TO_EXIT_SHIFT = 21
CAUSAL_LAG = 21
IC_WINDOW = 60
MIN_IC_OBS = 30
K_NOM = 10
TOP_K = 10
MIN_VOTES = 2
MIN_STOCKS = 5
MAX_STOCKS = 10
ROUND_TRIP_COST = 0.005
# V27_BALANCE router: 5-day cumulative V2/16 signals >= 5
BEAR_VOTE_THRESHOLD = 2
BEAR_CUMULATIVE_THRESHOLD = 5
BEAR_LOOKBACK = 5
# V19 factor-return voting (bear side)
BEAR_FACTOR_LOOKBACK_SESSIONS = 30
BEAR_TOP_FACTORS = 10
BEAR_TOP_STOCKS_PER_FACTOR = 10
BEAR_MIN_VOTES = 3
BEAR_MIN_STOCKS = 5
BEAR_MAX_STOCKS = 10
BEAR_MIN_FACTOR_MEAN_RETURN = 0.005
BEAR_FWD_EXIT_SHIFT = 21

ACADEMIC22 = [
    "l_size", "l_size3", "l_turnm", "l_turna", "l_ami", "l_dtvm", "l_dtva",
    "l_vdtv", "r_tv", "r_beta", "p_m1", "p_m3", "p_m6", "p_m11", "p_m24",
    "p_mchg", "p_52w", "p_mdr", "p_pr", "p_season", "v_bm", "v_ep",
]
NEW17 = [
    "winner_ratio", "efficiency_ratio", "fractal_dimension", "alpha191_040",
    "alpha191_095", "mom12m_jt", "maxret_bcw", "accruals_sloan", "idiovola_clmx",
    "gp_novymarx", "overnight_intraday_spread", "skew21_lottery", "pvcorr_21",
    "kurt21_returns", "coskew60", "hl_52w_disposition", "resmom_6m",
]
NON_FACTOR_EXACT = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "volume", "amount",
    "adj_factor", "pct_chg", "cap", "circ_cap", "turnover_rate", "pe", "pb", "bps",
    "roe", "roe_waa", "roa", "grossprofit_margin", "netprofit_margin", "current_ratio",
    "quick_ratio", "debt_to_assets", "assets_turn", "fcff", "cfps", "ocfps",
    "netprofit_yoy", "or_yoy", "ocf_yoy", "adj_close", "idx_close",
}
NON_FACTOR_PREFIXES = ("v10_1_", "idx_")
KNOWN_ALL_NULL = {"talib_MAX2", "talib_MIN2"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dt_expr(value: str) -> pl.Expr:
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def factor_manifest() -> dict:
    cols = pl.scan_parquet(PANEL).collect_schema().names()
    gtja = sorted(c for c in cols if c.startswith("gtja_"))
    alpha = sorted(c for c in cols if c.startswith("alpha_"))
    talib = sorted(c for c in cols if c.startswith("talib_"))
    academic = [c for c in ACADEMIC22 if c in cols]
    new17 = [c for c in NEW17 if c in cols]
    factors = list(dict.fromkeys(gtja + alpha + talib + academic + new17))
    return {
        "panel": str(PANEL),
        "factor_count_declared": len(factors),
        "factor_count_expected": 410,
        "factor_count_matches_expected": len(factors) == 410,
        "groups": {
            "gtja": gtja, "alpha": alpha, "talib": talib,
            "academic22": academic, "new17": new17,
        },
        "missing_expected": [c for c in ACADEMIC22 + NEW17 if c not in cols],
        "factors": factors,
        "known_all_null": sorted(c for c in KNOWN_ALL_NULL if c in factors),
    }


def load_net20() -> pl.DataFrame:
    base = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"])
        .sort(["ts_code", "trade_date"])
        .collect()
    )
    return (
        base
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
            pl.col("close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
        ])
        .with_columns(
            (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("net20")
        )
        .select(["trade_date", "ts_code", "net20"])
        .filter(pl.col("net20").is_not_null() & pl.col("net20").is_finite())
    )


def trading_dates() -> list[str]:
    dates = (
        pl.scan_parquet(PANEL)
        .select("trade_date")
        .filter((pl.col("trade_date") >= dt_expr(START)) & (pl.col("trade_date") <= dt_expr(END)))
        .unique()
        .sort("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    return [pd.Timestamp(d).date().isoformat() for d in dates]


def compute_causal_scores(
    manifest: dict,
    out_dir: Path,
    max_rebalances: int | None = None,
) -> pl.DataFrame:
    factors = manifest["factors"]
    dates = trading_dates()
    rebal_dates = dates[::REBAL_STEP]
    if max_rebalances is not None:
        rebal_dates = rebal_dates[:max_rebalances]
    net20 = load_net20()
    net20 = net20.filter(
        (pl.col("trade_date") >= dt_expr(IC_START)) & (pl.col("trade_date") <= dt_expr(END))
    )

    log(f"Declared factors: {len(factors)} (expected 410)")
    log(f"Trading dates: {len(dates)}, rebalances: {len(rebal_dates)}")
    log(f"IC contract: Spearman, window={IC_WINDOW}, causal_lag={CAUSAL_LAG}, min_obs={MIN_IC_OBS}")

    score_rows: list[tuple[str, str, float]] = []
    batch_size = 50
    t0 = time.time()

    for start in range(0, len(factors), batch_size):
        batch = factors[start:start + batch_size]
        work = (
            pl.scan_parquet(PANEL)
            .select(["trade_date", "ts_code"] + batch)
            .filter(
                (pl.col("trade_date") >= dt_expr(IC_START))
                & (pl.col("trade_date") <= dt_expr(END))
            )
            .collect()
            .join(net20, on=["trade_date", "ts_code"], how="left")
        )
        ic_wide = (
            work.group_by("trade_date")
            .agg([
                pl.corr(pl.col(c), pl.col("net20"), method="spearman").alias(c)
                for c in batch
            ])
            .sort("trade_date")
        )
        raw_dates = np.array(
            [np.datetime64(pd.Timestamp(d).date(), "D") for d in ic_wide["trade_date"].to_list()]
        )
        raw_values = ic_wide.select(batch).to_numpy()

        for rd in rebal_dates:
            rd_np = np.datetime64(rd, "D")
            count_le = int(np.searchsorted(raw_dates, rd_np, side="right"))
            end_exclusive = count_le - CAUSAL_LAG
            if end_exclusive < MIN_IC_OBS:
                continue
            begin = max(0, end_exclusive - IC_WINDOW)
            window = raw_values[begin:end_exclusive]
            with np.errstate(invalid="ignore", divide="ignore"):
                scores = np.nanmean(window, axis=0)
            for factor, score in zip(batch, scores):
                if np.isfinite(score):
                    score_rows.append((rd, factor, float(score)))
        if (start // batch_size) % 5 == 0:
            log(f"  IC batch {min(start + batch_size, len(factors))}/{len(factors)}; "
                f"scores={len(score_rows):,}; elapsed={time.time() - t0:.1f}s")
        del work, ic_wide, raw_values

    scores = pl.DataFrame(
        score_rows,
        schema={"signal_date": pl.String, "factor": pl.String, "causal_ic_score": pl.Float64},
        orient="row",
    )
    if scores.height:
        scores = scores.with_columns(
            pl.col("causal_ic_score").abs().rank(descending=True, method="ordinal")
            .over("signal_date").alias("abs_score_rank")
        ).sort(["signal_date", "abs_score_rank", "factor"])
    scores.write_csv(out_dir / "ic_scores.csv")
    log(f"IC scores saved: {scores.height:,} rows")
    return scores


def load_rebalance_panel(factors: list[str], rebal_dates: list[str]) -> pl.DataFrame:
    date_values = [datetime.strptime(d, "%Y-%m-%d") for d in rebal_dates]
    date_series = pl.Series("_dates", date_values, dtype=pl.Datetime("ms"))
    return (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(date_series))
        .collect()
    )


def compute_regime_per_rebal(rebal_dates: list[str]) -> dict:
    """V27_BALANCE router: 5-day cumulative V2/16 bear count >= 5.

    16 单日信号 (close vs MAs / ret signs / distance / slope / drawdown).
    V2 flag 当日票数 >= 2.
    10 日滚动累计 >= 4 → bear.
    """
    daily = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "idx_close"])
        .filter(pl.col("idx_close").is_not_null())
        .unique("trade_date")
        .sort("trade_date")
        .collect()
    )
    if daily.height == 0:
        return {d: {"regime": "bull_neutral"} for d in rebal_dates}

    daily = daily.with_columns([
        pl.col("idx_close").rolling_mean(window_size=5).alias("ma5"),
        pl.col("idx_close").rolling_mean(window_size=10).alias("ma10"),
        pl.col("idx_close").rolling_mean(window_size=20).alias("ma20"),
        pl.col("idx_close").rolling_mean(window_size=60).alias("ma60"),
        pl.col("idx_close").rolling_mean(window_size=120).alias("ma120"),
        pl.col("idx_close").pct_change(10).alias("ret10"),
        pl.col("idx_close").pct_change(20).alias("ret20"),
        pl.col("idx_close").pct_change(40).alias("ret40"),
        pl.col("idx_close").pct_change(60).alias("ret60"),
    ]).with_columns([
        ((pl.col("idx_close") - pl.col("ma20")) / pl.col("ma20")).alias("dist_ma20"),
        ((pl.col("idx_close") - pl.col("ma60")) / pl.col("ma60")).alias("dist_ma60"),
        pl.col("ma60").diff().alias("slope_ma60"),
        pl.col("idx_close").rolling_max(window_size=60).alias("high_60d"),
        pl.col("idx_close").rolling_max(window_size=120).alias("high_120d"),
    ]).with_columns([
        (pl.col("idx_close") / pl.col("high_60d") - 1).alias("dd_60d"),
        (pl.col("idx_close") / pl.col("high_120d") - 1).alias("dd_120d"),
    ])

    BEAR_SIGNAL_SPECS = [
        ("close_lt_MA5", lambda df: df["idx_close"] < df["ma5"]),
        ("close_lt_MA10", lambda df: df["idx_close"] < df["ma10"]),
        ("close_lt_MA20", lambda df: df["idx_close"] < df["ma20"]),
        ("close_lt_MA60", lambda df: df["idx_close"] < df["ma60"]),
        ("close_lt_MA120", lambda df: df["idx_close"] < df["ma120"]),
        ("MA20_lt_MA60", lambda df: df["ma20"] < df["ma60"]),
        ("MA60_lt_MA120", lambda df: df["ma60"] < df["ma120"]),
        ("ret10_lt_0", lambda df: df["ret10"] < 0),
        ("ret20_lt_0", lambda df: df["ret20"] < 0),
        ("ret40_lt_0", lambda df: df["ret40"] < 0),
        ("ret60_lt_0", lambda df: df["ret60"] < 0),
        ("dist_ma20_lt_neg2", lambda df: df["dist_ma20"] < -0.02),
        ("dist_ma60_lt_neg5", lambda df: df["dist_ma60"] < -0.05),
        ("slope_ma60_lt_0", lambda df: df["slope_ma60"] < 0),
        ("dd_60d_lt_neg5", lambda df: df["dd_60d"] < -0.05),
        ("dd_120d_lt_neg10", lambda df: df["dd_120d"] < -0.10),
    ]
    BEAR_VOTE_THRESHOLD = 2
    BEAR_LOOKBACK = 5
    BEAR_CUMULATIVE_THRESHOLD = 5

    pdf = daily.to_pandas()
    sig_count = pd.DataFrame()
    for name, fn in BEAR_SIGNAL_SPECS:
        sig_count[name] = fn(pdf).fillna(False)
    pdf["bear_vote_count"] = sig_count.sum(axis=1)
    pdf["bear_v2_flag"] = (pdf["bear_vote_count"] >= BEAR_VOTE_THRESHOLD).astype(int)
    pdf["bear_cumulative"] = pdf["bear_v2_flag"].rolling(BEAR_LOOKBACK).sum()
    pdf["is_bear"] = pdf["bear_cumulative"] >= BEAR_CUMULATIVE_THRESHOLD

    date_to_regime = {
        pd.Timestamp(d).date().isoformat(): "bear" if v else "bull_neutral"
        for d, v in zip(pdf["trade_date"], pdf["is_bear"])
    }
    out = {}
    for d in rebal_dates:
        out[d] = {"regime": date_to_regime.get(d, "bull_neutral")}
    return out


def collect_bear_sessions(panel, regime_map, rebal_dates):
    """Per regime_map, collect all bear-flagged trading dates for V19 lookback."""
    bear_dates = sorted([d for d, info in regime_map.items() if info["regime"] == "bear"])
    return bear_dates


def compute_bear_factor_returns(
    panel: pl.DataFrame,
    factors: list[str],
    bear_session_dates: list[str],
) -> dict:
    """Per factor, mean net return over completed bear sessions whose exit_date <= current.
    panel: polars DataFrame with columns trade_date, ts_code, _fwd_net, and all factors.
    Output: {bear_date_iso: {factor: mean_net_return}}.
    """
    if not bear_session_dates:
        return {}
    base_pd = panel.select(["trade_date", "ts_code", "_fwd_net"] + factors).to_pandas()
    base_pd["trade_date"] = pd.to_datetime(base_pd["trade_date"])

    out: dict = {d: {} for d in bear_session_dates}
    for cur in bear_session_dates:
        cur_ts = pd.Timestamp(cur).to_pydatetime()
        sub = base_pd[base_pd["trade_date"] <= cur_ts]
        if sub.empty:
            continue
        per_factor = {}
        fwd = sub["_fwd_net"]
        valid = fwd.notna()
        for f in factors:
            col = pd.to_numeric(sub[f], errors="coerce")
            ok = col.notna() & valid
            if ok.sum() < 5:
                continue
            per_factor[f] = float(fwd[ok].mean())
        out[cur] = per_factor
    return out


def build_picks(
    manifest: dict,
    scores: pl.DataFrame,
    tag: str,
    rebal_offset: int = 0,
    max_rebalances: int | None = None,
) -> dict:
    factors = manifest["factors"]
    all_dates = trading_dates()
    rebal_dates = all_dates[rebal_offset::REBAL_STEP]
    if max_rebalances is not None:
        rebal_dates = rebal_dates[:max_rebalances]
    regime_map = compute_regime_per_rebal(rebal_dates)

    score_map: dict[str, dict[str, float]] = {}
    for row in scores.iter_rows(named=True):
        score_map.setdefault(row["signal_date"], {})[row["factor"]] = row["causal_ic_score"]

    date_series = pl.Series("_dates",
                             [pd.Timestamp(d) for d in rebal_dates],
                             dtype=pl.Datetime("ms"))
    panel = pl.scan_parquet(PANEL).select(
        ["trade_date", "ts_code"] + factors
    ).filter(pl.col("trade_date").is_in(date_series)).collect()
    # _fwd_net must include all dates (not just rebal_dates) so that for an
    # early rebal_date the sub-window `trade_date <= cur_ts` has earlier history.
    # Baseline used the full 2004-2025 panel. Here we project back to 2010-01-01.
    start_dt = pd.Timestamp("2010-01-01")
    cache_path = Path("evidence/v37_v14_causal/_cache/bear_factor_returns.tsv")
    bear_factor_returns: dict = {}
    if cache_path.exists():
        # Fast path: load from cache
        cache_t0 = time.time()
        log(f"Loading bear cache from {cache_path}")
        with cache_path.open("r") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                d, f, v = parts
                if d and f and f != "_global":
                    bear_factor_returns.setdefault(d, {})[f] = float(v)
        log(f"Cache loaded: {len(bear_factor_returns)} dates in {time.time()-cache_t0:.1f}s")
    else:
        log(f"Computing _fwd_net (no cache; fallback path)")
        panel_full = (
            pl.scan_parquet(PANEL)
            .select(["trade_date", "ts_code", "open", "close"] + factors)
            .filter(pl.col("trade_date") >= start_dt)
            .sort(["ts_code", "trade_date"])
            .collect()
            .with_columns([
                pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
                pl.col("close").shift(-BEAR_FWD_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
            ])
            .with_columns(
                ((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST)
                 .alias("_fwd_net"))
            )
            .select(["trade_date", "ts_code", "_fwd_net"] + factors)
        )
        rebal_dates_for_bear = trading_dates()
        bear_factor_returns = compute_bear_factor_returns(
            panel_full, factors, rebal_dates_for_bear
        )

    picks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict] = {}
    t0 = time.time()
    bull_count = bear_count = 0

    for idx, rd in enumerate(rebal_dates, start=1):
        d = pd.Timestamp(rd).date().isoformat()
        regime = regime_map.get(d, {"regime": "bull_neutral"})["regime"]

        if regime == "bull_neutral":
            bull_count += 1
            factor_scores = score_map.get(d, {})
            if not factor_scores:
                continue
            ranked = sorted(factor_scores.items(), key=lambda kv: -abs(kv[1]))
            active = [f for f, _ in ranked[:TOP_K]]
            day = panel.filter(pl.col("trade_date") == dt_expr(d))
            if day.height == 0:
                continue
            codes = day.get_column("ts_code").to_list()
            values = day.select(active).to_numpy()
            votes = np.zeros(day.height, dtype=np.int32)

            for j, factor in enumerate(active):
                column = values[:, j]
                valid = np.flatnonzero(np.isfinite(column))
                if len(valid) < K_NOM:
                    continue
                if factor_scores[factor] >= 0:
                    chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
                else:
                    chosen = valid[np.argpartition(column[valid], K_NOM - 1)[:K_NOM]]
                votes[chosen] += 1

            order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
            selected = [i for i in order if int(votes[i]) >= MIN_VOTES]
            if len(selected) < MIN_STOCKS:
                selected = order[:MIN_STOCKS]
            selected = selected[:MAX_STOCKS]
            if not selected:
                continue
            picks[d] = {str(codes[i]): int(votes[i]) for i in selected}
            metadata[d] = {
                "regime": "bull_neutral",
                "selector": "v14_ic_voting",
                "n_active_factors": len(active),
                "n_picks": len(selected),
                "max_votes": int(votes.max()),
                "n_at_or_above_threshold": int((votes >= MIN_VOTES).sum()),
            }
        else:
            bear_count += 1
            factor_returns = bear_factor_returns.get(d, {})
            eligible = [(f, r) for f, r in factor_returns.items()
                        if r > BEAR_MIN_FACTOR_MEAN_RETURN]
            if not eligible:
                continue
            ranked = sorted(eligible, key=lambda kv: -kv[1])
            active = [f for f, _ in ranked[:BEAR_TOP_FACTORS]]
            day = panel.filter(pl.col("trade_date") == dt_expr(d))
            if day.height == 0:
                continue
            codes = day.get_column("ts_code").to_list()
            values = day.select(active).to_numpy()
            votes = np.zeros(day.height, dtype=np.int32)

            for j, factor in enumerate(active):
                column = values[:, j]
                valid = np.flatnonzero(np.isfinite(column))
                if len(valid) < K_NOM:
                    continue
                chosen = valid[np.argpartition(column[valid], -BEAR_TOP_STOCKS_PER_FACTOR)[-BEAR_TOP_STOCKS_PER_FACTOR:]]
                votes[chosen] += 1

            order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
            selected = [i for i in order if int(votes[i]) >= BEAR_MIN_VOTES]
            if len(selected) < BEAR_MIN_STOCKS:
                selected = order[:BEAR_MIN_STOCKS]
            selected = selected[:BEAR_MAX_STOCKS]
            if not selected:
                continue
            picks[d] = {str(codes[i]): int(votes[i]) for i in selected}
            metadata[d] = {
                "regime": "bear",
                "selector": "v19_factor_return_voting",
                "n_eligible_factors": len(eligible),
                "n_active_factors": len(active),
                "n_picks": len(selected),
                "max_votes": int(votes.max()),
                "n_at_or_above_threshold": int((votes >= BEAR_MIN_VOTES).sum()),
            }
        if idx == 1 or idx % 25 == 0:
            sel_n = metadata[d]["n_picks"]
            mv = metadata[d]["max_votes"]
            log(f"  picks {idx}/{len(rebal_dates)}, date={d}, regime={regime}, "
                f"selected={sel_n}, max_votes={mv}")

    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (out / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))
    summary = {
        "tag": tag,
        "panel": str(PANEL),
        "window": f"{START} ~ {END}",
        "factor_count_declared": len(factors),
        "all_factors_vote": True,
        "ic_window": IC_WINDOW,
        "causal_lag": CAUSAL_LAG,
        "fwd_contract": "T+1 raw open -> T+21 close",
        "rebalance_step": REBAL_STEP,
        "top_k": TOP_K,
        "k_nom": K_NOM,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
    "bear_router": "V27_BALANCE 5d cumulative V2/16 bear signals >= 5",
        "bear_top_factors": BEAR_TOP_FACTORS,
        "bear_min_votes": BEAR_MIN_VOTES,
        "bear_min_factor_mean_return": BEAR_MIN_FACTOR_MEAN_RETURN,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "n_bull_neutral_rebalances": bull_count,
        "n_bear_rebalances": bear_count,
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "min_n_stocks": int(min((len(p) for p in picks.values()), default=0)),
        "max_n_stocks_observed": int(max((len(p) for p in picks.values()), default=0)),
        "runtime_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"saved {out}: {len(picks)} dates "
        f"(bull_neutral={bull_count}, bear={bear_count}), "
        f"avg={summary['avg_n_stocks']:.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["picks"], default="picks")
    parser.add_argument("--tag", default="V14")
    parser.add_argument("--rebal-offset", type=int, default=0)
    parser.add_argument("--max-rebalances", type=int, default=None)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "factor_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = factor_manifest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["factor_count_matches_expected"]:
        raise RuntimeError(f"factor count contract failed: {manifest['factor_count_declared']}")

    scores = pl.read_csv("evidence/v34_ic_voting_2010_2025_correct/ic_scores.csv")
    build_picks(manifest, scores, args.tag, args.rebal_offset, args.max_rebalances)


if __name__ == "__main__":
    main()