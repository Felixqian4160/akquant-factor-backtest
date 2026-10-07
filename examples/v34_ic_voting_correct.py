"""V34 corrected: V33 panel, causal rolling IC score + all-factor voting.

Contract:
- Panel: V33 with 410 declared stock-selection factors.
- Trading window: 2010-01-01 through 2025-12-31.
- Rebalance: every 20 trading sessions.
- Forward net return used for daily IC: T+1 raw open -> T+21 close,
  less 0.5% round-trip cost.
- Causal IC score at signal date T: mean Spearman IC over the latest
  60 completed IC dates, excluding the latest 21 trading sessions.
- ALL declared factors vote. A factor's direction is high when its
  causal score is non-negative, low otherwise.
- Each factor nominates top 10 stocks. The vote threshold is supplied
  by --min-votes. A/B contracts are:
    A: min_votes=50, min_stocks=10, max_stocks=30
    B: min_votes=100, min_stocks=10, max_stocks=20

Stages:
  --stage ic     compute one shared causal IC-score artifact
  --stage picks  load the shared artifact and build one tag's picks
  --stage all    run both stages for one tag

This script only creates causal picks. AKQuant execution is a separate,
sequential step after the picks artifact passes its audit.
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
OUT_ROOT = Path("evidence/v34_ic_voting_2010_2025_correct")

START = "2010-01-01"
END = "2025-12-31"
# Buffer is used only to establish causal history before the first trade.
IC_START = "2009-08-01"
REBAL_STEP = 20
FWD_DAYS = 20
ENTRY_TO_EXIT_SHIFT = 21  # close[T+21] / open[T+1]
CAUSAL_LAG = 21            # score excludes the latest 21 IC dates
IC_WINDOW = 60
MIN_IC_OBS = 30
K_NOM = 10
ROUND_TRIP_COST = 0.005

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

# These are data/label/timing columns, not stock-selection factors.
NON_FACTOR_EXACT = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "volume", "amount",
    "adj_factor", "pct_chg", "cap", "circ_cap", "turnover_rate", "pe", "pb", "bps",
    "roe", "roe_waa", "roa", "grossprofit_margin", "netprofit_margin", "current_ratio",
    "quick_ratio", "debt_to_assets", "assets_turn", "fcff", "cfps", "ocfps",
    "netprofit_yoy", "or_yoy", "ocf_yoy", "adj_close", "idx_close",
}
NON_FACTOR_PREFIXES = ("v10_1_", "idx_")
KNOWN_ALL_NULL = {"talib_MAX2", "talib_MIN2"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dt_expr(value: str) -> pl.Expr:
    """A Datetime(ms) literal compatible with the V33 trade_date column."""
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def factor_manifest() -> dict:
    cols = pl.scan_parquet(PANEL).collect_schema().names()
    gtja = sorted(c for c in cols if c.startswith("gtja_"))
    alpha = sorted(c for c in cols if c.startswith("alpha_"))
    talib = sorted(c for c in cols if c.startswith("talib_"))
    academic = [c for c in ACADEMIC22 if c in cols]
    new17 = [c for c in NEW17 if c in cols]
    factors = list(dict.fromkeys(gtja + alpha + talib + academic + new17))
    missing_expected = [c for c in ACADEMIC22 + NEW17 if c not in cols]
    return {
        "panel": str(PANEL),
        "factor_count_declared": len(factors),
        "factor_count_expected": 410,
        "factor_count_matches_expected": len(factors) == 410,
        "groups": {
            "gtja": gtja,
            "alpha": alpha,
            "talib": talib,
            "academic22": academic,
            "new17": new17,
        },
        "missing_expected": missing_expected,
        "factors": factors,
        "known_all_null": sorted(c for c in KNOWN_ALL_NULL if c in factors),
    }


def load_net20() -> pl.DataFrame:
    """Build executable forward returns from the full stock history."""
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
    """Compute raw daily Spearman IC, then causal rolling scores at rebalance dates."""
    factors = manifest["factors"]
    dates = trading_dates()
    rebal_dates = dates[::REBAL_STEP]
    if max_rebalances is not None:
        rebal_dates = rebal_dates[:max_rebalances]
    net20 = load_net20()
    net20 = net20.filter((pl.col("trade_date") >= dt_expr(IC_START)) & (pl.col("trade_date") <= dt_expr(END)))

    log(f"Declared factors: {len(factors)} (expected 410)")
    log(f"Trading dates: {len(dates)}, rebalances: {len(rebal_dates)}")
    log(f"IC contract: Spearman, window={IC_WINDOW}, causal_lag={CAUSAL_LAG}, min_obs={MIN_IC_OBS}")

    score_rows: list[tuple[str, str, float, int]] = []
    batch_size = 50
    t0 = time.time()

    for start in range(0, len(factors), batch_size):
        batch = factors[start:start + batch_size]
        work = (
            pl.scan_parquet(PANEL)
            .select(["trade_date", "ts_code"] + batch)
            .filter((pl.col("trade_date") >= dt_expr(IC_START)) & (pl.col("trade_date") <= dt_expr(END)))
            .collect()
            .join(net20, on=["trade_date", "ts_code"], how="left")
        )

        # One grouped pass for this factor batch. Polars' spearman method is
        # rank correlation, not Pearson on raw factor values.
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
            finite_count = np.isfinite(window).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                scores = np.nanmean(window, axis=0)
            for factor, score, n_obs in zip(batch, scores, finite_count):
                if n_obs >= MIN_IC_OBS and np.isfinite(score):
                    score_rows.append((rd, factor, float(score), int(n_obs)))

        del work, ic_wide, raw_values
        log(f"IC batch {min(start + batch_size, len(factors))}/{len(factors)}; "
            f"scores={len(score_rows):,}; elapsed={time.time() - t0:.1f}s")

    scores = pl.DataFrame(
        score_rows,
        schema={
            "signal_date": pl.String,
            "factor": pl.String,
            "causal_ic_score": pl.Float64,
            "ic_obs": pl.Int64,
        },
        orient="row",
    )
    if scores.height:
        scores = scores.with_columns(
            pl.col("causal_ic_score").abs().rank(descending=True, method="ordinal")
            .over("signal_date").alias("abs_score_rank")
        ).sort(["signal_date", "abs_score_rank", "factor"])
    scores.write_csv(out_dir / "ic_scores.csv")

    scored_factors = set(scores.get_column("factor").unique().to_list()) if scores.height else set()
    manifest["factor_count_with_finite_scores"] = len(scored_factors)
    manifest["factors_without_finite_scores"] = sorted(set(factors) - scored_factors)
    (out_dir / "factor_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"IC scores saved: {scores.height:,} rows; factors with scores={len(scored_factors)}")
    return scores


def load_rebalance_panel(factors: list[str], rebal_dates: list[str]) -> pl.DataFrame:
    date_values = [datetime.strptime(d, "%Y-%m-%d") for d in rebal_dates]
    date_series = pl.Series("_rebal_dates", date_values, dtype=pl.Datetime("ms"))
    return (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(date_series))
        .collect()
    )


def build_picks(
    manifest: dict,
    scores: pl.DataFrame,
    tag: str,
    min_votes: int,
    min_stocks: int,
    max_stocks: int,
    max_rebalances: int | None = None,
) -> dict:
    factors = manifest["factors"]
    all_dates = trading_dates()
    rebal_dates = all_dates[::REBAL_STEP]
    if max_rebalances is not None:
        rebal_dates = rebal_dates[:max_rebalances]
    score_map: dict[str, dict[str, float]] = {}
    for row in scores.iter_rows(named=True):
        score_map.setdefault(row["signal_date"], {})[row["factor"]] = row["causal_ic_score"]

    panel_days = load_rebalance_panel(factors, rebal_dates)
    picks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict] = {}
    t0 = time.time()

    for idx, rd in enumerate(rebal_dates, start=1):
        day = panel_days.filter(pl.col("trade_date") == dt_expr(rd))
        factor_scores = score_map.get(rd, {})
        if day.height == 0 or not factor_scores:
            continue

        active = [f for f in factors if f in factor_scores]
        codes = day.get_column("ts_code").to_list()
        values = day.select(active).to_numpy()
        votes = np.zeros(day.height, dtype=np.int32)

        for j, factor in enumerate(active):
            column = values[:, j]
            valid = np.flatnonzero(np.isfinite(column))
            if len(valid) < K_NOM:
                continue
            k = K_NOM
            if factor_scores[factor] >= 0:
                chosen = valid[np.argpartition(column[valid], -k)[-k:]]
            else:
                chosen = valid[np.argpartition(column[valid], k - 1)[:k]]
            votes[chosen] += 1

        # Stable tie-break: more votes first, then stock code ascending.
        order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
        selected = [i for i in order if int(votes[i]) >= min_votes]
        if len(selected) < min_stocks:
            selected = order[:min_stocks]
        selected = selected[:max_stocks]
        if not selected:
            continue

        picks[rd] = {str(codes[i]): int(votes[i]) for i in selected}
        metadata[rd] = {
            "n_declared_factors": len(factors),
            "n_factors_with_causal_score": len(active),
            "n_picks": len(selected),
            "max_votes": int(votes.max()),
            "n_at_or_above_threshold": int((votes >= min_votes).sum()),
            "n_stocks_with_any_vote": int((votes > 0).sum()),
            "causal_score_date_end": rd,
        }
        if idx == 1 or idx % 25 == 0:
            log(f"{tag}: picks {idx}/{len(rebal_dates)}, date={rd}, "
                f"active_factors={len(active)}, selected={len(selected)}, max_votes={votes.max()}")

    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (out / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))
    summary = {
        "tag": tag,
        "panel": str(PANEL),
        "window": f"{START} ~ {END}",
        "factor_count_declared": len(factors),
        "factor_count_with_finite_scores": len(set(scores.get_column("factor").to_list())),
        "all_factors_vote": True,
        "ic_window": IC_WINDOW,
        "causal_lag": CAUSAL_LAG,
        "fwd_contract": "T+1 raw open -> T+21 close",
        "rebalance_step": REBAL_STEP,
        "k_nom": K_NOM,
        "min_votes": min_votes,
        "min_stocks": min_stocks,
        "max_stocks": max_stocks,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "avg_n_stocks": float(np.mean([len(x) for x in picks.values()])) if picks else 0.0,
        "min_n_stocks": int(min((len(x) for x in picks.values()), default=0)),
        "max_n_stocks_observed": int(max((len(x) for x in picks.values()), default=0)),
        "runtime_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"{tag}: picks saved to {out}; {len(picks)} dates, avg={summary['avg_n_stocks']:.2f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ic", "picks", "all"], default="all")
    parser.add_argument("--tag", choices=["A", "B"], default="A")
    parser.add_argument("--min-votes", type=int, default=50)
    parser.add_argument("--min-stocks", type=int, default=10)
    parser.add_argument("--max-stocks", type=int, default=30)
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

    shared_scores = OUT_ROOT / "ic_scores.csv"
    if args.stage in {"ic", "all"}:
        compute_causal_scores(manifest, OUT_ROOT, args.max_rebalances)
    if args.stage in {"picks", "all"}:
        if not shared_scores.exists():
            raise FileNotFoundError(f"missing shared IC artifact: {shared_scores}")
        scores = pl.read_csv(shared_scores)
        build_picks(
            manifest, scores, args.tag, args.min_votes, args.min_stocks,
            args.max_stocks, args.max_rebalances,
        )


if __name__ == "__main__":
    main()
