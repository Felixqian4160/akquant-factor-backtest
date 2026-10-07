"""V35 picks: rolling IC top-K factor subset voting (per rebalance).

Contract:
- Panel: V33 (410 factors)
- Window: 2010-01-04 ~ 2025-12-31
- IC: 60-day Spearman, causal lag = 21 (same as V34)
- Daily factor subset: each rebal date picks the |IC| top-K factors among the
  factors that have a finite causal IC score on that date.
- Voting: same as V32; each factor picks top-10 stocks by direction; min_votes=3,
  min_stocks=5, max_stocks=20.
- Reuse the shared IC artifact at evidence/v34_ic_voting_2010_2025_correct/.

CLI:
  --top-k N   factor subset size per rebal date (default 30)
  --tag TAG   output sub-directory (e.g. K10, K30, K50, K100)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

PICKS_ROOT = Path("evidence/v34_ic_voting_2010_2025_correct")
SCORES_PATH = PICKS_ROOT / "ic_scores.csv"
MANIFEST_PATH = PICKS_ROOT / "factor_manifest.json"
OUT_BASE = Path("evidence/v35_topk_picks")

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
K_NOM = 10
MIN_VOTES = 3
MIN_STOCKS = 5
MAX_STOCKS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--tag", default="K30")
    args = parser.parse_args()

    OUT_DIR = OUT_BASE / args.tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"V35 picks [{args.tag}]: top-K={args.top_k} factors/day, MIN_VOTES={MIN_VOTES}, "
        f"{MIN_STOCKS}-{MAX_STOCKS} stocks")

    manifest = json.loads(MANIFEST_PATH.read_text())
    factors = manifest["factors"]
    log(f"declared factors: {len(factors)}")

    dates = (
        pl.scan_parquet(Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet"))
        .select("trade_date")
        .filter((pl.col("trade_date") >= pl.lit(START).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
                & (pl.col("trade_date") <= pl.lit(END).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")))
        .unique()
        .sort("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    rebal_dates = dates[::REBAL_STEP]
    log(f"trading dates: {len(dates)}, rebalances: {len(rebal_dates)}")

    scores = pl.read_csv(SCORES_PATH)
    score_map: dict[str, dict[str, float]] = {}
    for row in scores.iter_rows(named=True):
        score_map.setdefault(row["signal_date"], {})[row["factor"]] = row["causal_ic_score"]

    date_values = [pd.Timestamp(r) for r in rebal_dates]
    date_series = pl.Series("_dates", date_values, dtype=pl.Datetime("ms"))
    log("loading slim panel for rebal dates...")
    t0 = time.time()
    panel_days = (
        pl.scan_parquet(Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet"))
        .select(["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(date_series))
        .collect()
    )
    log(f"slim panel: {panel_days.shape} ({time.time()-t0:.1f}s)")

    picks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict] = {}
    t0 = time.time()
    for idx, rd in enumerate(rebal_dates, start=1):
        d = pd.Timestamp(rd).date().isoformat()
        factor_scores = score_map.get(d, {})
        if not factor_scores:
            continue
        ranked = sorted(factor_scores.items(), key=lambda kv: -abs(kv[1]))
        active = [f for f, _ in ranked[:args.top_k]]
        day = panel_days.filter(pl.col("trade_date") == rd)
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
            "n_factors_with_causal_score": len(factor_scores),
            "n_active_factors_topk": len(active),
            "n_picks": len(selected),
            "max_votes": int(votes.max()),
            "n_at_or_above_threshold": int((votes >= MIN_VOTES).sum()),
        }
        if idx == 1 or idx % 25 == 0:
            log(f"  picks {idx}/{len(rebal_dates)}, date={d}, active={len(active)}, selected={len(selected)}, max_votes={votes.max()}")

    (OUT_DIR / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (OUT_DIR / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))
    summary = {
        "tag": args.tag,
        "panel": "data/wavehunter_hs300_v33_with_new_factors_20261003.parquet",
        "window": f"{START} ~ {END}",
        "factor_count_declared": len(factors),
        "top_k": args.top_k,
        "ic_window": 60,
        "causal_lag": 21,
        "fwd_contract": "T+1 raw open -> T+21 close",
        "rebalance_step": REBAL_STEP,
        "k_nom": K_NOM,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "min_n_stocks": int(min((len(p) for p in picks.values()), default=0)),
        "max_n_stocks_observed": int(max((len(p) for p in picks.values()), default=0)),
        "runtime_sec": time.time() - t0,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"saved {OUT_DIR}: {len(picks)} dates, avg={summary['avg_n_stocks']:.2f}, "
        f"range={summary['min_n_stocks']}-{summary['max_n_stocks_observed']}")


if __name__ == "__main__":
    main()