"""Build fsdb v4 = fsdb v3 base + recomputed factor library (412 factors).

Modes:
  --init              write the frozen input contract (pre-registration) + preflight stats
  --batch S N         compute factors[S:S+N] of the master list; save batch parquet
  --status            show which batches are done
  --assemble          assemble final v4 panel + manifest + null-rate audit

Input contract (frozen at --init):
  source      : data/wavehunter_hs300_fsdb_v3_20261009_001500.parquet
  prices      : *_hfq (continuous adjusted series; continuity verified — hfq removes
                dividend gaps, qfq amplifies them)
  volume      : /100  (shares -> lots, v34/canonical convention)
  amount      : /1000 (yuan -> thousand yuan, canonical convention)
  cap         : total_mv/1e4 (yuan -> 10k yuan); circ_cap: float_mv/1e4
  vwap        : amount(千元)*10/volume(手)*adj_factor_hfq
  returns     : close_hfq pct_change, fill_null(0)
  adv{w}      : volume(手).rolling_mean(w, min_samples=1).over(stock_code)
  industry    : sw_l2_name (also feeds sub_industry)
  removed     : accruals_sloan, gp_novymarx (deleted from factorlib 2026-10-10: require
                financial-statement inputs not available from stockdb)
  v_bm feeder : bps := close_hfq/pb  (v_bm = bps/close = 1/pb)
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"
CACHE = BASE / "factorlib_build" / "_v4_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = CACHE / "build.log"

import sys
sys.path.insert(0, str(BASE))

ADV_WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180]
# 2026-10-10: accruals_sloan / gp_novymarx were deleted from factorlib outright —
# both require financial-statement inputs (bps / netprofit_yoy / grossprofit_margin)
# that are not available from the stockdb source and will not be sourced from Tushare.
# Kept here only as a documentation record; restore the modules if a new fundamental
# data source is added.
REMOVED_FACTORS = {
    "github.accruals_sloan": "deleted 2026-10-10: requires bps+netprofit_yoy; no financial data source",
    "github.gp_novymarx": "deleted 2026-10-10: requires grossprofit_margin; no financial data source",
}

# Factors defined on the untraded (raw) price level rather than the adjusted series.
# v34 semantics: p_pr = log(raw close).
RAW_CLOSE_FACTORS = {"p_pr"}

INPUT_COLS = [
    "ts_code", "trade_date", "close", "open_hfq", "high_hfq", "low_hfq", "close_hfq",
    "adj_factor_hfq", "cum_latest", "volume", "amount", "total_mv", "float_mv", "pe_ttm", "pb",
    "sw_l2_name",
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def master_list() -> list[tuple[str, str, str]]:
    """Return ordered (family, module, out_col) over all 5 groups."""
    items: list[tuple[str, str, str]] = []
    for group, prefix in (("alpha", "alpha_"), ("gtja", "gtja_"), ("talib", ""),
                          ("academic", ""), ("github", "")):
        stems = sorted(
            p.stem for p in (BASE / "factorlib" / group).glob("*.py")
            if p.name != "__init__.py"
        )
        for stem in stems:
            items.append((group, stem, f"{prefix}{stem}"))
    cols = [c for _, _, c in items]
    assert len(cols) == len(set(cols)), "duplicate output columns"
    return items


def build_frame() -> pl.DataFrame:
    df = pl.read_parquet(FSDB, columns=INPUT_COLS)
    # Anchor convention: fsdb *_hfq = raw * cum_t/cum_latest (latest-anchored).
    # The v34 factor library was computed on listing-anchored prices
    # (raw * cum_t). Multiply by cum_latest to restore the v34 convention;
    # verified empirically: listing anchor lifts level-factor rank corr vs v34
    # from ~0.67 to ~0.99 (talib_SMA), ~0.95 to ~1.00 (MACD), etc.
    mult = pl.col("cum_latest")
    f = df.select([
        pl.col("ts_code").alias("stock_code"),
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date"),
        (pl.col("open_hfq") * mult).alias("open"),
        (pl.col("high_hfq") * mult).alias("high"),
        (pl.col("low_hfq") * mult).alias("low"),
        (pl.col("close_hfq") * mult).alias("close"),
        # raw close kept for factors defined on untraded price levels (p_pr)
        pl.col("close").alias("close_raw"),
        (pl.col("volume") / 100.0).alias("volume"),
        (pl.col("volume") / 100.0).alias("vol"),
        (pl.col("amount") / 1000.0).alias("amount"),
        (pl.col("total_mv") / 1e4).alias("cap"),
        (pl.col("float_mv") / 1e4).alias("circ_cap"),
        pl.col("pe_ttm").alias("pe"),
        # bps (book per share) anchored on the same scale as `close`:
        (pl.col("close_hfq") / pl.col("pb") * mult).alias("bps"),
        pl.col("sw_l2_name").alias("industry"),
        pl.col("sw_l2_name").alias("sub_industry"),
        (pl.col("adj_factor_hfq") * mult).alias("adj_factor_hfq"),
    ]).sort(["stock_code", "trade_date"])
    f = f.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1.0)
        .fill_null(0.0).alias("returns"),
        ((pl.col("amount") * 10.0 / pl.col("volume").clip(lower_bound=1))
         * pl.col("adj_factor_hfq")).alias("vwap"),
    ])
    for w in ADV_WINDOWS:
        f = f.with_columns(
            pl.col("volume").rolling_mean(w, min_samples=1).over("stock_code").alias(f"adv{w}")
        )
    return f


def run_init() -> None:
    items = master_list()
    groups = {}
    for g, _, _ in items:
        groups[g] = groups.get(g, 0) + 1
    contract = {
        "source_panel": str(FSDB),
        "factor_count": len(items),
        "groups": groups,
        "removed_from_library": REMOVED_FACTORS,
        "price_series": "open_hfq/high_hfq/low_hfq/close_hfq * cum_latest (listing-anchored = raw*cum_t)",
        "adjustment_note": (
            "fsdb *_hfq columns are latest-anchored (raw*cum_t/cum_latest = raw at newest date). "
            "The v34 library was computed on listing-anchored prices (raw*cum_t). "
            "Multiplying *_hfq by cum_latest restores the listing anchor; empirically verified: "
            "level-factor rank-corr vs v34 rises from 0.67->0.99 (talib_SMA), 0.95->1.00 (MACD). "
            "The published *_qfq columns amplify dividend gaps and are NOT used."
        ),
        "unit_conversions": {
            "volume": "shares / 100 -> lots",
            "amount": "yuan / 1000 -> thousand yuan",
            "total_mv,float_mv": "yuan / 1e4 -> 10k yuan",
            "vwap": "amount(千元)*10/volume(手)*adj_factor_hfq",
        },
        "industry": "sw_l2_name -> industry and sub_industry",
        "v_bm_note": "bps feeder := close_hfq/pb; v_bm = 1/pb",
        "raw_close_factors": {
            "p_pr": "v34 semantics = log(raw untraded close); v4 uses fsdb raw `close` column",
        },
        "master_list": [f"{g}/{m}" for g, m, _ in items],
    }
    (CACHE / "v4_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    log(f"contract written: {len(items)} factors {groups}")

    # preflight stats
    df = pl.read_parquet(FSDB, columns=["ts_code", "trade_date", "pb", "sw_l2_name",
                                        "close_hfq", "volume", "amount", "total_mv"])
    n = df.height
    pb_nulls = df["pb"].null_count()
    ind_nulls = df["sw_l2_name"].null_count()
    log(f"preflight: rows={n} stocks={df['ts_code'].n_unique()} "
        f"pb_null={pb_nulls} ({pb_nulls/n:.2%}) sw_l2_null={ind_nulls} ({ind_nulls/n:.2%})")
    keys = df.select(["ts_code", "trade_date"]).unique()
    log(f"keys: unique={keys.height} total={n} dup={'YES' if keys.height != n else 'no'}")


def run_batch(start: int, count: int) -> None:
    items = master_list()
    total = len(items)
    end = min(start + count, total)
    todo = items[start:end]
    out_path = CACHE / f"batch_{start:04d}_{end:04d}.parquet"
    meta_path = CACHE / f"batch_{start:04d}_{end:04d}.json"
    if out_path.exists():
        log(f"batch {start}:{end} already exists, skip")
        return
    log(f"batch {start}:{end} -- {len(todo)} factors; building frame...")
    t0 = time.time()
    frame = build_frame()
    log(f"  frame: {frame.shape} in {time.time()-t0:.1f}s")

    keys = frame.select(["stock_code", "trade_date"])
    series_list = []
    null_rates: dict[str, float] = {}
    errors: dict[str, str] = {}
    for family, module, col in todo:
        t1 = time.time()
        try:
            mod = importlib.import_module(f"factorlib.{family}.{module}")
            frame_in = frame
            if module in RAW_CLOSE_FACTORS:
                # v34 semantics for these factors: computed on the RAW (untraded) close
                frame_in = frame.with_columns(pl.col("close_raw").alias("close"))
            s = mod.compute(frame_in)
            s = s.alias(col)
            series_list.append(s)
            null_rates[col] = round(s.null_count() / len(s), 6)
            log(f"  ok {col:<34} {time.time()-t1:.1f}s null={null_rates[col]:.4f}")
        except Exception as exc:  # noqa: BLE001
            errors[col] = f"{type(exc).__name__}: {exc}"[:300]
            log(f"  ERR {col}: {errors[col]}")

    out = keys.with_columns(series_list)
    out = out.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d").alias("trade_date"))
    out = out.rename({"stock_code": "ts_code"}).sort(["ts_code", "trade_date"])
    out.write_parquet(out_path)
    meta = {
        "start": start, "end": end, "factors": [c for _, _, c in todo],
        "null_rates": null_rates, "errors": errors,
        "runtime_s": round(time.time() - t0, 1), "cols_saved": out.shape[1] - 2,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  saved {out_path.name}: {out.shape} in {meta['runtime_s']}s")


def run_status() -> None:
    items = master_list()
    total = len(items)
    done = 0
    for p in sorted(CACHE.glob("batch_*.json")):
        meta = json.loads(p.read_text())
        done += meta["end"] - meta["start"]
        err = len(meta.get("errors", {}))
        log(f"  {p.stem}: factors {meta['start']}-{meta['end']} err={err} {meta['runtime_s']}s")
    log(f"status: {done}/{total} factors computed")


def run_assemble() -> None:
    items = master_list()
    total = len(items)
    batches = sorted(CACHE.glob("batch_*.parquet"))
    if not batches:
        raise SystemExit("no batches")
    log("assembling...")
    v3 = pl.read_parquet(FSDB)
    log(f"  v3: {v3.shape}")

    # check row-key alignment between v3 and batch frames
    b0 = pl.read_parquet(batches[0], columns=["ts_code", "trade_date"])
    v3_keys = v3.select(["ts_code", "trade_date"])
    aligned = (
        v3_keys.height == b0.height
        and v3_keys.select(pl.col("ts_code")).equals(b0.select(pl.col("ts_code")))
        and v3_keys.select(pl.col("trade_date")).equals(b0.select(pl.col("trade_date")))
    )
    log(f"  key alignment (row-order): {'YES' if aligned else 'NO -> join path'}")

    if aligned:
        frames = [v3]
        for p in batches:
            frames.append(pl.read_parquet(p).drop(["ts_code", "trade_date"]))
        v4 = pl.concat(frames, how="horizontal")
    else:
        v4 = v3
        for p in batches:
            bf = pl.read_parquet(p)
            v4 = v4.join(bf, on=["ts_code", "trade_date"], how="left", validate="1:1")
        assert v4.height == v3.height, f"join changed height: {v4.height} != {v3.height}"
    log(f"  v4: {v4.shape}")

    factor_cols = [c for _, _, c in items]
    missing = [c for c in factor_cols if c not in v4.columns]
    assert not missing, f"missing factor cols: {missing}"

    # null-rate audit
    rows = []
    for c in factor_cols:
        s = v4[c]
        rows.append({
            "factor": c, "null_rate": round(s.null_count() / s.len(), 6),
            "n_unique": int(s.n_unique()), "std": float(s.std()) if s.null_count() < s.len() else None,
        })
    audit = pl.DataFrame(rows)
    audit.write_csv(CACHE / "v4_factor_audit.csv")
    const_cols = audit.filter(pl.col("n_unique") <= 1)["factor"].to_list()
    hi_null = audit.filter(pl.col("null_rate") > 0.5)["factor"].to_list()
    log(f"  audit: const_cols={const_cols} hi_null(>50%)={len(hi_null)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = BASE / "data" / f"wavehunter_hs300_fsdb_v4_{ts}.parquet"
    v4.write_parquet(out_path)
    log(f"saved {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")

    manifest = {
        "panel": str(out_path),
        "shape": list(v4.shape),
        "base_panel": str(FSDB),
        "factor_count_expected": total,
        "factor_count_present": len(factor_cols),
        "removed_from_library": REMOVED_FACTORS,
        "const_cols": const_cols,
        "hi_null_count": len(hi_null),
        "hi_null_sample": hi_null[:40],
        "built_at": ts,
    }
    (CACHE / "v4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log("assemble done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--batch", nargs=2, type=int, metavar=("START", "COUNT"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.init:
        run_init()
    elif args.batch:
        run_batch(args.batch[0], args.batch[1])
    elif args.status:
        run_status()
    elif args.assemble:
        run_assemble()
    else:
        ap.print_help()
