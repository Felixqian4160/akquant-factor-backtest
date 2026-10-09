"""构建 v34 去重面板 —— 去重标准：完全重复（数值/公式完全一致）。

与"同类型/高相关"无关（相关聚类结果仅作参考，不作为删除依据）。

方法:
  1) find 阶段: 对 428 个投票因子逐列计算全量指纹
     (n_null, n_nan, n_finite, sum, sumsq, min, max) → 候选组 →
     候选组逐值严格验证 (NaN 模式一致 + 全序列 allclose rtol=1e-9) →
     输出 evidence/audit_lookahead_20261010/v34_exact_dups.json
  2) build 阶段: 读取验证结果 → 生成新面板（每组保留面板列序最先出现者）
     输出 data/wavehunter_hs300_v34_dedup_20261010.parquet
          + evidence/audit_lookahead_20261010/v34_dedup_manifest.json
  说明: 全 NaN 列不参与去重（单独标注）；v34 原面板只读、不覆盖。

用法:
  python3.12 -u examples/build_v34_dedup_panel.py --stage find
  python3.12 -u examples/build_v34_dedup_panel.py --stage build
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

AKQ = Path("/media/felix/f/quant/akquant-factor-backtest")
SRC = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
DST = AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet"
DUPS = AKQ / "evidence" / "audit_lookahead_20261010" / "v34_exact_dups.json"
MANIFEST = AKQ / "evidence" / "audit_lookahead_20261010" / "v34_dedup_manifest.json"

BASE_COLS = {
    "trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount",
    "pct_chg", "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
    "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60", "turnover_rate",
    "circ_cap", "cap", "symbol", "volume",
    "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
}
ZIGZAG = {
    "v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
    "v10_1_b1_start", "v10_1_b1_interval",
    "v10_1_down_start", "v10_1_down_interval",
    "v10_1_peak_zone", "v10_1_valley_zone",
    "v10_1_zig_peak", "v10_1_zig_valley",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def factor_list() -> tuple[list[str], list[str]]:
    cols = list(pl.read_parquet_schema(SRC).keys())
    factors = [c for c in cols if c not in (BASE_COLS | ZIGZAG)]
    return cols, factors


def stage_find() -> None:
    cols, factors = factor_list()
    log(f"panel cols={len(cols)}; voting factors={len(factors)}")

    fp = {}
    B = 40
    for i in range(0, len(factors), B):
        batch = factors[i:i + B]
        df = pl.read_parquet(SRC, columns=batch)
        for c in batch:
            s = df[c].cast(pl.Float64)
            fin = s.filter(s.is_finite())
            n_fin = fin.len()
            if n_fin == 0:
                fp[c] = ("all_nan", s.null_count(), int(s.is_nan().sum()))
            else:
                fp[c] = (s.null_count(), int(s.is_nan().sum()), n_fin,
                         float(fin.sum()), float((fin * fin).sum()),
                         float(fin.min()), float(fin.max()))
        log(f"  fingerprint {min(i + B, len(factors))}/{len(factors)}")

    all_nan = sorted(c for c in factors if fp[c][0] == "all_nan")
    groups = defaultdict(list)
    for c in factors:
        if fp[c][0] == "all_nan":
            continue
        k = tuple(round(x, 6) if isinstance(x, float) else x for x in fp[c])
        groups[k].append(c)
    cand = [sorted(g) for g in groups.values() if len(g) > 1]
    log(f"candidate groups: {len(cand)}; all-nan cols: {all_nan}")

    verified = []
    for g in cand:
        df = pl.read_parquet(SRC, columns=g)
        arrs = [df[c].cast(pl.Float64).to_numpy() for c in g]
        base = arrs[0]
        ok = True
        maxd = 0.0
        bnan = np.isnan(base)
        for a in arrs[1:]:
            if not np.array_equal(bnan, np.isnan(a)):
                ok = False
                break
            mask = np.isfinite(base)
            if mask.any():
                d = float(np.max(np.abs(base[mask] - a[mask])))
                maxd = max(maxd, d)
                if not np.allclose(base[mask], a[mask], rtol=1e-9, atol=1e-12):
                    ok = False
                    break
        if ok:
            verified.append({"members": g, "keep": g[0], "max_abs_diff": maxd})
        else:
            log(f"  group FAILED verification: {g}")

    verified.sort(key=lambda x: -len(x["members"]))
    out = {"n_factors": len(factors), "n_groups": len(verified),
           "n_removed": sum(len(v["members"]) - 1 for v in verified),
           "all_nan_cols": all_nan, "groups": verified}
    DUPS.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"verified exact-dup groups: {len(verified)}; will remove "
        f"{out['n_removed']} cols")
    for v in verified[:30]:
        log(f"  [{len(v['members'])}] keep={v['keep']:<30s} "
            f"drop={[m for m in v['members'] if m != v['keep']]}  max|d|={v['max_abs_diff']:.2e}")


def stage_build() -> None:
    cols, factors = factor_list()
    dups = json.loads(DUPS.read_text())
    remove = sorted({m for v in dups["groups"] for m in v["members"] if m != v["keep"]})
    keep_cols = [c for c in cols if c not in remove]
    log(f"cols {len(cols)} -> {len(keep_cols)} (removing {len(remove)})")

    if DST.exists():
        log(f"DST exists, skip write: {DST}")
    else:
        t0 = time.time()
        try:
            pl.scan_parquet(SRC).select(keep_cols).sink_parquet(str(DST))
        except Exception as exc:  # noqa: BLE001
            log(f"sink_parquet failed ({type(exc).__name__}), fallback read/write")
            pl.read_parquet(SRC, columns=keep_cols).write_parquet(DST)
        log(f"saved {DST} ({DST.stat().st_size / 1e9:.2f} GB, {time.time()-t0:.0f}s)")

    chk = pl.scan_parquet(DST).select(pl.len()).collect()[0, 0]
    src_rows = pl.scan_parquet(SRC).select(pl.len()).collect()[0, 0]
    out_cols = list(pl.read_parquet_schema(DST).keys())
    assert chk == src_rows, f"row count mismatch: {chk} != {src_rows}"
    assert len(out_cols) == len(keep_cols), "col count mismatch"
    assert all(c not in out_cols for c in remove), "removed col still present"
    log(f"verify: rows={chk} (same), cols={len(out_cols)} OK")

    manifest = {
        "source": str(SRC), "output": str(DST),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "criterion": "exact duplicate only (values/formula identical; NOT correlation-based)",
        "n_cols_before": len(cols), "n_cols_after": len(out_cols),
        "n_voting_factors_before": len(factors),
        "n_voting_factors_after": len([c for c in out_cols if c not in (BASE_COLS | ZIGZAG)]),
        "removed_columns": remove,
        "groups": dups["groups"],
        "all_nan_cols": dups["all_nan_cols"],
        "note": "reference only: correlation clustering (|rho|>=0.95 -> 318 reps) lives in "
                "factor_dedup_groups.json; NOT used for this panel",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"manifest: {MANIFEST}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["find", "build"], required=True)
    args = ap.parse_args()
    if args.stage == "find":
        stage_find()
    else:
        stage_build()


if __name__ == "__main__":
    main()
