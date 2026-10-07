"""Stage 5 sim runner — resumable, batches 3 sims per call.

10 offsets × 2 windows (full 2010-2025 + 2026 OOS) = 20 sims, each ~80s.
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
import time

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
OUT_BASE = AKQ / "evidence" / "stage5_20261007"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"

OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
WINDOWS = [
    ("sims", "2025-12-31"),
    ("sims_2026", "2026-08-27"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(offset, window_name, end_date):
    out_dir = OUT_BASE / window_name
    result_path = out_dir / f"V14_{offset}" / "result.json"
    if result_path.exists():
        log(f"  skip V14_{offset}/{window_name}")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(RUNNER),
        "--tag", f"V14_{offset}",
        "--picks-base", str(OUT_BASE / "picks"),
        "--actions", "on", "--ca-mode", "all", "--prices", "raw",
        "--end", end_date,
        "--out-base", str(out_dir),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
    rt = time.time() - t0
    ok = result_path.exists()
    log(f"  V14_{offset} end={end_date}: {'OK' if ok else 'FAIL'} ({rt:.0f}s)")
    if not ok:
        log(f"    tail: {(r.stdout or '')[-300:]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=4)
    args = ap.parse_args()

    work = []
    for wname, edate in WINDOWS:
        for off in OFFSETS:
            work.append((off, wname, edate))

    done = 0
    for off, wname, edate in work:
        if done >= args.max:
            break
        result_path = OUT_BASE / wname / f"V14_{off}" / "result.json"
        if result_path.exists():
            continue
        if run_one(off, wname, edate):
            done += 1

    remaining = [(o, w, e) for o, w, e in work
                 if not (OUT_BASE / w / f"V14_{o}" / "result.json").exists()]
    log(f"remaining: {len(remaining)}")


if __name__ == "__main__":
    main()
