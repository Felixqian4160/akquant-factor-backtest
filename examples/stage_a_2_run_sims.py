"""Stage A-2: run sims for picks variants (resumable, sequential).

Sims: p2_v34rawfwd, p3_v33adjfwd, null_01..null_10, subsample.
Canonical (repro_v34 == v34-lb20-fixed) already run separately.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

AKQ = Path("/media/felix/f/quant/akquant-factor-backtest")
PICKS_ROOT = AKQ / "evidence" / "stage_a_20261007" / "picks"
OUT_ROOT = AKQ / "evidence" / "stage_a_20261007" / "sims"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3, help="max sims per invocation")
    args = ap.parse_args()

    variants = ["p2_v34rawfwd", "p3_v33adjfwd"] + [f"null_{i:02d}" for i in range(1, 11)] + ["subsample"]
    done = 0
    for v in variants:
        if done >= args.max:
            break
        out_dir = OUT_ROOT / v
        if (out_dir / "V14_0" / "result.json").exists():
            continue
        picks_dir = PICKS_ROOT / v
        if not (picks_dir / "V14_0" / "picks.json").exists():
            log(f"skip {v}: picks missing")
            continue
        log(f"===== sim: {v} =====")
        cmd = [
            sys.executable, "-u", str(RUNNER),
            "--tag", "V14_0",
            "--picks-base", str(picks_dir),
            "--actions", "on", "--ca-mode", "all", "--prices", "raw",
            "--out-base", str(out_dir),
        ]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
        rt = time.time() - t0
        ok = (out_dir / "V14_0" / "result.json").exists()
        log(f"  {v}: {'OK' if ok else 'FAIL'} ({rt:.0f}s)")
        if not ok:
            tail = (r.stdout or "")[-400:]
            log(f"  tail: {tail}")
        done += 1

    remaining = [v for v in variants if not (OUT_ROOT / v / "V14_0" / "result.json").exists()]
    log(f"remaining variants: {len(remaining)}: {remaining[:6]}...")


if __name__ == "__main__":
    main()
