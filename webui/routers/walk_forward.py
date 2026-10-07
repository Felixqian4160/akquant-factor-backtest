"""Tab 2: Walk-Forward Validation.

Spawns the walk-forward CLI which runs multiple time-series splits
(train → val → OOS) with the Stage 5b configuration
(top_k=10, kill DD>8%/cooldown=20d, 6-factor bull_composite).

Each window: ~10-30 seconds. Total run: ~2-5 minutes for 5 windows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/api/walkforward", tags=["walkforward"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
JOBS_DIR = ROOT / "webui" / "data" / "wf_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

CLI_SCRIPT = ROOT / "webui" / "scripts" / "walk_forward_v5b.py"


def _job_paths(job_id: str) -> dict[str, Path]:
    jd = JOBS_DIR / job_id
    return {
        "dir": jd,
        "result": jd / "result.json",
        "progress": jd / "progress.jsonl",
        "error": jd / "error.log",
        "log": jd / "log.txt",
    }


@router.post("/run")
def run_walk_forward(payload: dict = Body(...)):
    """Launch a walk-forward job (non-blocking). Returns job_id."""
    train_start = payload.get("train_start", "2010-01-01")
    train_end = payload.get("train_end", "2018-12-31")
    val_months = int(payload.get("val_months", 6))
    oos_months = int(payload.get("oos_months", 12))
    step_months = int(payload.get("step_months", 6))
    max_windows = int(payload.get("max_windows", 6))
    if not CLI_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CLI script not found: {CLI_SCRIPT}. Run setup first.",
        )
    job_id = f"wf_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/python3.12",
        str(CLI_SCRIPT),
        "--job-id", job_id,
        "--train-start", train_start,
        "--train-end", train_end,
        "--val-months", str(val_months),
        "--oos-months", str(oos_months),
        "--step-months", str(step_months),
        "--max-windows", str(max_windows),
        "--out-dir", str(paths["dir"]),
    ]
    log = open(paths["log"], "w")
    log.write(f"cmd: {' '.join(cmd)}\n")
    log.flush()
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"job_id": job_id, "cmd": cmd, "status": "started"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    paths = _job_paths(job_id)
    if not paths["dir"].exists():
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    progress = []
    if paths["progress"].exists():
        for line in paths["progress"].read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    progress.append(json.loads(line))
                except Exception:
                    progress.append({"raw": line})
    finished = paths["result"].exists()
    error = paths["error"].exists() and paths["error"].stat().st_size > 0
    return {
        "job_id": job_id,
        "finished": finished,
        "error": error,
        "progress": progress,
        "log_tail": paths["log"].read_text().splitlines()[-20:] if paths["log"].exists() else [],
    }


@router.get("/result/{job_id}")
def get_result(job_id: str):
    paths = _job_paths(job_id)
    if not paths["result"].exists():
        raise HTTPException(status_code=404, detail=f"result not ready for {job_id}")
    return json.loads(paths["result"].read_text())


@router.get("/jobs")
def list_jobs():
    jobs = []
    for jd in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not jd.is_dir():
            continue
        job_id = jd.name
        result = jd / "result.json"
        jobs.append({
            "job_id": job_id,
            "finished": result.exists(),
            "created": jd.stat().st_mtime,
        })
    return {"jobs": jobs[:30]}