"""Tab 3: Daily Strategy — pick top-K stocks for a given date.

Reads v2 panel, computes 6-factor composite score per stock on the chosen
date, returns top-10 picks with scores.

Use cases:
  - User picks a date → server returns today's bull_composite picks
  - User can verify picks by comparing to the previous day's close

Fast: single backtest per request (< 5 seconds for any date in 2004-2026).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, date

from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/api/daily", tags=["daily"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
JOBS_DIR = ROOT / "webui" / "data" / "daily_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

CLI_SCRIPT = ROOT / "webui" / "scripts" / "daily_picks_v5b.py"


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
def run_daily_picks(payload: dict = Body(...)):
    """Compute top-K picks for a given date (YYYY-MM-DD)."""
    pick_date = payload.get("pick_date")
    top_k = int(payload.get("top_k", 10))
    if not pick_date:
        raise HTTPException(status_code=400, detail="pick_date required")
    if not CLI_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CLI script not found: {CLI_SCRIPT}. Run setup first.",
        )
    # Validate date.
    try:
        date.fromisoformat(pick_date)
    except Exception:
        raise HTTPException(status_code=400, detail=f"invalid date: {pick_date}")
    job_id = f"daily_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/python3.12",
        str(CLI_SCRIPT),
        "--job-id", job_id,
        "--date", pick_date,
        "--top-k", str(top_k),
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
    finished = paths["result"].exists()
    error_log = paths["error"].exists() and paths["error"].stat().st_size > 0
    return {
        "job_id": job_id,
        "finished": finished,
        "error": error_log,
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
        result = jd / "result.json"
        jobs.append({
            "job_id": jd.name,
            "finished": result.exists(),
            "created": jd.stat().st_mtime,
        })
    return {"jobs": jobs[:30]}


@router.get("/dates")
def get_date_range():
    """Return valid date range from the v2 panel (2004-01-02 ~ 2026-08-21)."""
    return {
        "start": "2004-01-02",
        "end": "2026-08-21",
        "n_days": 5498,
    }