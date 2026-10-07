"""Tab 5: Daily Factor Reselect — full pipeline with AKQuant backtest.

Reads pipeline_v3 progress + result, shows live NAV curve.

Use cases:
  - User clicks Run → server spawns pipeline_v3.py
  - WebUI polls progress.jsonl every 2 seconds
  - Live NAV curve drawn from progress events
  - Final result.json has full ledger + nav_curve
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/api/daily_factor", tags=["daily_factor"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
EVIDENCE_DIR = ROOT / "evidence" / "daily_factor_reselect" / "pipeline_v3"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = EVIDENCE_DIR / "progress.jsonl"

CLI_SCRIPT = ROOT / "examples" / "daily_factor_pipeline_v3.py"


def _job_paths(job_id: str) -> dict[str, Path]:
    jd = EVIDENCE_DIR / job_id
    return {
        "dir": jd,
        "result": jd / "result.json",
        "progress": jd / "progress.jsonl",
        "error": jd / "error.log",
        "log": jd / "pipeline.log",
    }


@router.post("/run")
def run_pipeline(payload: dict = Body(...)):
    """Launch daily factor reselect pipeline (long-running)."""
    start_date = payload.get("start_date", "2010-01-01")
    end_date = payload.get("end_date", "2025-12-31")
    rebal_days = int(payload.get("rebal_days", 20))
    top_k = int(payload.get("top_k", 20))
    window_days = int(payload.get("window_days", 60))

    if not CLI_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CLI not found: {CLI_SCRIPT}",
        )

    job_id = f"daily_v3_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    cmd = [
        "/usr/bin/python3.12",
        str(CLI_SCRIPT),
        "--job-id", job_id,
        "--start-date", start_date,
        "--end-date", end_date,
        "--rebal-days", str(rebal_days),
        "--top-k", str(top_k),
        "--window-days", str(window_days),
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
    return {
        "job_id": job_id,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "rebal_days": rebal_days,
            "top_k": top_k,
            "window_days": window_days,
        },
        "status": "started",
    }


@router.get("/progress/{job_id}")
def get_progress(job_id: str):
    """Read live progress.jsonl for a job."""
    paths = _job_paths(job_id)
    if not paths["dir"].exists():
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    events = []
    progress_file = paths["progress"]
    if progress_file.exists():
        for line in progress_file.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    finished = paths["result"].exists()
    return {
        "job_id": job_id,
        "finished": finished,
        "events": events,
        "n_events": len(events),
        "latest_event": events[-1] if events else None,
    }


@router.get("/result/{job_id}")
def get_result(job_id: str):
    """Read final result.json (when pipeline done)."""
    paths = _job_paths(job_id)
    if not paths["result"].exists():
        raise HTTPException(status_code=404, detail=f"result not ready for {job_id}")
    return json.loads(paths["result"].read_text())


@router.get("/jobs")
def list_jobs():
    """List all pipeline jobs."""
    jobs = []
    for d in sorted(EVIDENCE_DIR.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("daily_v3_"):
            result_file = d / "result.json"
            progress_file = d / "progress.jsonl"
            jobs.append({
                "job_id": d.name,
                "finished": result_file.exists(),
                "n_events": sum(1 for _ in progress_file.open()) if progress_file.exists() else 0,
                "started_at": d.stat().st_mtime,
            })
    return {"jobs": jobs}


@router.get("/latest_progress")
def get_latest_progress():
    """Read the most recent progress events across all jobs (for the overview)."""
    if not PROGRESS_FILE.exists():
        return {"events": [], "n_events": 0, "latest_event": None}
    events = []
    for line in PROGRESS_FILE.read_text().splitlines()[-50:]:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return {
        "events": events,
        "n_events": len(events),
        "latest_event": events[-1] if events else None,
    }