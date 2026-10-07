"""Tab 6: v34-lb20 Factor-Return Voting — production-locked strategy.

Spawns `system.sh` subcommands (verify / picks / sims / plot / report) and
streams log/progress/result back to the WebUI.

Endpoints:
  POST /api/v34lb20/verify         -> run SYSTEM_VERIFY.py (~1 sec)
  POST /api/v34lb20/derive-picks   -> 10 offset picks via matrix (~30 sec)
  POST /api/v34lb20/run-sim        -> 1 offset demo sim (offset 0, ~90 sec)
  POST /api/v34lb20/run-all        -> full batch: picks + 10 sims + plots
  POST /api/v34lb20/report         -> regenerate STAGE5_REPORT.md
  POST /api/v34lb20/plot           -> regenerate 4 NAV/yearly charts
  GET  /api/v34lb20/jobs           -> list historical jobs

This router NEVER runs calculations itself — it only spawns CLI subprocesses
and reads their on-disk outputs.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/api/v34lb20", tags=["v34lb20"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
SYSTEM_SH = ROOT / "system.sh"
EVIDENCE_BASE = ROOT / "evidence" / "v34lb20_jobs"
EVIDENCE_BASE.mkdir(parents=True, exist_ok=True)


def _job_paths(job_id: str) -> dict[str, Path]:
    jd = EVIDENCE_BASE / job_id
    return {
        "dir": jd,
        "log": jd / "log.txt",
        "result": jd / "result.json",
        "progress": jd / "progress.jsonl",
        "error": jd / "error.log",
    }


def _spawn_subcommand(subcommand: str, extra_args: list[str] | None = None) -> str:
    """Spawn `system.sh <subcommand> [extra_args]` in background.

    Passes JOB_ID env var so system.sh can write a result.json marker
    that the router's /status endpoint uses to mark finished=True.
    """
    job_id = f"{subcommand}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cmd = [str(SYSTEM_SH), subcommand] + (extra_args or [])
    log = open(paths["log"], "w")
    log.write(f"cmd: {' '.join(cmd)}\n")
    log.flush()
    env = os.environ.copy()
    env["JOB_ID"] = job_id
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return job_id


@router.post("/verify")
def run_verify():
    """Lock check: re-hash every locked artifact (~1 sec)."""
    return {"job_id": _spawn_subcommand("verify")}


@router.post("/derive-picks")
def derive_picks():
    """Matrix-based 10 offset picks (~30 seconds)."""
    return {"job_id": _spawn_subcommand("picks")}


@router.post("/run-sim")
def run_one_sim(payload: dict = Body(default={})):
    """Run 1 offset sim (offset 0 only, ~90 sec).

    For full batch (10 sims), use POST /run-all instead.
    """
    offset = int(payload.get("offset", 0))
    window = payload.get("window", "full")  # 'full' | 'oos'
    end = "2025-12-31" if window == "full" else "2026-08-27"
    job_id = f"v34lb20_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cli = ROOT / "examples" / "v41_run_akquant_v34.py"
    cmd = [
        "/usr/bin/python3.12", "-u", str(cli),
        "--tag", f"V14_{offset}",
        "--picks-base", str(ROOT / "evidence" / "stage5_20261007" / "picks"),
        "--actions", "on",
        "--ca-mode", "all",
        "--prices", "raw",
        "--end", end,
        "--out-base", str(paths["dir"] / "sim"),
    ]
    log = open(paths["log"], "w")
    log.write(f"cmd: {' '.join(cmd)}\n")
    log.flush()
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"job_id": job_id, "cmd": cmd}


@router.post("/run-all")
def run_full_batch():
    """Full pipeline: picks + 10 sims + plots + report (~25 minutes)."""
    return {"job_id": _spawn_subcommand("reproduce")}


@router.post("/report")
def regen_report():
    """Regenerate STAGE5_REPORT.md from current artifacts (~3 sec)."""
    return {"job_id": _spawn_subcommand("report")}


@router.post("/plot")
def regen_plot():
    """Regenerate 4 NAV/yearly charts (~5 sec)."""
    return {"job_id": _spawn_subcommand("plot")}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    paths = _job_paths(job_id)
    if not paths["dir"].exists():
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    finished = paths["result"].exists()
    error_log = paths["error"].exists() and paths["error"].stat().st_size > 0
    log_tail = paths["log"].read_text().splitlines()[-20:] if paths["log"].exists() else []
    return {
        "job_id": job_id,
        "finished": finished,
        "error": error_log,
        "log_tail": log_tail,
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
    for jd in sorted(EVIDENCE_BASE.iterdir(), reverse=True):
        if not jd.is_dir():
            continue
        result = jd / "result.json"
        jobs.append({
            "job_id": jd.name,
            "finished": result.exists(),
            "created": jd.stat().st_mtime,
        })
    return {"jobs": jobs[:30]}


@router.get("/status")
def system_status():
    """Quick read-only status of the v34-lb20 system (no subprocess)."""
    lock_file = ROOT / "SYSTEM_LOCK.json"
    strategy_doc = ROOT / "STRATEGIES" / "v34_lb20_factor_return_voting.md"
    panel = ROOT / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
    matrix = ROOT / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
    return {
        "lock_exists": lock_file.exists(),
        "lock_locked_at": json.loads(lock_file.read_text())["locked_at"] if lock_file.exists() else None,
        "strategy_doc_exists": strategy_doc.exists(),
        "panel_exists": panel.exists(),
        "panel_size_gb": round(panel.stat().st_size / 1e9, 2) if panel.exists() else 0,
        "matrix_exists": matrix.exists(),
        "matrix_size_mb": round(matrix.stat().st_size / 1e6, 1) if matrix.exists() else 0,
    }
