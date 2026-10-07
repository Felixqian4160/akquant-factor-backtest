"""Tab 4: Paper Trading Simulator.

State machine for a simulated portfolio:
- start: pick start_date + initial cash
- for each rebalance_date in [start_date, end_date]:
    - get top-K picks for that date (uses daily_strategy job)
    - execute trades (T+1 open, cost 25bps commission + 10bps slippage)
    - mark-to-market at end of each trading day
- end: report final NAV, total return, MDD, trades ledger

State is persisted as JSON: webui/data/paper_<session_id>/state.json
Each session is independent (user can start multiple paper portfolios).
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

router = APIRouter(prefix="/api/paper", tags=["paper"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
SESSIONS_DIR = ROOT / "webui" / "data" / "paper_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

CLI_SCRIPT = ROOT / "webui" / "scripts" / "paper_trade_v5b.py"


def _session_paths(session_id: str) -> dict[str, Path]:
    sd = SESSIONS_DIR / session_id
    return {
        "dir": sd,
        "state": sd / "state.json",
        "ledger": sd / "ledger.json",
        "result": sd / "result.json",
        "progress": sd / "progress.jsonl",
        "error": sd / "error.log",
        "log": sd / "log.txt",
    }


@router.post("/start")
def start_session(payload: dict = Body(...)):
    """Start a new paper-trading session."""
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    initial_cash = float(payload.get("initial_cash", 1_000_000.0))
    top_k = int(payload.get("top_k", 10))
    rebal_days = int(payload.get("rebal_days", 20))
    if not CLI_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CLI script not found: {CLI_SCRIPT}. Run setup first.",
        )
    try:
        date.fromisoformat(start_date)
        date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format, use YYYY-MM-DD")
    session_id = f"paper_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    paths = _session_paths(session_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/python3.12",
        str(CLI_SCRIPT),
        "--session-id", session_id,
        "--start-date", start_date,
        "--end-date", end_date,
        "--initial-cash", str(initial_cash),
        "--top-k", str(top_k),
        "--rebal-days", str(rebal_days),
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
    return {"session_id": session_id, "cmd": cmd, "status": "started"}


@router.get("/status/{session_id}")
def get_status(session_id: str):
    paths = _session_paths(session_id)
    if not paths["dir"].exists():
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    finished = paths["result"].exists()
    error_log = paths["error"].exists() and paths["error"].stat().st_size > 0
    state = None
    if paths["state"].exists():
        try:
            state = json.loads(paths["state"].read_text())
        except Exception:
            state = None
    return {
        "session_id": session_id,
        "finished": finished,
        "error": error_log,
        "state": state,
        "log_tail": paths["log"].read_text().splitlines()[-20:] if paths["log"].exists() else [],
    }


@router.get("/result/{session_id}")
def get_result(session_id: str):
    paths = _session_paths(session_id)
    if not paths["result"].exists():
        raise HTTPException(status_code=404, detail=f"result not ready for {session_id}")
    return json.loads(paths["result"].read_text())


@router.get("/ledger/{session_id}")
def get_ledger(session_id: str):
    paths = _session_paths(session_id)
    if not paths["ledger"].exists():
        raise HTTPException(status_code=404, detail=f"ledger not ready for {session_id}")
    return json.loads(paths["ledger"].read_text())


@router.get("/sessions")
def list_sessions():
    sessions = []
    for sd in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not sd.is_dir():
            continue
        result = sd / "result.json"
        sessions.append({
            "session_id": sd.name,
            "finished": result.exists(),
            "created": sd.stat().st_mtime,
        })
    return {"sessions": sessions[:30]}