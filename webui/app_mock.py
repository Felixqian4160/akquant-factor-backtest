"""Mock v34-lb20 webui: returns synthetic JSON without spawning subprocesses.

Purpose:
  - Verify that the FRONTEND (v34_lb20.js) parses and renders the JSON
    correctly without actually running any backtest / verify subprocess.
  - Use as a smoke test for tab layout, button wiring, schema compatibility.

Run:
  /usr/bin/python3.12 webui/app_mock.py

Then open http://localhost:8089 in browser. The UI should render:
  - System status with lock/panel/matrix state
  - Synthetic job history (3 done, 1 running, 1 failed)
  - Click any historical job → log tail visible

This is a NO-EXECUTION surface — all "jobs" are hardcoded fixtures.
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "webui" / "static"

# Synthetic fixtures (no subprocess ever runs)
MOCK_LOCK = {
    "lock_version": 1,
    "locked_at": "2026-10-07T17:13:35+0800",
    "scripts": {f"examples/script_{i}.py": {"sha256": f"abcd1234{i}", "bytes": 1000 + i} for i in range(9)},
    "artifacts": {
        "data/wavehunter_hs300_v34_adj_20261007.parquet": {"sha256": "panel_sha", "bytes": 2_973_000_000},
        "evidence/stage_a_20261007/matrix_v34_ADJ.parquet": {"sha256": "matrix_sha", "bytes": 17_700_000},
        "STRATEGIES/v34_lb20_factor_return_voting.md": {"sha256": "doc_sha", "bytes": 18_000},
    },
    "ev_dirs": ["evidence/stage_a_20261007", "evidence/stage5_20261007", "evidence/v34_adj_20261007", "evidence/causal_zigzag_router_20261006"],
}

NOW = time.time()

# 5 fake jobs: 3 finished, 1 running, 1 failed
MOCK_JOBS = [
    {
        "job_id": "verify_1791387449_51abb3",
        "subcommand": "verify",
        "finished": True,
        "error": False,
        "created": NOW - 3600,
        "result": {"job_id": "verify_1791387449_51abb3", "status": "ok", "message": "verified", "exit_code": 0, "completed_at": "2026-10-07T15:37:32Z"},
        "log_tail": [
            "cmd: /media/felix/f/quant/akquant-factor-backtest/system.sh verify",
            "  OK  examples/build_picks_v34_factorrank.py",
            "  OK  examples/v41_run_akquant_v34.py",
            "  OK  examples/stage_a_1_matrices_and_picks.py",
            "  OK  examples/stage_a_2_run_sims.py",
            "  OK  examples/stage5_picks_and_sims.py",
            "  OK  examples/stage5_run_sims.py",
            "  OK  examples/stage5_report.py",
            "  OK  examples/diag_split_mechanism.py",
            "  OK  examples/derive_corporate_actions_local.py",
            "  OK  data/wavehunter_hs300_v34_adj_20261007.parquet",
            "  OK  evidence/stage_a_20261007/matrix_v34_ADJ.parquet",
            "  OK  evidence/v34_adj_20261007/corporate_actions_derived.parquet",
            "  OK  evidence/causal_zigzag_router_20261006/router_map.json",
            "  OK  STRATEGIES/v34_lb20_factor_return_voting.md",
            "  OK  SYSTEM_VERIFY.py",
            "  OK  system.sh",
            "",
            "ALL 16 LOCKED ARTIFACTS VERIFIED OK",
            "",
            '{"job_id":"verify_1791387449_51abb3","status":"ok","message":"verified","exit_code":0,"completed_at":"2026-10-07T15:37:32Z"}',
        ],
    },
    {
        "job_id": "picks_1791389000_aabb01",
        "subcommand": "picks",
        "finished": True,
        "error": False,
        "created": NOW - 7200,
        "result": {"job_id": "picks_1791389000_aabb01", "status": "ok", "message": "10 offset picks derived", "exit_code": 0, "completed_at": "2026-10-07T13:00:00Z"},
        "log_tail": [
            ">>> Deriving 10 offset picks (matrix-based)...",
            "[17:01:23] derived V14_0: 195 dates (104 bull + 91 bear) elapsed 2.7s",
            "[17:01:26] derived V14_2: 195 dates elapsed 2.7s",
            "[17:01:29] derived V14_4: 195 dates elapsed 2.7s",
            "[17:01:32] derived V14_6: 195 dates elapsed 2.7s",
            "[17:01:35] derived V14_8: 195 dates elapsed 2.7s",
            "[17:01:38] derived V14_10: 195 dates elapsed 2.7s",
            "[17:01:41] derived V14_12: 195 dates elapsed 2.7s",
            "[17:01:44] derived V14_14: 195 dates elapsed 2.7s",
            "[17:01:47] derived V14_16: 195 dates elapsed 2.7s",
            "[17:01:50] derived V14_18: 195 dates elapsed 2.7s",
            '{"job_id":"picks_1791389000_aabb01","status":"ok","message":"10 offset picks derived","exit_code":0}',
        ],
    },
    {
        "job_id": "plot_1791390000_bbbb02",
        "subcommand": "plot",
        "finished": True,
        "error": False,
        "created": NOW - 10800,
        "result": {"job_id": "plot_1791390000_bbbb02", "status": "ok", "message": "4 charts regenerated", "exit_code": 0, "completed_at": "2026-10-07T11:30:00Z"},
        "log_tail": [
            ">>> Regenerating charts...",
            "Loading navs + HS300...",
            "  full: 10, oos: 10; HS300: 3886",
            "Plotting...",
            "  -> 01_nav_full_2010_2025.png",
            "  -> 02_nav_full_2010_2026.png",
            "  -> 03_zoom_2026_oos.png",
            "  -> 04_yearly_bars_10offsets.png",
            "DONE",
            '{"job_id":"plot_1791390000_bbbb02","status":"ok","message":"4 charts regenerated","exit_code":0}',
        ],
    },
    {
        "job_id": "run-sim_1791401100_cccc03",
        "subcommand": "run-sim",
        "finished": False,
        "error": False,
        "created": NOW - 30,  # 30 seconds ago, still running
        "result": None,
        "log_tail": [
            "cmd: /usr/bin/python3.12 examples/v41_run_akquant_v34.py --tag V14_0 --picks-base evidence/stage5_20261007/picks --actions on --ca-mode all --prices raw --end 2026-08-27 --out-base evidence/v34lb20_jobs/run-sim_1791401100_cccc03/sim",
            "[17:05:00] Loading panel v34 (461 cols, 1298335 rows)...",
            "[17:05:02] Loading 428 voting factors...",
            "[17:05:04] CA events: 6563 (split 4259, dividend 5799)",
            "[17:05:08] Starting sim (offset 0, 195 rebal dates, T+1 raw open + 25bps commission + 10bps slippage)...",
            "[17:05:30] rebal 50/195 (idx 28), open positions=10, NAV=12050000",
        ],
    },
    {
        "job_id": "report_1791402000_dddd04",
        "subcommand": "report",
        "finished": True,
        "error": True,
        "created": NOW - 86400,
        "result": {"job_id": "report_1791402000_dddd04", "status": "fail", "message": "stage5_report.py: ModuleNotFoundError: stage5_report (synthetic)", "exit_code": 1, "completed_at": "2026-10-06T15:00:00Z"},
        "log_tail": [
            ">>> Regenerating report...",
            "Traceback (most recent call last):",
            '  File "<string>", line 8, in <module>',
            "    import stage5_report",
            "ModuleNotFoundError: No module named 'stage5_report' (synthetic mock)",
            '{"job_id":"report_1791402000_dddd04","status":"fail","exit_code":1}',
        ],
    },
]

app = FastAPI(title="AKQuant v2 Workflow (MOCK)", version="0.1.0-mock")

# Mount the real static dir so we use the real frontend (no changes)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "service": "akquant-v2-webui-mock", "port": 8089, "mode": "MOCK: NO SUBPROCESS EVER SPAWNED"}


@app.get("/")
def root():
    return FileResponse(str(STATIC / "index.html"))


# --- /api/v34lb20/* mock endpoints ---

@app.get("/api/v34lb20/status")
def system_status():
    """Returns hardcoded system state (no real files)."""
    return {
        "lock_exists": True,
        "lock_locked_at": MOCK_LOCK["locked_at"],
        "strategy_doc_exists": True,
        "panel_exists": True,
        "panel_size_gb": round(MOCK_LOCK["artifacts"]["data/wavehunter_hs300_v34_adj_20261007.parquet"]["bytes"] / 1e9, 2),
        "matrix_exists": True,
        "matrix_size_mb": round(MOCK_LOCK["artifacts"]["evidence/stage_a_20261007/matrix_v34_ADJ.parquet"]["bytes"] / 1e6, 1),
        "mode": "MOCK",
    }


@app.get("/api/v34lb20/jobs")
def list_jobs():
    return {
        "jobs": [
            {"job_id": j["job_id"], "finished": j["finished"], "created": j["created"]}
            for j in MOCK_JOBS
        ]
    }


@app.get("/api/v34lb20/status/{job_id}")
def get_status(job_id: str):
    job = next((j for j in MOCK_JOBS if j["job_id"] == job_id), None)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return {
        "job_id": job["job_id"],
        "finished": job["finished"],
        "error": job["error"],
        "log_tail": job["log_tail"],
    }


@app.get("/api/v34lb20/result/{job_id}")
def get_result(job_id: str):
    job = next((j for j in MOCK_JOBS if j["job_id"] == job_id), None)
    if not job or not job.get("result"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"result not ready for {job_id}")
    return job["result"]


# All action POST endpoints: return synthetic job_id instantly, NO subprocess
MOCK_POST_RESPONSES = {
    "verify": {"job_id": "verify_1791387449_51abb3", "mode": "MOCK"},
    "derive-picks": {"job_id": "picks_1791389000_aabb01", "mode": "MOCK"},
    "run-sim": {"job_id": "run-sim_1791401100_cccc03", "mode": "MOCK"},
    "run-all": {"job_id": "run-all_1791403000_eeee05", "mode": "MOCK"},
    "report": {"job_id": "report_1791402000_dddd04", "mode": "MOCK"},
    "plot": {"job_id": "plot_1791390000_bbbb02", "mode": "MOCK"},
}


@app.post("/api/v34lb20/{action}")
def run_action(action: str, payload: dict = Body(default={})):
    if action not in MOCK_POST_RESPONSES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"unknown action {action}")
    return MOCK_POST_RESPONSES[action]


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("MOCK WEBUI — NO SUBPROCESS EVER SPAWNED")
    print("Open http://localhost:8089 in browser")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8089, log_level="info")