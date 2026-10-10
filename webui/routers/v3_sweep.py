"""Tab 7: v3 共振参数扫描 — WebUI 程序端.

一个完整的参数扫描控制面板: 启动/继续扫描, 实时进度+日志, 收益/回撤比排名,
净值曲线 (Chart.js). 扫描本身由 examples/v3_param_sweep.py CLI 承担。

Endpoints:
  GET  /api/v3sweep/overview  -> 进度 (wave1/54, wave2, stage, running, log_tail)
  GET  /api/v3sweep/results   -> 全部配置排名 (直读 journal.jsonl)
  GET  /api/v3sweep/curves    -> curves.json (phase-mean 净值序列)
  POST /api/v3sweep/run       -> 启动/继续 全自动扫描 (run+validate+report)
  POST /api/v3sweep/curves-gen-> 生成净值曲线数据+PNG (~20s)
  POST /api/v3sweep/report-gen-> 重新生成 sweep_report.md (~5s)

This router NEVER runs calculations itself — it only spawns CLI subprocesses
and reads their on-disk outputs (journal.jsonl / auto_run.log / curves.json).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v3sweep", tags=["v3sweep"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
SWEEP = ROOT / "evidence" / "v3_sweep_20261010"
JOURNAL = SWEEP / "journal.jsonl"
LOG = SWEEP / "auto_run.log"
CURVES = SWEEP / "curves.json"
REPORT = SWEEP / "sweep_report.md"
STATE = SWEEP / "current_run.json"
PY = "/usr/bin/python3.12"
CLI = ROOT / "examples" / "v3_param_sweep.py"
WAVE1_TOTAL = 54
CHAMPION_IDX = 26  # hold=0.5 cool=3 cap=40 pos=10

ENV_OVERRIDES = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "PYTHONPATH": "/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages",
}


def _read_journal() -> list[dict]:
    entries = []
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return entries


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _running():
    """Return (running_bool, detail). Managed state first, then external detection."""
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
            pid = int(st.get("pid", 0))
            if pid and _pid_alive(pid):
                return True, {"pid": pid, "started": st.get("started")}
        except Exception:  # noqa: BLE001
            pass
    if LOG.exists():
        txt = LOG.read_text()[-4000:]
        if "AUTO_RUN_COMPLETE" not in txt and (time.time() - LOG.stat().st_mtime) < 600:
            return True, {"external": True}
    return False, {}


@router.get("/overview")
def overview():
    entries = _read_journal()
    w1 = {e["idx"] for e in entries if e.get("ok") and e.get("wave") == 1}
    w2 = {(e["idx"], e["offset"]) for e in entries if e.get("ok") and e.get("wave") == 2}
    log_lines = LOG.read_text().splitlines()[-80:] if LOG.exists() else []
    complete = any("AUTO_RUN_COMPLETE" in ln for ln in log_lines)
    if complete:
        stage = "完成"
    elif len(w1) < WAVE1_TOTAL:
        stage = f"Wave-1 扫描中 ({len(w1)}/{WAVE1_TOTAL})"
    elif len(w2) < 6 * 4:
        stage = f"Wave-2 验证中 ({len(w2)} 相位-配置)"
    else:
        stage = "汇总中"
    running, detail = _running()
    return {
        "wave1_done": len(w1),
        "wave1_total": WAVE1_TOTAL,
        "wave2_done": len(w2),
        "stage": stage,
        "running": running,
        "running_detail": detail,
        "complete": complete,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(LOG.stat().st_mtime)) if LOG.exists() else None,
        "log_tail": log_lines,
    }


@router.get("/results")
def results():
    entries = _read_journal()
    by_idx: dict[int, dict] = {}
    for e in entries:
        if e.get("ok"):
            by_idx.setdefault(e["idx"], {})[e["offset"]] = e
    rows = []
    for idx, offs in sorted(by_idx.items()):
        w1 = offs.get(0)
        if not w1:
            continue
        segs = [o["seg"] for o in offs.values() if o.get("seg")]
        ratios = [o["seg_ratio"] for o in offs.values() if o.get("seg_ratio") is not None]
        fulls = [o["full"] for o in offs.values() if o.get("full")]
        row = {
            "idx": idx,
            "tag": w1["tag"],
            "params": w1["params"],
            "n_phases": len(offs),
            "seg_ann": float(sum(s["ann"] for s in segs) / len(segs)),
            "seg_mdd": float(sum(s["mdd"] for s in segs) / len(segs)),
            "seg_mdd_worst": float(min(s["mdd"] for s in segs)),
            "ratio": float(sum(ratios) / len(ratios)),
            "full_ann": float(sum(f["ann"] for f in fulls) / len(fulls)),
            "trades": int(w1["full"]["trades"]),
            "is_champion": idx == CHAMPION_IDX,
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["ratio"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"rows": rows, "total": len(rows), "champion_idx": CHAMPION_IDX}


@router.get("/curves")
def curves_data():
    if not CURVES.exists():
        return {"available": False}
    data = json.loads(CURVES.read_text())
    data["available"] = True
    return data


@router.post("/run")
def run_sweep():
    running, detail = _running()
    if running:
        raise HTTPException(status_code=409,
                            detail=f"扫描已在运行中 ({detail})，请等待完成或先停止。")
    chain = (f"{PY} -u {CLI} --run 99 && "
             f"{PY} -u {CLI} --validate 6 && "
             f"{PY} -u {CLI} --report")
    log = open(LOG, "a")
    log.write(f"\n=== WEBUI RUN {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    env = {**os.environ, **ENV_OVERRIDES}
    proc = subprocess.Popen(
        ["/bin/bash", "-c", chain],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    STATE.write_text(json.dumps({"pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return {"ok": True, "pid": proc.pid}


@router.post("/curves-gen")
def curves_gen():
    env = {**os.environ, **ENV_OVERRIDES}
    r = subprocess.run([PY, "-u", str(CLI), "--curves"], cwd=str(ROOT),
                       capture_output=True, text=True, env=env, timeout=600)
    ok = r.returncode == 0 and CURVES.exists()
    return {"ok": ok, "stdout_tail": (r.stdout or "").splitlines()[-5:],
            "stderr_tail": (r.stderr or "").splitlines()[-3:]}


@router.post("/report-gen")
def report_gen():
    env = {**os.environ, **ENV_OVERRIDES}
    r = subprocess.run([PY, "-u", str(CLI), "--report"], cwd=str(ROOT),
                       capture_output=True, text=True, env=env, timeout=300)
    md = REPORT.read_text() if REPORT.exists() else ""
    return {"ok": r.returncode == 0, "report_md": md}
