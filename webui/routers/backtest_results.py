"""Tab 1: Backtest Results — read-only display of v1+v2 workflow evidence files.

This is the simplest tab (no CLI spawning). It loads per-stage JSON files
and displays them in a clean table.

v1+v2 final workflow stage directory pattern:
  evidence/factor_research_workflow_v1_stage{3,4,5b,5c,6a}_*/per_leg_bull*.json
  evidence/factor_research_workflow_v2_stage7a_kill{4,6,8,10,12}_*/per_leg_bull_topk10_kill*.json
  evidence/factor_research_workflow_v2_stage7c_cooldown{10,30,40,60}_*/per_leg_bull_topk10.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
EVIDENCE_ROOT = ROOT / "evidence"


def _aggregate_per_leg(results: list[dict]) -> dict:
    """Compute aggregate stats from per-leg records."""
    valid = [r for r in results if "closed_only_return_pct" in r]
    if not valid:
        return {"n_legs": 0}
    n_target = sum(
        1 for r in valid
        if r.get("sharpe_ratio", 0) >= 1.2
        and r.get("max_drawdown_pct", 100) <= 12
        and ((1 + r["closed_only_return_pct"] / 100) ** (365 / r["days"]) - 1) * 100 >= 12
    )
    ann_list = [
        ((1 + r["closed_only_return_pct"] / 100) ** (365 / r["days"]) - 1) * 100
        for r in valid
    ]
    return {
        "n_legs": len(valid),
        "n_positive": sum(1 for r in valid if r["closed_only_return_pct"] > 0),
        "n_meet_target": n_target,
        "mean_closed_only_pct": round(statistics.mean(r["closed_only_return_pct"] for r in valid), 2),
        "mean_annualized_pct": round(statistics.mean(ann_list), 1),
        "median_annualized_pct": round(statistics.median(ann_list), 1),
        "mean_sharpe": round(statistics.mean(r["sharpe_ratio"] for r in valid), 3),
        "mean_mdd_pct": round(statistics.mean(r["max_drawdown_pct"] for r in valid), 2),
        "median_mdd_pct": round(statistics.median(r["max_drawdown_pct"] for r in valid), 2),
        "mean_trades": round(statistics.mean(r["closed_trade_count"] for r in valid), 0),
    }


def _discover_stages() -> list[dict]:
    """Discover all per-leg bull backtest files matching the v1+v2 naming pattern."""
    stages = []
    for d in sorted(EVIDENCE_ROOT.glob("factor_research_workflow_v*_*")):
        for f in sorted(d.glob("per_leg*.json")):
            if "bull" in f.name or "kill" in f.name or "topk" in f.name:
                stages.append({"dir": d.name, "file": f.name, "path": str(f.relative_to(ROOT))})
    return stages


@router.get("/stages")
def list_stages():
    """List all discovered v1+v2 workflow stages with their aggregate stats."""
    discovered = _discover_stages()
    items = []
    for entry in discovered:
        full_path = ROOT / entry["path"]
        try:
            data = json.loads(full_path.read_text())
            if isinstance(data, list):
                agg = _aggregate_per_leg(data)
                items.append({
                    "stage_id": entry["dir"].replace("factor_research_workflow_", ""),
                    "file": entry["file"],
                    "path": entry["path"],
                    "aggregate": agg,
                })
        except Exception as exc:
            items.append({
                "stage_id": entry["dir"],
                "file": entry["file"],
                "path": entry["path"],
                "error": str(exc),
            })
    return {"stages": items, "count": len(items)}


@router.get("/stage/{dir_name}/{file_name}")
def get_stage_detail(dir_name: str, file_name: str):
    """Return the full per-leg data for a specific stage."""
    full_path = EVIDENCE_ROOT / dir_name / file_name
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {full_path}")
    try:
        data = json.loads(full_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"parse error: {exc}")
    if not isinstance(data, list):
        return {"raw": data}
    return {
        "stage_id": dir_name.replace("factor_research_workflow_", ""),
        "file": file_name,
        "aggregate": _aggregate_per_leg(data),
        "legs": data,
    }


@router.get("/summary")
def get_summary():
    """Return v1+v2 workflow target achievement summary."""
    return {
        "target": {"annualized": ">= 12%", "sharpe": ">= 1.2", "mdd": "<= 12%"},
        "achieved": {
            "annualized": "EXCEEDED (mean +134% across v1+v2 sweeps)",
            "sharpe": "EXCEEDED (mean 2.93)",
            "mdd": "NEAR MISS (mean 17.67%, gap 5.67pp)",
        },
        "terminal_state": "Stage 5b (top_k=10, kill DD>8%/cooldown=20d, 6-factor bull_composite)",
        "production_realistic": "FAILED (Stage 6a full panel ann +2%, MDD -63%)",
        "interpretation": (
            "bull_composite is a bull-leg specialist, not a standalone production strategy. "
            "Per-leg alpha is real and large; full-panel exposure shows bull-leg-internal "
            "bear segments that kill-switch cannot control."
        ),
    }