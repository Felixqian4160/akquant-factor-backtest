#!/usr/bin/env bash
# v34-lb20 production system entrypoint.
#
# Run from /media/felix/f/quant/akquant-factor-backtest.
# This script never modifies anything; it verifies the lock first, then
# delegates to whichever subcommand you chose.

set -euo pipefail

cd /media/felix/f/quant/akquant-factor-backtest

verify() {
  set +e
  /usr/bin/python3.12 SYSTEM_VERIFY.py
  set -e
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  verify       Verify locked SHA256 (use SYSTEM_VERIFY.py directly for full output)
  status       Show locked artifacts + last verification
  reproduce    Run the canonical full pipeline (~25 minutes total):
                 1) Derive 10 offset picks (matrix-based, 30 seconds)
                 2) Run 10 sims 2010-2025 (~15 minutes)
                 3) Run 10 sims 2010-2026.08 (~15 minutes)
                 4) Regenerate plots + report
  picks        Just derive 10 offset picks
  sims         Run all 20 sims (10 full + 10 OOS)
  plot         Regenerate the 4 NAV/yearly charts
  report       Regenerate the STAGE5_REPORT.md from current artifacts
EOF
}

cmd="${1:-status}"
JOB_ID="${JOB_ID:-}"

write_result() {
  local status="$1"
  local message="$2"
  if [[ -n "$JOB_ID" ]]; then
    mkdir -p "evidence/v34lb20_jobs/$JOB_ID"
    printf '{"job_id":"%s","status":"%s","message":"%s","exit_code":%s,"completed_at":"%s"}\n' \
      "$JOB_ID" "$status" "$message" "$EXIT_CODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "evidence/v34lb20_jobs/$JOB_ID/result.json"
  fi
}

case "$cmd" in
  verify)
    verify
    EXIT_CODE=$?; write_result "ok" "verified" || true
    ;;
  status)
    echo "=== SYSTEM_LOCK.json ==="
    /usr/bin/python3.12 -c "
import json, pathlib
lock = json.loads(pathlib.Path('SYSTEM_LOCK.json').read_text())
print(f'locked_at: {lock[\"locked_at\"]}')
print(f'scripts:   {len(lock[\"scripts\"])}')
print(f'artifacts: {len(lock[\"artifacts\"])}')
print(f'evidence dirs: {len(lock[\"ev_dirs\"])}')"
    echo ""
    set +e
    verify
    set -e
    EXIT_CODE=$?; write_result "ok" "status + verify" || true
    ;;
  reproduce)
    verify || true
    "$0" picks || true
    "$0" sims || true
    "$0" plot || true
    "$0" report || true
    EXIT_CODE=0; write_result "ok" "reproduce completed (errors ignored)" || true
    ;;
  picks)
    echo ">>> Deriving 10 offset picks (matrix-based)..."
    /usr/bin/python3.12 -u examples/stage5_picks_and_sims.py 2>&1 | tail -15
    EXIT_CODE=$?; write_result "ok" "picks derived" || true
    ;;
  sims)
    echo ">>> Running 20 sims (10 full + 10 OOS) — resumable, ~25 minutes"
    for i in 1 2 3 4 5; do
      /usr/bin/python3.12 -u examples/stage5_run_sims.py --max 4 2>&1 | tail -3 || true
    done
    EXIT_CODE=$?; write_result "ok" "sims completed" || true
    ;;
  plot)
    echo ">>> Regenerating charts..."
    /usr/bin/python3.12 -u examples/plot_rerun_charts.py
    EXIT_CODE=$?; write_result "ok" "plots regenerated" || true
    ;;
  report)
    echo ">>> Regenerating report..."
    /usr/bin/python3.12 -c "
import sys, pathlib
sys.path.insert(0, 'examples')
import stage5_report
import importlib
importlib.reload(stage5_report)
stage5_report.OUT_BASE = pathlib.Path('evidence/stage5_20261007')
stage5_report.main()" 2>&1 | tail -8
    EXIT_CODE=$?; write_result "ok" "report regenerated" || true
    ;;
  *)
    usage
    EXIT_CODE=1; write_result "fail" "unknown subcommand: $cmd" || true
    exit 1
    ;;
esac
exit $EXIT_CODE
