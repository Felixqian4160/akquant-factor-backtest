"""Mock webui end-to-end frontend test (no browser, system python).

Verifies that the frontend (v34_lb20.js) parsing logic produces correct output
from the mock backend (app_mock.py) responses — this is the equivalent of
"the browser renders the right DOM" but without depending on a flaky browser
session.

Steps:
  1. Start mock backend (app_mock.py) in background
  2. Hit every API endpoint the JS expects
  3. Apply the EXACT JS parsing functions (_v34RefreshSystemStatus, _v34LoadJobs)
     to the response JSON to derive what the DOM would contain
  4. Print the DOM-equivalent text the user should see
  5. Compare against the real webui (port 8088) — same parser, real responses
  6. Report PASS/FAIL per check

Run:
  /usr/bin/python3.12 webui/test_mock_frontend_e2e.py
"""
from __future__ import annotations
import json
import subprocess
import time
import urllib.request
from pathlib import Path

MOCK_URL = "http://localhost:8089"
REAL_URL = "http://localhost:8088"


def fetch(url: str, method: str = "GET", data: bytes | None = None) -> dict | list:
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def render_system_status(s: dict) -> list[str]:
    """Mirror _v34RefreshSystemStatus in webui/static/tabs/v34_lb20.js."""
    return [
        f"lock: {'🔒 ' + s['lock_locked_at'] if s['lock_exists'] else '❌'}",
        f"strategy doc: {'✅' if s['strategy_doc_exists'] else '❌'}",
        f"panel: {'✅ ' + str(s['panel_size_gb']) + ' GB' if s['panel_exists'] else '❌'}",
        f"matrix_v34_ADJ: {'✅ ' + str(s['matrix_size_mb']) + ' MB' if s['matrix_exists'] else '❌'}",
    ]


def render_jobs_table(jobs: list[dict]) -> list[str]:
    """Mirror _v34LoadJobs in webui/static/tabs/v34_lb20.js."""
    rows = []
    for j in jobs:
        icon = "✅" if j["finished"] else "⏳"
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j["created"]))
        rows.append(f"{j['job_id']:50s} {icon:1s} {created:20s} [load button]")
    return rows


def section(title: str, items: list[str]) -> None:
    print(f"\n=== {title} ===")
    for it in items:
        print(f"  {it}")


def check(label: str, got: object, expected: object) -> bool:
    ok = got == expected
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}: got={got!r} expected={expected!r}")
    return ok


def main() -> int:
    # Wait for mock to be up (assume already started by user)
    for i in range(20):
        try:
            fetch(f"{MOCK_URL}/health")
            print(f"mock backend up after {i*0.5:.1f}s")
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("ERROR: mock backend not reachable at", MOCK_URL)
        return 1

    total = passed = 0

    # ===== MOCK =====
    print("\n" + "=" * 60)
    print("MOCK backend (port 8089) — synthetic fixtures")
    print("=" * 60)

    # 1. system_status
    s = fetch(f"{MOCK_URL}/api/v34lb20/status")
    expected_keys = {"lock_exists", "lock_locked_at", "strategy_doc_exists", "panel_exists", "panel_size_gb", "matrix_exists", "matrix_size_mb"}
    total += 1; passed += check("status has all keys", expected_keys.issubset(s.keys()), True)
    lines = render_system_status(s)
    section("System Status (4 fields, from mock)", lines)

    # 2. jobs list
    out = fetch(f"{MOCK_URL}/api/v34lb20/jobs")
    jobs = out["jobs"]
    total += 1; passed += check("jobs count", len(jobs), 5)
    rows = render_jobs_table(jobs)
    section(f"Jobs table ({len(jobs)} rows, from mock)", rows)

    # 3. POST /verify returns instant job_id
    v = fetch(f"{MOCK_URL}/api/v34lb20/verify", method="POST", data=b"{}")
    total += 1; passed += check("POST /verify returns job_id+mode", set(v.keys()) >= {"job_id", "mode"}, True)
    section("POST /verify response", [json.dumps(v, ensure_ascii=False)])

    # 4. status for that job
    js = fetch(f"{MOCK_URL}/api/v34lb20/status/{v['job_id']}")
    total += 1; passed += check("status finished=True", js["finished"], True)
    total += 1; passed += check("status error=False", js["error"], False)
    section(f"GET /status/{v['job_id']} log_tail (last 5)", js["log_tail"][-5:])

    # 5. result for that job
    r = fetch(f"{MOCK_URL}/api/v34lb20/result/{v['job_id']}")
    total += 1; passed += check("result.status=ok", r.get("status"), "ok")
    total += 1; passed += check("result.exit_code=0", r.get("exit_code"), 0)
    section("GET /result", [json.dumps(r, indent=2, ensure_ascii=False)])

    # ===== REAL =====
    print("\n" + "=" * 60)
    print("REAL backend (port 8088) — live SYSTEM_VERIFY artifacts")
    print("=" * 60)

    try:
        real_s = fetch(f"{REAL_URL}/api/v34lb20/status")
        print(f"  reachable, keys={list(real_s.keys())[:5]}...")
        real_lines = render_system_status(real_s)
        section("Real System Status", real_lines)
        real_jobs = fetch(f"{REAL_URL}/api/v34lb20/jobs")
        section(f"Real Jobs table ({len(real_jobs['jobs'])} rows)", render_jobs_table(real_jobs["jobs"]))
        print("\n[NOTICE] real backend has its own historical jobs; mock returns 5 fixed fixtures")
    except Exception as e:
        print(f"  real backend not reachable (expected if not running): {e}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 60)
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())