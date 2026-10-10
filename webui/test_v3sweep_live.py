"""WebUI v3sweep tab — raw camofox API end-to-end verification.

Creates/reuses a camofox tab, navigates to the WebUI, clicks the v3sweep tab,
verifies the DOM state (progress/log/table/chart), saves a screenshot.
"""
import json
import pathlib
import sys
import time

import requests

BASE = "http://localhost:9377"
USER = "buffett"
SCRATCH = pathlib.Path("/home/felix/.hermes/profiles/buffett/cache/scratch")
OUT_PNG = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest/evidence/"
                       "v3_sweep_20261010/charts/webui_v3sweep.png")
TID_FILE = SCRATCH / "v3s_tab_id.txt"


def get_tab():
    if TID_FILE.exists():
        tid = TID_FILE.read_text().strip()
        r = requests.get(f"{BASE}/tabs/{tid}/stats", params={"userId": USER}, timeout=15)
        if r.status_code == 200:
            return tid
    r = requests.post(f"{BASE}/tabs", json={"userId": USER, "sessionKey": "v3sweep"}, timeout=40)
    r.raise_for_status()
    tid = r.json()["tabId"]
    TID_FILE.write_text(tid)
    return tid


def ev(tid, expr, t=30):
    r = requests.post(f"{BASE}/tabs/{tid}/evaluate",
                      json={"userId": USER, "expression": expr}, timeout=t)
    r.raise_for_status()
    return r.json()


def main():
    tid = get_tab()
    print("tab:", tid)
    r = requests.post(f"{BASE}/tabs/{tid}/navigate",
                      json={"userId": USER, "url": "http://localhost:8088"}, timeout=40)
    print("navigate:", r.status_code)
    time.sleep(3)

    print("click:", ev(tid, "document.querySelector('[data-tab=\"v3sweep\"]').click(); 'clicked'"))
    time.sleep(8)  # allow boot + curves fetch + chart draw

    state = ev(tid, """
JSON.stringify({
  tabActive: document.querySelector('#tab-v3sweep').classList.contains('active'),
  stage: document.getElementById('v3sStage').textContent,
  progress: document.getElementById('v3sProgress').textContent,
  running: document.getElementById('v3sRunning').textContent,
  logLines: document.getElementById('v3sLog').textContent.split('\\n').filter(x=>x.trim()).length,
  tableRows: document.querySelectorAll('#v3sResultsTable tr').length,
  hint: document.getElementById('v3sChartHint').textContent,
  chart: (window._v3sChart) ? {
    datasets: window._v3sChart.data.datasets.length,
    labels: window._v3sChart.data.labels.length,
    firstLabel: window._v3sChart.data.datasets[0].label
  } : null,
  canvasDataLen: document.getElementById('v3sChart').toDataURL().length
})
""")
    print("state:", json.dumps(state, ensure_ascii=False)[:900])

    r = requests.get(f"{BASE}/tabs/{tid}/screenshot", params={"userId": USER}, timeout=25)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.write_bytes(r.content)
    print(f"screenshot: {OUT_PNG} ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()
