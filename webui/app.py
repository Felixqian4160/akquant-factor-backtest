"""AKQuant v2 WebUI — entry point.

Run:
    /usr/bin/python3.12 /media/felix/f/quant/akquant-factor-backtest/webui/app.py

Then open http://localhost:8088 in browser.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project paths are importable.
ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Routers — register each as it becomes available.
from routers import (
    backtest_results,
    walk_forward,
    daily_strategy,
    paper_trading,
    daily_factor_reselect,
)  # noqa: E402

app = FastAPI(title="AKQuant v2 Workflow", version="0.1.0")

# Static files (CSS, JS).
app.mount(
    "/static",
    StaticFiles(directory=str(ROOT / "webui" / "static")),
    name="static",
)

# Routers.
app.include_router(backtest_results.router)
app.include_router(walk_forward.router)
app.include_router(daily_strategy.router)
app.include_router(paper_trading.router)
app.include_router(daily_factor_reselect.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "akquant-v2-webui", "port": 8088}


@app.get("/")
def root():
    """Serve the main page (single-page app with 4 tabs)."""
    from fastapi.responses import FileResponse
    return FileResponse(str(ROOT / "webui" / "static" / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")