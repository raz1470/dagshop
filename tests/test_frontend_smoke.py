"""Live-server, real-browser smoke test for the DAG workshop UI.

Marker "frontend" (see `pyproject.toml`): skipped by default, since it
needs Playwright's Chromium download, not just Python packages. Run it
explicitly:

    uv sync --group frontend
    uv run playwright install --with-deps chromium
    uv run pytest -m frontend

Mirrors `how_wrong_is_your_mmm`'s headless-browser chart-legibility check
(PREFERENCES.md's precedent for a frontend CI job): not a full UI test
suite, just confirmation that the page actually renders once
`static/`'s vendored JS is real -- the ranking table has rows, the
Cytoscape canvas has drawn nodes with the right treatment/outcome roles,
and nothing throws a console error on load.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest
import uvicorn

from dagshop.server import create_app

pytestmark = pytest.mark.frontend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    """A real uvicorn server, in a background thread, serving `create_app`.

    Playwright drives an actual browser and needs an actual HTTP URL to
    navigate to -- `TestClient` (used in test_server.py) doesn't open a
    real socket, so it can't be reused here.
    """
    rng = np.random.default_rng(0)
    n = 60
    treated = rng.integers(0, 2, size=n)
    frame = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treated": treated,
            "outcome": rng.normal(size=n) + treated * 2.0,
            "other": rng.normal(size=n),
        }
    )
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    app = create_app(csv_path, treatments=["treated"], outcomes=["outcome"], random_state=0)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start within 10s"

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_workshop_page_renders(live_server, page):
    console_errors: list[str] = []
    page.on("console", lambda msg: msg.type == "error" and console_errors.append(msg.text))
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))

    page.goto(live_server)

    # Ranking tables: scoped scan (treated + outcome designated), so both
    # side-by-side tables should render with at least one row.
    page.wait_for_selector("#tables-container table.rank-table tbody tr")
    rows = page.query_selector_all("#tables-container table.rank-table tbody tr")
    assert len(rows) > 0

    # Cytoscape canvas: drawn once nodes exist and the container has a
    # real size (one <canvas> element per cytoscape rendering layer).
    page.wait_for_selector("#cy canvas")
    canvases = page.query_selector_all("#cy canvas")
    assert len(canvases) > 0

    # Treatment/outcome roles reached the canvas (app.js exposes the
    # live cytoscape instance at window.__dagshop.cy for this check).
    node_roles = page.evaluate("() => window.__dagshop.cy.nodes().map((n) => n.data('role'))")
    assert "treatment" in node_roles
    assert "outcome" in node_roles

    assert console_errors == [], f"console errors on load: {console_errors}"
