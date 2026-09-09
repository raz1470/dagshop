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

`test_shift_drag_creates_edge` (added session 8, after Ryan reported
"it wont let me draw arrows") is the reason this file's original scope
statement above was a real gap, not just a formality: no test here ever
exercised edge creation, so the drawMode bug fixed in `app.js` this
session shipped straight through a green frontend job. This is still
untested from the bridge Claude runs on -- its network allowlist blocks
Playwright's Chromium download (see NOTES.md sessions 5+) -- so it has
only been validated by static reading of the vendored
cytoscape-edgehandles source, not by actually running it. CI is where
this gets a real signal.
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


def test_shift_drag_creates_edge(live_server, page):
    """Session 8's fix: Shift+drag from one node to another should draw
    a real, signed edge (via the sign modal), matching the on-page hint
    text. Plain drag (no Shift) must still just reposition a node and
    create nothing -- the whole point of the modifier-key gesture is
    that both interactions coexist without a persistent mode toggle.
    """
    console_errors: list[str] = []
    page.on("console", lambda msg: msg.type == "error" and console_errors.append(msg.text))
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))

    page.goto(live_server)
    page.wait_for_selector("#cy canvas")

    node_ids = page.evaluate("() => window.__dagshop.cy.nodes().map((n) => n.id())")
    assert len(node_ids) >= 2
    source_id, target_id = node_ids[0], node_ids[1]

    def node_center(node_id):
        return page.evaluate(
            "(id) => { "
            "const n = window.__dagshop.cy.getElementById(id); "
            "const p = n.renderedPosition(); "
            "const rect = document.getElementById('cy').getBoundingClientRect(); "
            "return { x: rect.left + p.x, y: rect.top + p.y }; "
            "}",
            node_id,
        )

    source = node_center(source_id)
    target = node_center(target_id)

    # Plain drag (no Shift): repositions the source node, creates no edge.
    page.mouse.move(source["x"], source["y"])
    page.mouse.down()
    page.mouse.move(source["x"] + 40, source["y"] + 40, steps=5)
    page.mouse.up()
    edge_count_after_plain_drag = page.evaluate("() => window.__dagshop.cy.edges().length")
    assert edge_count_after_plain_drag == 0

    # Re-fetch source's position: the plain drag above just moved it.
    source = node_center(source_id)

    # Shift+drag: should start an edgehandles gesture ending in the sign
    # modal (server.py's edges always need a user-asserted sign, no
    # default -- see graph.py's add_edge docstring).
    page.keyboard.down("Shift")
    page.mouse.move(source["x"], source["y"])
    page.mouse.down()
    page.mouse.move(target["x"], target["y"], steps=10)
    page.mouse.up()
    page.keyboard.up("Shift")

    page.wait_for_selector("#sign-modal:not(.hidden)")
    page.click("#sign-plus")

    edges = page.evaluate(
        "() => window.__dagshop.cy.edges().map((e) => "
        "({source: e.data('source'), target: e.data('target'), sign: e.data('sign')}))"
    )
    assert {"source": source_id, "target": target_id, "sign": "+"} in edges

    assert console_errors == [], f"console errors during drag: {console_errors}"
