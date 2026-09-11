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

`test_shift_drag_creates_edge` is the reason this file's original scope
statement above was a real gap, not just a formality: no test here ever
exercised edge creation, so the drawMode bug fixed in `app.js` (see
that file's own comments) shipped straight through a green frontend
job. This is still untested from the bridge Claude runs on -- its
network allowlist blocks Playwright's Chromium download (see
NOTES.md) -- so it has only been validated by static reading of the
vendored cytoscape-edgehandles source, not by actually running it. CI
is where this gets a real signal.

`test_causal_build_and_attribute_panel` (SCOPE.md build order
step 6) covers the causal-attribution panel added to `server.py`/`app.js`:
the Build button, the falsification/sign-disagreement
summary, the target-node picker, the ranked contribution table, and
the row-click reuse of this same plot modal. Same bridge limitation as
above -- validated by static reading only here, real signal from CI.

`test_left_panel_tabs` covers the Workshop/Causal model/Causal impact
tab split of `#left-panel`: clicking a tab shows only that tab's panel,
and the canvas (which lives outside `#left-panel`) stays visible and
its nodes stay clickable regardless of which tab is active.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request

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
    # Imported here, not at module level: this module is collected (and
    # so imported) by the `test`/`lint` CI jobs too, which deliberately
    # install only the lean `test` dependency group without playwright
    # (see pyproject.toml's [dependency-groups] comment) -- a top-level
    # import broke that on the first attempt at this diagnostic (session
    # 9: "ModuleNotFoundError: No module named 'playwright'" in the
    # `test` job, which never even runs this function's body since the
    # test is marker-deselected there).
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

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

    # Record edgehandles' own lifecycle events (all emitted on `cy` as
    # "eh" + <name> -- see cytoscape-edgehandles.js's `emit()`) so a
    # failure below can say exactly how far the gesture got, instead of
    # a bare selector timeout. See app.js's `initCytoscape` for why this
    # matters: an earlier version of this test failed at the
    # wait_for_selector below on every CI run because the source node
    # was still natively grabbable when the drag started, which
    # silently prevented edgehandles from ever noticing the target node
    # (cytoscape core skips tapdragover/tapdragout entirely while any
    # node reports grabbed() === true). app.js now arms `drawMode`
    # (autoungrabify) on Shift keydown, before mousedown, which avoids
    # that -- this log is kept so a regression shows up as a clear
    # diagnostic rather than another blind CI round-trip.
    page.evaluate(
        "() => { "
        "window.__ehLog = []; "
        "for (const name of ['ehstart', 'ehpreviewon', 'ehcancel', 'ehcomplete', 'ehstop']) { "
        "  window.__dagshop.cy.on(name, () => window.__ehLog.push(name)); "
        "} "
        "}"
    )

    # Shift+drag: should start an edgehandles gesture ending in the sign
    # modal (server.py's edges always need a user-asserted sign, no
    # default -- see graph.py's add_edge docstring). Shift goes down
    # *before* mousedown on the source node deliberately -- see the
    # app.js comment above `document.addEventListener("keydown", ...)`
    # for why that ordering is now load-bearing, not just convenient.
    page.keyboard.down("Shift")
    page.mouse.move(source["x"], source["y"])
    page.mouse.down()
    page.mouse.move(target["x"], target["y"], steps=10)
    # edgehandles' `hoverDelay: 150` (app.js) defers actually setting
    # `targetNode` until 150ms after the pointer arrives over it -- see
    # the vendored cytoscape-edgehandles.js `preview()`, which schedules
    # `applyPreview` via `setTimeout(..., options.hoverDelay)` rather
    # than setting it synchronously. Wait comfortably past that before
    # mouseup.
    page.wait_for_timeout(250)
    page.mouse.up()
    page.keyboard.up("Shift")

    try:
        page.wait_for_selector("#sign-modal:not(.hidden)", timeout=5000)
    except PlaywrightTimeoutError:
        eh_log = page.evaluate("() => window.__ehLog")
        eh_state = page.evaluate(
            "() => ({ "
            "drawMode: window.__dagshop.eh.drawMode, "
            "active: window.__dagshop.eh.active, "
            "targetNode: window.__dagshop.eh.targetNode && window.__dagshop.eh.targetNode.id ? "
            "window.__dagshop.eh.targetNode.id() : null "
            "})"
        )
        raise AssertionError(
            f"sign modal never opened after Shift+drag from {source_id!r} to "
            f"{target_id!r}. eh lifecycle events seen: {eh_log}. eh state at "
            f"failure: {eh_state}. console errors: {console_errors}"
        ) from None
    page.click("#sign-plus")

    # createEdge() (app.js) awaits a POST to /api/edges before calling
    # cy.add(...) -- give that round-trip a chance to land instead of
    # reading cy.edges() the instant the click handler returns.
    page.wait_for_function("() => window.__dagshop.cy.edges().length > 0")

    edges = page.evaluate(
        "() => window.__dagshop.cy.edges().map((e) => "
        "({source: e.data('source'), target: e.data('target'), sign: e.data('sign')}))"
    )
    assert {"source": source_id, "target": target_id, "sign": "+"} in edges

    assert console_errors == [], f"console errors during drag: {console_errors}"


def _post_json(url: str, payload: dict) -> None:
    """Tiny stdlib POST helper -- `live_server` runs a real uvicorn thread,
    so unlike `test_server.py`'s `TestClient` this needs an actual HTTP
    call, and `page.request` isn't used here to keep this setup step
    independent of Playwright's own request-context quirks. No new
    dependency: `urllib.request` is stdlib.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status in (200, 201), resp.status


def test_causal_build_and_attribute_panel(live_server, page):
    """SCOPE.md build order step 6: Build panel, target-node picker,
    ranked contribution table, row click reusing the plot modal.

    Wires up one real edge first (`age -> outcome`, "+") via a direct
    API call rather than re-driving Shift+drag -- that gesture is
    already covered by `test_shift_drag_creates_edge` above, and this
    test's focus is the causal panel itself, not edge creation.
    """
    console_errors: list[str] = []

    page.goto(live_server)
    page.wait_for_selector("#cy canvas")

    _post_json(f"{live_server}/api/edges", {"source": "age", "target": "outcome", "sign": "+"})
    page.reload()
    page.wait_for_selector("#cy canvas")

    # Console listeners attached after reload: a fresh page load resets
    # any listeners bound to the previous document.
    page.on("console", lambda msg: msg.type == "error" and console_errors.append(msg.text))
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))

    assert "hidden" in (page.get_attribute("#causal-attribute-controls", "class") or "")

    # The build button lives on the "Causal model" tab, hidden by
    # default (the "Workshop" tab is active on load).
    page.click('[data-tab="causal-model"]')
    page.click("#btn-causal-build")

    # Falsification result and (in this fixture, edge-less-until-now
    # DAG so genuinely) empty sign-disagreements section render as soon
    # as the build responds, on the same "Causal model" tab the user
    # is already looking at (no tab switch on build -- see app.js's
    # buildCausalModel comment for why one was tried and reverted).
    page.wait_for_selector("#causal-build-result:not(.hidden)")
    page.wait_for_selector(".causal-falsify-summary")

    # The build auto-runs attribution for the default target (the
    # designated outcome, "outcome") in the background; its result
    # renders on the "Causal impact" tab, one click away.
    page.click('[data-tab="causal-impact"]')
    page.wait_for_selector("#causal-contribution-container table.rank-table tbody tr")
    rows = page.query_selector_all("#causal-contribution-container table.rank-table tbody tr")
    assert len(rows) >= 1
    row_texts = [r.inner_text() for r in rows]
    assert any("age" in t for t in row_texts)

    assert "hidden" not in (page.get_attribute("#causal-attribute-controls", "class") or "")
    assert page.input_value("#causal-target-select") == "outcome"

    # Row click reuses the existing plot modal (SCOPE.md: "reuses the
    # existing plot modal for a PDP-style curve"). Click the "age" row
    # specifically, not just rows[0] -- the ranking's top row can be
    # "outcome" itself (`intrinsic_causal_influence` includes the
    # target's own unexplained variance, per causal_model.py), and a
    # self-pair is never in the association scan's plot cache (it only
    # ever scores ordered pairs of *different* columns), so asserting a
    # real Plotly render below needs a row the scan actually covers.
    # "age -> outcome" is: a scoped scan covers every ordered pair,
    # `outcome_table` included (see associate.py's own decision note),
    # so any covariate as a predictor of the designated outcome is
    # cached.
    age_row_index = next(i for i, t in enumerate(row_texts) if "age" in t)
    rows[age_row_index].click()
    page.wait_for_selector("#plot-modal:not(.hidden)")
    page.wait_for_selector("#plot-modal-body .js-plotly-plot")

    assert console_errors == [], f"console errors in causal panel: {console_errors}"


def test_left_panel_tabs(live_server, page):
    """Workshop is active on load; clicking a tab shows only that tab's
    panel and hides the others. The canvas lives outside #left-panel, so
    a node stays visible and clickable no matter which tab is active --
    checked here via a plain drag on the "Causal model" tab, mirroring
    the plain-drag half of test_shift_drag_creates_edge.
    """
    console_errors: list[str] = []
    page.on("console", lambda msg: msg.type == "error" and console_errors.append(msg.text))
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))

    page.goto(live_server)
    page.wait_for_selector("#cy canvas")

    assert "active" in page.get_attribute('[data-tab="workshop"]', "class")
    assert "hidden" not in (page.get_attribute("#tab-workshop", "class") or "")
    assert "hidden" in (page.get_attribute("#tab-causal-model", "class") or "")
    assert "hidden" in (page.get_attribute("#tab-causal-impact", "class") or "")

    page.click('[data-tab="causal-model"]')
    assert "active" in page.get_attribute('[data-tab="causal-model"]', "class")
    assert "active" not in page.get_attribute('[data-tab="workshop"]', "class")
    assert "hidden" in (page.get_attribute("#tab-workshop", "class") or "")
    assert "hidden" not in (page.get_attribute("#tab-causal-model", "class") or "")
    page.wait_for_selector("#btn-causal-build")

    # Canvas node stays visible and draggable while a non-Workshop tab
    # is active -- #canvas-wrap is a sibling of #left-panel, untouched
    # by the tab switch above.
    node_ids = page.evaluate("() => window.__dagshop.cy.nodes().map((n) => n.id())")
    assert len(node_ids) >= 1
    node_id = node_ids[0]
    before = page.evaluate("(id) => window.__dagshop.cy.getElementById(id).position()", node_id)
    center = page.evaluate(
        "(id) => { "
        "const n = window.__dagshop.cy.getElementById(id); "
        "const p = n.renderedPosition(); "
        "const rect = document.getElementById('cy').getBoundingClientRect(); "
        "return { x: rect.left + p.x, y: rect.top + p.y }; "
        "}",
        node_id,
    )
    page.mouse.move(center["x"], center["y"])
    page.mouse.down()
    page.mouse.move(center["x"] + 30, center["y"] + 30, steps=5)
    page.mouse.up()
    after = page.evaluate("(id) => window.__dagshop.cy.getElementById(id).position()", node_id)
    assert (after["x"], after["y"]) != (before["x"], before["y"])

    page.click('[data-tab="causal-impact"]')
    assert "active" in page.get_attribute('[data-tab="causal-impact"]', "class")
    assert "hidden" in (page.get_attribute("#tab-causal-model", "class") or "")
    assert "hidden" not in (page.get_attribute("#tab-causal-impact", "class") or "")
    # state="attached" rather than the default "visible": this test never
    # triggers a build, so the container is genuinely empty here (no
    # text, zero height) even though its tab is showing -- an empty
    # element with a zero-size box never satisfies Playwright's default
    # visible wait, regardless of display/hidden state.
    page.wait_for_selector("#causal-contribution-container", state="attached")

    page.click('[data-tab="workshop"]')
    assert "active" in page.get_attribute('[data-tab="workshop"]', "class")
    assert "hidden" not in (page.get_attribute("#tab-workshop", "class") or "")
    assert "hidden" in (page.get_attribute("#tab-causal-model", "class") or "")
    assert "hidden" in (page.get_attribute("#tab-causal-impact", "class") or "")

    assert console_errors == [], f"console errors switching tabs: {console_errors}"
