"""Tests for `server.py`: `create_app` and its REST API.

Uses FastAPI's `TestClient` (httpx-based, no live socket) against
`create_app` pointed at a small synthetic CSV written to `tmp_path` per
test. Covers the state-model decisions from server.py's module docstring:
one in-memory `DAGModel`/`AssociationScan` per app, mutation endpoints
mirroring `graph.py`'s API, on-demand plot lookup, and the
KeyError/ValueError/OSError -> 404/400 error mapping.

A live-server, real-browser smoke test lives separately in
`test_frontend_smoke.py` (marker "frontend", skipped by default -- see
`pyproject.toml`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dagshop.graph import DAGModel
from dagshop.server import create_app

RANDOM_STATE = 0


def _write_csv(tmp_path, frame: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def unscoped_csv(tmp_path):
    """Three plain continuous columns, no treatment/outcome designated."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 60
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    return _write_csv(tmp_path, frame)


@pytest.fixture
def scoped_csv(tmp_path):
    """One treatment, one outcome, plus two plain covariates."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 80
    treated = rng.integers(0, 2, size=n)
    frame = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "prior_engagement": rng.normal(size=n),
            "treated": treated,
            "outcome": rng.normal(size=n) + treated * 2.0,
        }
    )
    return _write_csv(tmp_path, frame)


@pytest.fixture
def client(unscoped_csv):
    app = create_app(unscoped_csv, random_state=RANDOM_STATE)
    return TestClient(app)


@pytest.fixture
def scoped_client(scoped_csv):
    app = create_app(
        scoped_csv, treatments=["treated"], outcomes=["outcome"], random_state=RANDOM_STATE
    )
    return TestClient(app)


def _first_pair(client: TestClient) -> tuple[str, str]:
    row = client.get("/api/tables").json()["full_table"][0]
    return row["predictor"], row["target"]


# -- app factory / initial layout ------------------------------------------------


def test_create_app_builds_one_node_per_column(client):
    graph = client.get("/api/graph").json()
    assert {n["name"] for n in graph["nodes"]} == {"a", "b", "c"}
    assert graph["edges"] == []


def test_scoped_app_pins_treatment_and_outcome_positions(scoped_client):
    graph = scoped_client.get("/api/graph").json()
    by_name = {n["name"]: n for n in graph["nodes"]}
    assert by_name["treated"]["role"] == "treatment"
    assert by_name["outcome"]["role"] == "outcome"
    assert by_name["treated"]["x"] == pytest.approx(150.0)
    assert by_name["outcome"]["x"] == pytest.approx(1050.0)
    # unlinked nodes are scattered, not pinned to either fixed column
    assert by_name["age"]["role"] is None
    assert by_name["age"]["x"] not in (150.0, 1050.0)
    assert graph["treatments"] == ["treated"]
    assert graph["outcomes"] == ["outcome"]


def test_overlapping_treatment_and_outcome_rejected(scoped_csv):
    with pytest.raises(ValueError, match="both a treatment and an outcome"):
        create_app(scoped_csv, treatments=["outcome"], outcomes=["outcome"])


def test_initial_session_resume_overrides_fresh_layout(unscoped_csv, tmp_path):
    # Build a session with a node moved off its fresh-layout position and
    # an edge added, save it, then confirm create_app(initial_session=...)
    # serves that saved state rather than a freshly scattered layout.
    # Session-5 decision: the association scan still runs against
    # `unscoped_csv` regardless (checked via /api/tables below), since a
    # loaded session never touches the scan.
    saved = DAGModel()
    for name in ("a", "b", "c"):
        saved.add_node(name)
    saved.set_position("a", x=999.0, y=888.0)
    saved.add_edge("a", "b", sign="+")
    session_path = tmp_path / "session.json"
    saved.save_session(session_path)

    app = create_app(unscoped_csv, random_state=RANDOM_STATE, initial_session=session_path)
    client = TestClient(app)

    graph = client.get("/api/graph").json()
    by_name = {n["name"]: n for n in graph["nodes"]}
    assert by_name["a"]["x"] == pytest.approx(999.0)
    assert by_name["a"]["y"] == pytest.approx(888.0)
    assert graph["edges"] == [{"source": "a", "target": "b", "sign": "+"}]

    # scan still ran against unscoped_csv, untouched by the loaded session
    assert client.get("/api/tables").json()["scoped"] is False


def test_initial_session_missing_file_raises(unscoped_csv, tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        create_app(unscoped_csv, initial_session=missing)


def test_multiple_treatments_and_outcomes_are_evenly_spaced(tmp_path):
    rng = np.random.default_rng(RANDOM_STATE)
    n = 50
    frame = pd.DataFrame(
        {
            "t1": rng.integers(0, 2, size=n),
            "t2": rng.integers(0, 2, size=n),
            "o1": rng.normal(size=n),
            "o2": rng.normal(size=n),
            "cov": rng.normal(size=n),
        }
    )
    csv_path = _write_csv(tmp_path, frame)
    app = create_app(
        csv_path, treatments=["t1", "t2"], outcomes=["o1", "o2"], random_state=RANDOM_STATE
    )
    client = TestClient(app)
    by_name = {n["name"]: n for n in client.get("/api/graph").json()["nodes"]}

    assert by_name["t1"]["x"] == by_name["t2"]["x"] == pytest.approx(150.0)
    assert by_name["o1"]["x"] == by_name["o2"]["x"] == pytest.approx(1050.0)

    treatment_ys = sorted(by_name[n]["y"] for n in ("t1", "t2"))
    assert treatment_ys == [pytest.approx(100.0), pytest.approx(700.0)]
    outcome_ys = sorted(by_name[n]["y"] for n in ("o1", "o2"))
    assert outcome_ys == [pytest.approx(100.0), pytest.approx(700.0)]


# -- health / tables / plot ------------------------------------------------------


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_columns"] == 3
    assert body["n_rows"] == 60
    assert body["scoped"] is False


def test_tables_unscoped_uses_full_table(client):
    body = client.get("/api/tables").json()
    assert body["scoped"] is False
    assert body["full_table"] != []
    assert body["treatment_table"] == []
    assert body["outcome_table"] == []
    assert body["covariate_table"] == []
    row = body["full_table"][0]
    assert set(row) == {"predictor", "target", "score", "score_name", "n_used"}


def test_tables_scoped_splits_treatment_and_outcome(scoped_client):
    body = scoped_client.get("/api/tables").json()
    assert body["scoped"] is True
    assert body["full_table"] == []
    assert {row["target"] for row in body["treatment_table"]} == {"treated"}
    assert {row["target"] for row in body["outcome_table"]} == {"outcome"}
    # covariates = {age, prior_engagement}: every ordered pair, 2 * 1 = 2.
    assert {(row["predictor"], row["target"]) for row in body["covariate_table"]} == {
        ("age", "prior_engagement"),
        ("prior_engagement", "age"),
    }


def test_plot_found_for_scanned_pair(client):
    predictor, target = _first_pair(client)
    body = client.get(f"/api/plot/{predictor}/{target}").json()
    assert body["predictor"] == predictor
    assert body["target"] == target
    assert len(body["x"]) == len(body["y"])
    assert len(body["grid_x"]) == len(body["grid_prediction"])


def test_plot_missing_pair_is_404(client):
    resp = client.get("/api/plot/a/a")  # self-pairs are never scanned
    assert resp.status_code == 404


# -- node mutations ---------------------------------------------------------------


def test_set_position(client):
    resp = client.put("/api/nodes/a/position", json={"x": 12.5, "y": -3.0})
    assert resp.status_code == 200
    node = _node(client, "a")
    assert node["x"] == 12.5
    assert node["y"] == -3.0


def test_set_position_missing_node_is_404(client):
    resp = client.put("/api/nodes/nope/position", json={"x": 0, "y": 0})
    assert resp.status_code == 404


def test_set_role(client):
    resp = client.put("/api/nodes/a/role", json={"role": "treatment"})
    assert resp.status_code == 200
    assert client.get("/api/graph").json()["treatments"] == ["a"]


def test_set_role_clears_with_null(client):
    client.put("/api/nodes/a/role", json={"role": "treatment"})
    client.put("/api/nodes/a/role", json={"role": None})
    assert client.get("/api/graph").json()["treatments"] == []


def test_set_role_invalid_value_is_422(client):
    resp = client.put("/api/nodes/a/role", json={"role": "not-a-role"})
    assert resp.status_code == 422


def test_delete_node_removes_incident_edges(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    resp = client.delete("/api/nodes/a")
    assert resp.status_code == 204
    graph = client.get("/api/graph").json()
    assert {n["name"] for n in graph["nodes"]} == {"b", "c"}
    assert graph["edges"] == []


def test_delete_node_missing_is_404(client):
    resp = client.delete("/api/nodes/nope")
    assert resp.status_code == 404


def _node(client: TestClient, name: str) -> dict:
    graph = client.get("/api/graph").json()
    return next(n for n in graph["nodes"] if n["name"] == name)


# -- edges -------------------------------------------------------------------------


def test_add_edge(client):
    resp = client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    assert resp.status_code == 201
    assert resp.json() == {"source": "a", "target": "b", "sign": "+", "cycle_warning": None}
    assert client.get("/api/graph").json()["edges"] == [{"source": "a", "target": "b", "sign": "+"}]


def test_add_edge_missing_node_is_404(client):
    resp = client.post("/api/edges", json={"source": "a", "target": "nope", "sign": "+"})
    assert resp.status_code == 404


def test_add_edge_self_loop_is_400(client):
    resp = client.post("/api/edges", json={"source": "a", "target": "a", "sign": "+"})
    assert resp.status_code == 400


def test_add_edge_invalid_sign_is_422(client):
    resp = client.post("/api/edges", json={"source": "a", "target": "b", "sign": "?"})
    assert resp.status_code == 422


def test_add_edge_reports_cycle_warning(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    client.post("/api/edges", json={"source": "b", "target": "c", "sign": "-"})
    resp = client.post("/api/edges", json={"source": "c", "target": "a", "sign": "+"})
    warning = resp.json()["cycle_warning"]
    assert warning is not None
    assert "a" in warning
    assert "b" in warning


def test_add_edge_acyclic_reports_no_cycle_warning(client):
    resp = client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    assert resp.json()["cycle_warning"] is None


def test_remove_edge(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    resp = client.delete("/api/edges/a/b")
    assert resp.status_code == 204
    assert client.get("/api/graph").json()["edges"] == []


def test_remove_edge_missing_is_404(client):
    resp = client.delete("/api/edges/a/b")
    assert resp.status_code == 404


def test_set_sign(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    resp = client.put("/api/edges/a/b/sign", json={"sign": "-"})
    assert resp.status_code == 200
    assert client.get("/api/graph").json()["edges"] == [{"source": "a", "target": "b", "sign": "-"}]


def test_set_sign_missing_edge_is_404(client):
    resp = client.put("/api/edges/a/b/sign", json={"sign": "-"})
    assert resp.status_code == 404


def test_set_sign_invalid_value_is_422(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    resp = client.put("/api/edges/a/b/sign", json={"sign": "?"})
    assert resp.status_code == 422


# -- validate / export -------------------------------------------------------------


def test_validate_acyclic(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    assert client.get("/api/validate").json() == {"valid": True, "cycles": []}


def test_validate_cyclic(client):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    client.post("/api/edges", json={"source": "b", "target": "a", "sign": "-"})
    body = client.get("/api/validate").json()
    assert body["valid"] is False
    assert len(body["cycles"]) == 1


def test_export_json(client, tmp_path):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    out = tmp_path / "export.json"
    resp = client.post("/api/export", json={"path": str(out), "format": "json"})
    assert resp.status_code == 200
    data = json.loads(out.read_text())
    assert data["edges"] == [{"source": "a", "target": "b", "sign": "+"}]


def test_export_graphml(client, tmp_path):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    out = tmp_path / "export.graphml"
    resp = client.post("/api/export", json={"path": str(out), "format": "graphml"})
    assert resp.status_code == 200
    assert out.exists()


def test_export_cyclic_graph_is_400_and_writes_nothing(client, tmp_path):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    client.post("/api/edges", json={"source": "b", "target": "a", "sign": "-"})
    out = tmp_path / "export.json"
    resp = client.post("/api/export", json={"path": str(out), "format": "json"})
    assert resp.status_code == 400
    assert not out.exists()


def test_export_bad_path_is_400(client, tmp_path):
    out = tmp_path / "no-such-dir" / "export.json"
    resp = client.post("/api/export", json={"path": str(out), "format": "json"})
    assert resp.status_code == 400


# -- session save/load --------------------------------------------------------------


def test_session_round_trip(client, tmp_path):
    client.post("/api/edges", json={"source": "a", "target": "b", "sign": "+"})
    client.put("/api/nodes/c/position", json={"x": 5.0, "y": 6.0})
    session_path = tmp_path / "session.json"
    resp = client.post("/api/session/save", json={"path": str(session_path)})
    assert resp.status_code == 200
    assert session_path.exists()

    # mutate further, then reload and confirm the save point wins
    client.post("/api/edges", json={"source": "b", "target": "c", "sign": "-"})
    resp = client.post("/api/session/load", json={"path": str(session_path)})
    assert resp.status_code == 200
    assert client.get("/api/graph").json()["edges"] == [{"source": "a", "target": "b", "sign": "+"}]


def test_session_save_bad_path_is_400(client, tmp_path):
    out = tmp_path / "no-such-dir" / "session.json"
    resp = client.post("/api/session/save", json={"path": str(out)})
    assert resp.status_code == 400


def test_session_load_missing_file_is_404(client, tmp_path):
    resp = client.post("/api/session/load", json={"path": str(tmp_path / "nope.json")})
    assert resp.status_code == 404


def test_session_load_does_not_touch_tables(client, tmp_path):
    """Loading a session restores graph state only, not the association
    scan (server.py docstring: the scan is tied to the data.csv loaded
    at startup, not to whatever session file gets loaded later)."""
    before = client.get("/api/tables").json()
    session_path = tmp_path / "session.json"
    client.post("/api/session/save", json={"path": str(session_path)})
    client.post("/api/session/load", json={"path": str(session_path)})
    after = client.get("/api/tables").json()
    assert before == after


# -- static assets ------------------------------------------------------------------


def test_index_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "DAGshop" in resp.text


@pytest.mark.parametrize(
    "path",
    [
        "/static/vendor/cytoscape/cytoscape.min.js",
        "/static/vendor/cytoscape-edgehandles/cytoscape-edgehandles.js",
        "/static/vendor/plotly/plotly-basic.min.js",
        "/static/js/app.js",
        "/static/js/compat-shim.js",
        "/static/css/styles.css",
    ],
)
def test_vendored_and_app_assets_served_locally(client, path):
    # No CDN, per SCOPE.md: every frontend asset the page references
    # must be servable from this app's own /static mount.
    resp = client.get(path)
    assert resp.status_code == 200
