"""FastAPI app serving the DAG workshop UI.

Vendored Cytoscape.js frontend (drag/drop nodes, click-drag directional
edges via cytoscape-edgehandles) and Plotly.js for pairwise association
plots, both bundled under `static/vendor/`, no CDN (see
`static/vendor/VENDOR.md`). Ranking tables from `associate.py`, plot
modals, DAG canvas backed by `graph.py`. Local-only (localhost), no
external network calls, per SCOPE.md's data handling and security
constraints.

State model: a single server-side,
in-memory session. `create_app` loads `data.csv` once, runs the
association scan once, and builds one `DAGModel` -- all held as closures
over the route handlers, not per-request or per-client state. This
matches SCOPE.md: DAGshop is a single-device, single-workshop-session
tool, so there is no need for session IDs or multi-tenant state. The
frontend calls one endpoint per user action (add edge, move a node, set
a sign, ...) rather than round-tripping the whole graph.

Data entry: `create_app(data_path, treatments=..., outcomes=...)`
is the whole boundary with `cli.py` (SCOPE.md build order step 4, not
built yet). `cli.py`'s job will be exactly: parse
`dagshop launch data.csv [--treatment X ...] [--outcome Y ...]` and call
this factory, then run it with uvicorn. No upload endpoint: the CSV is
read from disk before the server ever starts serving requests.

Plot wire format: `AssociationScan.plot_cache` is keyed by
`(predictor, target)` tuples and is never bulk-serialized. Instead
`GET /api/plot/{predictor}/{target}` looks up one entry on demand, fetched
only when a ranking-table row or a canvas edge is clicked. This matters
once `n` is 50+ (SCOPE.md's stated scale target): the cache can hold
`n * (n - 1)` entries, each with up to `max_rows` scatter points.

Needs graph.py and associate.py working first, since it serves their
outputs. See SCOPE.md build order step 3.

Causal attribution (SCOPE.md build order step 5): two more
endpoints on top of the same in-memory state model above, not a second
session concept. `POST /api/causal/build` calls `causal_model.fit_causal_model`
against the *current* `dag` (whatever the workshop has edited it to by the
time build is clicked, not the startup snapshot) and `data`, then
`causal_model.falsify_causal_graph`, and caches the fitted model in a
closure variable (`causal_model_state`) the same way `dag` itself is
cached -- "cache the result" per SCOPE.md's build-order wording. Editing
the graph after a build does not invalidate that cache automatically
(matches the rest of this file: nothing here auto-invalidates the
association scan either); the workshop rebuilds by calling the endpoint
again. `GET /api/causal/attribute/{target_node}` reads that cache and
raises a plain 400 (via `HTTPException`, same pattern as the export/
session-load routes) if nothing has been built yet, rather than silently
building on first use -- fitting is not cheap enough to hide behind a GET.
Both routes reuse the existing `KeyError`/`ValueError` exception handlers
above rather than adding new ones: `causal_model.py`'s own errors
(`GraphValidationError` for a cyclic graph, `ColumnTypeError` for a
non-numeric column, `KeyError` for an unknown target node) already map
cleanly to 400/404 through them.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dagshop.associate import scan_associations
from dagshop.causal_model import (
    FittedCausalModel,
    attribute_target,
    falsify_causal_graph,
    fit_causal_model,
)
from dagshop.graph import CycleWarning, DAGModel, GraphValidationError, Role, Sign

STATIC_DIR = Path(__file__).parent / "static"

# Layout convention for the initial, server-assigned node positions (an
# arbitrary virtual canvas; the frontend pans/zooms/rescales regardless).
# Treatment nodes are pinned in a left column, outcome nodes in a right
# column (SCOPE.md: "visually pinned and distinctly colored, giving the
# workshop a fixed structure to build around"). Every other node starts
# scattered in the middle band, seeded by `random_state` so a given
# dataset/role selection produces the same starting layout on every
# launch, per SCOPE.md's step 2 "unlinked, scattered nodes" -- reproducible
# scatter beats re-randomizing on every server restart.
_LAYOUT_PINNED_Y_RANGE = (100.0, 700.0)
_LAYOUT_TREATMENT_X = 150.0
_LAYOUT_OUTCOME_X = 1050.0
_LAYOUT_SCATTER_X_RANGE = (350.0, 850.0)
_LAYOUT_SCATTER_Y_RANGE = (100.0, 700.0)


# -- request bodies -----------------------------------------------------------


class PositionUpdate(BaseModel):
    x: float
    y: float


class RoleUpdate(BaseModel):
    role: Role | None = None


class EdgeCreate(BaseModel):
    source: str
    target: str
    sign: Sign


class SignUpdate(BaseModel):
    sign: Sign


class ExportRequest(BaseModel):
    path: str
    format: Literal["json", "graphml"] = "json"


class SessionPathRequest(BaseModel):
    path: str


# -- app factory ----------------------------------------------------------------


def create_app(
    data_path: str | Path,
    *,
    treatments: Sequence[str] | None = None,
    outcomes: Sequence[str] | None = None,
    max_rows: int = 5000,
    test_size: float = 0.2,
    random_state: int = 0,
    plot_grid_size: int = 50,
    strong_r2: float = 0.01,
    strong_auc: float = 0.55,
    initial_session: str | Path | None = None,
) -> FastAPI:
    """Build the DAGshop FastAPI app for one workshop session.

    Reads `data_path` with pandas, runs `associate.scan_associations`
    once, and builds the initial `DAGModel` (one node per column, roles
    and pinned positions for any designated treatments/outcomes). All of
    this happens synchronously before the app is returned -- there is no
    "loading" state for the frontend to poll for, by design: `cli.py`
    (next slice) is expected to call this, then hand the result straight
    to uvicorn.

    `initial_session` (for `cli.py`'s `--session` flag):
    when given, the freshly built DAG from `treatments`/`outcomes` is
    discarded in favor of `DAGModel.load_session(initial_session)`. The
    association scan still runs against `data_path` regardless: a
    loaded session never touches the scan,
    since the scan is tied to whichever CSV the server was launched
    with, not to whichever session file gets resumed.

    Raises whatever `pd.read_csv`, `scan_associations`, or `DAGModel`
    raise on bad input (e.g. `associate.ColumnTypeError` for a
    non-numeric column), or `FileNotFoundError` if `initial_session` is
    given and doesn't exist -- all startup-time failures, not requests
    to handle gracefully in a route.
    """
    data_path = Path(data_path)
    data = pd.read_csv(data_path)

    treatments = list(treatments or [])
    outcomes = list(outcomes or [])
    both = set(treatments) & set(outcomes)
    if both:
        # Checked before the (potentially expensive, n*(t+o) fits) scan
        # runs, not after: no point paying for a scan we're about to
        # reject the input on. graph.py's Role is one tag per node,
        # so a column can't be pinned to both the
        # treatment and outcome layout columns at once.
        raise ValueError(f"columns cannot be both a treatment and an outcome: {sorted(both)}")

    scan = scan_associations(
        data,
        treatments=treatments,
        outcomes=outcomes,
        max_rows=max_rows,
        test_size=test_size,
        random_state=random_state,
        plot_grid_size=plot_grid_size,
        strong_r2=strong_r2,
        strong_auc=strong_auc,
    )
    dag = _build_initial_dag(
        list(data.columns), treatments=treatments, outcomes=outcomes, random_state=random_state
    )
    if initial_session is not None:
        dag = DAGModel.load_session(initial_session)

    # Cache for the fitted causal model, mirroring `dag`/`scan` above: set
    # by POST /api/causal/build, read by GET /api/causal/attribute/... .
    # None until the first successful build.
    causal_model_state: FittedCausalModel | None = None

    app = FastAPI(title="DAGshop")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(KeyError)
    def _handle_key_error(_request: Any, exc: KeyError) -> JSONResponse:
        # KeyError's str() wraps the message in an extra layer of repr
        # quotes (`str(KeyError("x"))` -> `"'x'"`, since graph.py always
        # raises with a single formatted string); strip that for a clean
        # API error message. Unconditional strip(), not an if-guarded
        # slice: every KeyError reaching this handler comes from
        # graph.py's `_require_node`/`_require_edge`, so the quotes are
        # always there, and `.strip` is a no-op on input that lacks them.
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(ValueError)
    def _handle_value_error(_request: Any, exc: ValueError) -> JSONResponse:
        # Covers plain ValueError (bad sign, bad role, self-loop) and
        # GraphValidationError, which subclasses it.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "data_path": str(data_path),
            "n_rows": int(data.shape[0]),
            "n_columns": int(data.shape[1]),
            "scoped": scan.scoped,
        }

    # -- graph -----------------------------------------------------------

    @app.get("/api/graph")
    def get_graph() -> dict[str, Any]:
        return {**dag.to_dict(), "treatments": dag.treatments, "outcomes": dag.outcomes}

    @app.put("/api/nodes/{name}/position")
    def set_position(name: str, body: PositionUpdate) -> dict[str, Any]:
        dag.set_position(name, body.x, body.y)
        return {"name": name, "x": body.x, "y": body.y}

    @app.put("/api/nodes/{name}/role")
    def set_role(name: str, body: RoleUpdate) -> dict[str, Any]:
        dag.set_role(name, body.role)
        return {"name": name, "role": body.role}

    @app.delete("/api/nodes/{name}", status_code=204)
    def remove_node(name: str) -> None:
        # Removes incident edges too (networkx.DiGraph.remove_node's
        # default behavior) -- a workshop may decide a column shouldn't
        # be modeled at all. Does not touch the association scan or
        # plot_cache: those stay tied to the data.csv loaded at startup,
        # same as session load below.
        dag.remove_node(name)

    @app.post("/api/edges", status_code=201)
    def add_edge(body: EdgeCreate) -> dict[str, Any]:
        # graph.py's add_edge only ever warns with CycleWarning (see
        # graph.py), so everything `catch_warnings` records here is one:
        # no need to filter caught by category, just take the first (and
        # only ever) one if the edge closed a cycle.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", CycleWarning)
            dag.add_edge(body.source, body.target, body.sign)
        cycle_warning = str(caught[0].message) if caught else None
        return {
            "source": body.source,
            "target": body.target,
            "sign": body.sign,
            "cycle_warning": cycle_warning,
        }

    @app.delete("/api/edges/{source}/{target}", status_code=204)
    def remove_edge(source: str, target: str) -> None:
        dag.remove_edge(source, target)

    @app.put("/api/edges/{source}/{target}/sign")
    def set_sign(source: str, target: str, body: SignUpdate) -> dict[str, Any]:
        dag.set_sign(source, target, body.sign)
        return {"source": source, "target": target, "sign": body.sign}

    @app.get("/api/validate")
    def validate() -> dict[str, Any]:
        cycles = dag.find_cycles()
        return {"valid": not cycles, "cycles": cycles}

    @app.post("/api/export")
    def export(body: ExportRequest) -> dict[str, Any]:
        try:
            if body.format == "json":
                dag.export_json(body.path)
            else:
                dag.export_graphml(body.path)
        except GraphValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=400, detail=f"could not write to {body.path!r}: {exc}"
            ) from exc
        return {"path": body.path, "format": body.format}

    # -- session save/load -------------------------------------------------

    @app.post("/api/session/save")
    def save_session(body: SessionPathRequest) -> dict[str, Any]:
        try:
            dag.save_session(body.path)
        except OSError as exc:
            raise HTTPException(
                status_code=400, detail=f"could not write to {body.path!r}: {exc}"
            ) from exc
        return {"path": body.path}

    @app.post("/api/session/load")
    def load_session(body: SessionPathRequest) -> dict[str, Any]:
        nonlocal dag
        try:
            dag = DAGModel.load_session(body.path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"session file not found: {body.path!r}"
            ) from exc
        return {**dag.to_dict(), "treatments": dag.treatments, "outcomes": dag.outcomes}

    # -- association scan (read-only: scan itself runs once, at startup) ---

    @app.get("/api/tables")
    def get_tables() -> dict[str, Any]:
        return {
            "scoped": scan.scoped,
            "treatment_table": [asdict(r) for r in scan.treatment_table],
            "outcome_table": [asdict(r) for r in scan.outcome_table],
            "covariate_table": [asdict(r) for r in scan.covariate_table],
            "full_table": [asdict(r) for r in scan.full_table],
            "skipped": [asdict(s) for s in scan.skipped],
        }

    @app.get("/api/plot/{predictor}/{target}")
    def get_plot(predictor: str, target: str) -> dict[str, Any]:
        plot = scan.plot_cache.get((predictor, target))
        if plot is None:
            raise HTTPException(
                status_code=404,
                detail=f"no cached plot for {predictor!r} -> {target!r}",
            )
        return asdict(plot)

    # -- causal attribution (SCOPE.md build order step 5) ------------------

    @app.post("/api/causal/build")
    def build_causal_model() -> dict[str, Any]:
        nonlocal causal_model_state
        fitted = fit_causal_model(dag, data, random_state=random_state)
        falsify = falsify_causal_graph(fitted, data)
        causal_model_state = fitted
        return {
            "fitted": True,
            "attributable_nodes": dag.nodes,
            "sign_disagreements": [asdict(d) for d in fitted.sign_disagreements],
            "falsify": asdict(falsify),
        }

    @app.get("/api/causal/attribute/{target_node}")
    def get_causal_attribution(target_node: str) -> dict[str, Any]:
        if causal_model_state is None:
            raise HTTPException(
                status_code=400,
                detail="causal model not built yet -- POST /api/causal/build first",
            )
        # random_state=random_state (fixes a
        # flaky ranking test): without it, gcm.intrinsic_causal_influence
        # draws from numpy's unseeded global RNG, so a workshop clicking
        # "Show drivers" twice for the same target could see the ranking
        # shuffle between clicks -- see causal_model.py's module docstring.
        contributions = attribute_target(causal_model_state, target_node, random_state=random_state)
        return {
            "target_node": target_node,
            "contributions": [asdict(r) for r in contributions],
        }

    return app


# -- initial layout -------------------------------------------------------------


def _build_initial_dag(
    columns: list[str],
    *,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    random_state: int,
) -> DAGModel:
    """One node per data column, roles/pinned positions applied.

    Treatment and outcome nodes get a fixed position in their own column
    (see the `_LAYOUT_*` constants above); every other node gets a
    reproducible scattered position, per SCOPE.md step 2. Assumes
    `treatments`/`outcomes` are already disjoint -- `create_app` checks
    that upfront, before the scan runs.
    """
    dag = DAGModel()
    rng = random.Random(random_state)

    treatment_ys = _spaced(len(treatments), *_LAYOUT_PINNED_Y_RANGE)
    outcome_ys = _spaced(len(outcomes), *_LAYOUT_PINNED_Y_RANGE)
    treatment_y = dict(zip(treatments, treatment_ys, strict=True))
    outcome_y = dict(zip(outcomes, outcome_ys, strict=True))

    for name in columns:
        if name in treatment_y:
            dag.add_node(name, x=_LAYOUT_TREATMENT_X, y=treatment_y[name], role="treatment")
        elif name in outcome_y:
            dag.add_node(name, x=_LAYOUT_OUTCOME_X, y=outcome_y[name], role="outcome")
        else:
            x = rng.uniform(*_LAYOUT_SCATTER_X_RANGE)
            y = rng.uniform(*_LAYOUT_SCATTER_Y_RANGE)
            dag.add_node(name, x=x, y=y, role=None)
    return dag


def _spaced(n: int, low: float, high: float) -> list[float]:
    """`n` evenly spaced values across `[low, high]` (a single value is centered)."""
    if n <= 0:
        return []
    if n == 1:
        return [(low + high) / 2]
    step = (high - low) / (n - 1)
    return [low + i * step for i in range(n)]


if __name__ == "__main__":
    # Manual dev entrypoint only -- not the real CLI. `cli.py` (SCOPE.md
    # build order step 4) will own argument parsing, `--treatment`/
    # `--outcome` flags, and packaging; this is just
    # `python -m dagshop.server data.csv` for local testing during this
    # slice, no flags of its own.
    import sys

    import uvicorn

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m dagshop.server DATA.csv")
    dev_app = create_app(sys.argv[1])
    uvicorn.run(dev_app, host="127.0.0.1", port=8000)
