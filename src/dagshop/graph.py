"""DAG data model: nodes, edges, user-asserted signs, cycle handling.

networkx.DiGraph-backed. Cycle detection warns rather than blocks, to
support free brainstorming; validity is enforced before export. JSON and
graphml serialization, save/resume.

Pure logic, no UI or fitting dependency: first module in the build order,
easiest to unit test in isolation. See SCOPE.md build order step 1.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import networkx as nx

Sign = Literal["+", "-"]
Role = Literal["treatment", "outcome"]

_VALID_SIGNS: tuple[Sign, ...] = ("+", "-")
_VALID_ROLES: tuple[Role, ...] = ("treatment", "outcome")

# Bumped only if the on-disk JSON shape changes in a way that breaks
# `from_dict`/`load_session` for files written by an older version.
SCHEMA_VERSION = 1


class CycleWarning(UserWarning):
    """An edge was added that closes a cycle.

    Per SCOPE.md step 2, cycles are allowed mid-session to support free
    brainstorming. This is a warning, not an exception: the edge is still
    added. Call `DAGModel.validate()` before export to enforce
    acyclicity.
    """


class GraphValidationError(ValueError):
    """Raised by `DAGModel.validate()` when the graph is not export-ready."""


@dataclass
class DAGModel:
    """DAG data model for a DAGshop workshop session.

    Backed by a `networkx.DiGraph`. Each node carries an optional (x, y)
    canvas position and an optional treatment/outcome role tag. Each edge
    carries a user-asserted "+"/"-" sign (SCOPE.md: no data-derived
    default, the sign is domain knowledge set by the PM/DS).

    Nodes must be added with `add_node` before an edge referencing them
    can be added; `add_edge` does not auto-create endpoints. This matches
    the workshop UI flow (drag a node onto the canvas, then click-drag
    between two nodes that already exist) and turns a typo'd node name
    into an immediate `KeyError` instead of a silent phantom node.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    # -- nodes ------------------------------------------------------------

    def add_node(
        self,
        name: str,
        *,
        x: float | None = None,
        y: float | None = None,
        role: Role | None = None,
    ) -> None:
        """Add a node, or update an existing one's position/role.

        `x`/`y` are `None` until the node is placed on the canvas
        (SCOPE.md: unlinked variables "start as unlinked, scattered
        nodes"). `role` is `None` unless the node has been designated a
        treatment or outcome.
        """
        self._check_role(role)
        self.graph.add_node(name, x=x, y=y, role=role)

    def remove_node(self, name: str) -> None:
        self._require_node(name)
        self.graph.remove_node(name)

    def has_node(self, name: str) -> bool:
        return self.graph.has_node(name)

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    def set_position(self, name: str, x: float, y: float) -> None:
        self._require_node(name)
        self.graph.nodes[name]["x"] = x
        self.graph.nodes[name]["y"] = y

    def position(self, name: str) -> tuple[float | None, float | None]:
        self._require_node(name)
        data = self.graph.nodes[name]
        return data.get("x"), data.get("y")

    def set_role(self, name: str, role: Role | None) -> None:
        """Designate (or clear, with `role=None`) a node's treatment/outcome tag.

        SCOPE.md allows one or more treatments and one or more outcomes
        per session; nothing here caps how many nodes carry each role.
        A single node holding both roles at once is not modeled: role is
        one tag per node (`None`, `"treatment"`, or `"outcome"`), which
        matches "treatment/outcome role tags" being described in
        SCOPE.md's Output section as a single per-node tag.
        """
        self._require_node(name)
        self._check_role(role)
        self.graph.nodes[name]["role"] = role

    def role(self, name: str) -> Role | None:
        self._require_node(name)
        return self.graph.nodes[name].get("role")

    @property
    def treatments(self) -> list[str]:
        """Nodes designated as treatment variables, sorted by name."""
        return sorted(n for n, d in self.graph.nodes(data=True) if d.get("role") == "treatment")

    @property
    def outcomes(self) -> list[str]:
        """Nodes designated as outcome variables, sorted by name."""
        return sorted(n for n, d in self.graph.nodes(data=True) if d.get("role") == "outcome")

    # -- edges --------------------------------------------------------------

    def add_edge(self, source: str, target: str, sign: Sign) -> None:
        """Add a directed, signed edge `source -> target`.

        `sign` is required (SCOPE.md: every edge needs a user-asserted
        +/- sign, with no default). If this edge closes a cycle, a
        `CycleWarning` is issued but the edge is still added: cycles are
        allowed mid-session, and rejected only by `validate()` at export
        time.
        """
        self._check_sign(sign)
        if source == target:
            raise ValueError(f"self-loops are not allowed: {source!r} -> {target!r}")
        self._require_node(source)
        self._require_node(target)
        self.graph.add_edge(source, target, sign=sign)
        cycle = self._cycle_created_by(source, target)
        if cycle is not None:
            path = " -> ".join([*cycle, cycle[0]])
            warnings.warn(
                f"edge {source!r} -> {target!r} closes a cycle: {path}. "
                "Allowed mid-session; run validate() before export.",
                CycleWarning,
                stacklevel=2,
            )

    def remove_edge(self, source: str, target: str) -> None:
        self._require_edge(source, target)
        self.graph.remove_edge(source, target)

    def has_edge(self, source: str, target: str) -> bool:
        return self.graph.has_edge(source, target)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)

    def set_sign(self, source: str, target: str, sign: Sign) -> None:
        self._require_edge(source, target)
        self._check_sign(sign)
        self.graph.edges[source, target]["sign"] = sign

    def sign(self, source: str, target: str) -> Sign:
        self._require_edge(source, target)
        return self.graph.edges[source, target]["sign"]

    # -- cycle detection (warn, don't block; see class docstring) -----------

    def has_cycle(self) -> bool:
        return not nx.is_directed_acyclic_graph(self.graph)

    def find_cycles(self) -> list[list[str]]:
        """All simple cycles currently in the graph, as lists of node names."""
        return list(nx.simple_cycles(self.graph))

    def validate(self) -> None:
        """Raise `GraphValidationError` if the graph is not export-ready.

        Cycles are the only export-blocking condition today, per
        SCOPE.md ("validity is enforced before export" refers
        specifically to acyclicity). Call this before `export_json`/
        `export_graphml` -- both call it internally too, so a caller
        that wants to surface a validation error before attempting an
        export (e.g. to show it in the UI without touching the
        filesystem) can call it directly first.
        """
        cycles = self.find_cycles()
        if cycles:
            formatted = "; ".join(" -> ".join([*c, c[0]]) for c in cycles)
            raise GraphValidationError(f"graph contains {len(cycles)} cycle(s): {formatted}")

    def _cycle_created_by(self, source: str, target: str) -> list[str] | None:
        """Cycle closed by the just-added `source -> target` edge, if any.

        DFS from `target` following out-edges will retrace back to
        `source` (if a path existed) and then re-traverse the new
        `source -> target` edge, hitting `target` again as an ancestor on
        the DFS stack -- exactly the cycle this edge closed. This is a
        local check for the warning message; `find_cycles()`/`has_cycle()`
        scan the whole graph and are the source of truth for validation.
        """
        try:
            cycle_edges = nx.find_cycle(self.graph, source=target)
        except nx.NetworkXNoCycle:
            return None
        return [u for u, _ in cycle_edges]

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The DAG file format described in SCOPE.md's Output section.

        Nodes carry position and role inline rather than as separate
        top-level lists, so there is one place, not two, that can go out
        of sync when a node is added or renamed.
        """
        nodes = [
            {"name": n, "x": d.get("x"), "y": d.get("y"), "role": d.get("role")}
            for n, d in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "sign": d["sign"]} for u, v, d in self.graph.edges(data=True)
        ]
        return {"schema_version": SCHEMA_VERSION, "nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DAGModel:
        model = cls()
        for node in data.get("nodes", []):
            model.add_node(node["name"], x=node.get("x"), y=node.get("y"), role=node.get("role"))
        for edge in data.get("edges", []):
            model.add_edge(edge["source"], edge["target"], edge["sign"])
        return model

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> DAGModel:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def save_session(self, path: str | Path) -> None:
        """Save the current state, cycles and all, so a session can resume.

        Same JSON format as `export_json`, deliberately without the
        `validate()` gate: SCOPE.md allows saving mid-brainstorm, before
        the DAG is acyclic.
        """
        self.to_json(path)

    @classmethod
    def load_session(cls, path: str | Path) -> DAGModel:
        """Resume a session file written by `save_session` or `export_json`."""
        return cls.from_json(path)

    # -- export (validation-gated) ---------------------------------------------

    def export_json(self, path: str | Path) -> None:
        """Write the DAG file for downstream use, after checking it's acyclic.

        Raises `GraphValidationError` (and writes nothing) if the graph
        still has a cycle.
        """
        self.validate()
        self.to_json(path)

    def export_graphml(self, path: str | Path) -> None:
        """Write a GraphML conversion of the DAG file, after validating it.

        Raises `GraphValidationError` (and writes nothing) if the graph
        still has a cycle. Only attributes GraphML/networkx can round-trip
        cleanly (str/float) are written, and position/role are omitted per
        node when unset rather than written as `None`.
        """
        self.validate()
        export_graph: nx.DiGraph = nx.DiGraph()
        for n, d in self.graph.nodes(data=True):
            attrs: dict[str, Any] = {}
            if d.get("x") is not None:
                attrs["x"] = float(d["x"])
            if d.get("y") is not None:
                attrs["y"] = float(d["y"])
            if d.get("role") is not None:
                attrs["role"] = d["role"]
            export_graph.add_node(n, **attrs)
        for u, v, d in self.graph.edges(data=True):
            export_graph.add_edge(u, v, sign=d["sign"])
        nx.write_graphml(export_graph, path)

    # -- internal guards --------------------------------------------------------

    def _require_node(self, name: str) -> None:
        if not self.graph.has_node(name):
            raise KeyError(f"no node {name!r}")

    def _require_edge(self, source: str, target: str) -> None:
        if not self.graph.has_edge(source, target):
            raise KeyError(f"no edge {source!r} -> {target!r}")

    @staticmethod
    def _check_sign(sign: Sign) -> None:
        if sign not in _VALID_SIGNS:
            raise ValueError(f"sign must be one of {_VALID_SIGNS}, got {sign!r}")

    @staticmethod
    def _check_role(role: Role | None) -> None:
        if role is not None and role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES} or None, got {role!r}")
