"""Unit tests for dagshop.graph.DAGModel.

Pure-logic module (no UI, no fitting dependency): see SCOPE.md build
order step 1 for why this is unit tested in isolation first.
"""

from __future__ import annotations

import warnings

import networkx as nx
import pytest

from dagshop.graph import SCHEMA_VERSION, CycleWarning, DAGModel, GraphValidationError


def _triangle() -> DAGModel:
    """X -> Y -> Z, no closing edge yet (acyclic)."""
    model = DAGModel()
    for name in ("X", "Y", "Z"):
        model.add_node(name)
    model.add_edge("X", "Y", "+")
    model.add_edge("Y", "Z", "-")
    return model


# -- nodes --------------------------------------------------------------------


def test_add_node_defaults_to_no_position_or_role() -> None:
    model = DAGModel()
    model.add_node("age")
    assert model.has_node("age")
    assert model.position("age") == (None, None)
    assert model.role("age") is None


def test_add_node_with_position_and_role() -> None:
    model = DAGModel()
    model.add_node("age", x=1.5, y=-2.0, role="treatment")
    assert model.position("age") == (1.5, -2.0)
    assert model.role("age") == "treatment"


def test_add_node_invalid_role_raises() -> None:
    model = DAGModel()
    with pytest.raises(ValueError, match="role"):
        model.add_node("age", role="confounder")


def test_set_role_updates_role() -> None:
    model = DAGModel()
    model.add_node("age")
    model.set_role("age", "outcome")
    assert model.role("age") == "outcome"
    model.set_role("age", None)
    assert model.role("age") is None


def test_set_role_invalid_raises() -> None:
    model = DAGModel()
    model.add_node("age")
    with pytest.raises(ValueError, match="role"):
        model.set_role("age", "not-a-role")


def test_set_role_missing_node_raises_keyerror() -> None:
    model = DAGModel()
    with pytest.raises(KeyError):
        model.set_role("ghost", "treatment")


def test_position_missing_node_raises_keyerror() -> None:
    model = DAGModel()
    with pytest.raises(KeyError):
        model.position("ghost")


def test_set_position_missing_node_raises_keyerror() -> None:
    model = DAGModel()
    with pytest.raises(KeyError):
        model.set_position("ghost", 0.0, 0.0)


def test_set_position_updates_existing_node() -> None:
    model = DAGModel()
    model.add_node("age")
    model.set_position("age", 3.0, 4.0)
    assert model.position("age") == (3.0, 4.0)


def test_treatments_and_outcomes_are_sorted() -> None:
    model = DAGModel()
    for name in ("spend_b", "spend_a", "revenue", "region"):
        model.add_node(name)
    model.set_role("spend_b", "treatment")
    model.set_role("spend_a", "treatment")
    model.set_role("revenue", "outcome")
    assert model.treatments == ["spend_a", "spend_b"]
    assert model.outcomes == ["revenue"]


def test_remove_node() -> None:
    model = DAGModel()
    model.add_node("age")
    model.remove_node("age")
    assert not model.has_node("age")


def test_remove_node_missing_raises_keyerror() -> None:
    model = DAGModel()
    with pytest.raises(KeyError):
        model.remove_node("ghost")


def test_nodes_property() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    assert set(model.nodes) == {"a", "b"}


# -- edges --------------------------------------------------------------------


def test_add_edge_stores_sign() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    model.add_edge("a", "b", "+")
    assert model.has_edge("a", "b")
    assert model.sign("a", "b") == "+"
    assert model.edges == [("a", "b")]


def test_add_edge_invalid_sign_raises() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    with pytest.raises(ValueError, match="sign"):
        model.add_edge("a", "b", "0")


def test_add_edge_self_loop_raises() -> None:
    model = DAGModel()
    model.add_node("a")
    with pytest.raises(ValueError, match="self-loop"):
        model.add_edge("a", "a", "+")


def test_add_edge_missing_source_raises_keyerror() -> None:
    model = DAGModel()
    model.add_node("b")
    with pytest.raises(KeyError):
        model.add_edge("a", "b", "+")


def test_add_edge_missing_target_raises_keyerror() -> None:
    model = DAGModel()
    model.add_node("a")
    with pytest.raises(KeyError):
        model.add_edge("a", "b", "+")


def test_re_adding_edge_updates_sign_without_duplicate() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    model.add_edge("a", "b", "+")
    model.add_edge("a", "b", "-")
    assert model.sign("a", "b") == "-"
    assert model.edges == [("a", "b")]


def test_set_sign_and_getter() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    model.add_edge("a", "b", "+")
    model.set_sign("a", "b", "-")
    assert model.sign("a", "b") == "-"


def test_set_sign_invalid_raises() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    model.add_edge("a", "b", "+")
    with pytest.raises(ValueError, match="sign"):
        model.set_sign("a", "b", "0")


def test_sign_missing_edge_raises_keyerror() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    with pytest.raises(KeyError):
        model.sign("a", "b")


def test_remove_edge() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    model.add_edge("a", "b", "+")
    model.remove_edge("a", "b")
    assert not model.has_edge("a", "b")


def test_remove_edge_missing_raises_keyerror() -> None:
    model = DAGModel()
    model.add_node("a")
    model.add_node("b")
    with pytest.raises(KeyError):
        model.remove_edge("a", "b")


# -- cycle detection: warn, don't block ----------------------------------------


def test_add_edge_creating_cycle_warns_but_still_adds_edge() -> None:
    model = _triangle()
    with pytest.warns(CycleWarning, match="X"):
        model.add_edge("Z", "X", "+")
    assert model.has_edge("Z", "X")
    assert model.has_cycle()


def test_add_edge_not_creating_cycle_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _triangle()
    assert not any(issubclass(w.category, CycleWarning) for w in caught)


def test_has_cycle_false_when_acyclic() -> None:
    model = _triangle()
    assert not model.has_cycle()
    assert model.find_cycles() == []


def test_has_cycle_true_and_find_cycles_reports_it() -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    assert model.has_cycle()
    cycles = model.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"X", "Y", "Z"}


def test_removing_edge_breaks_cycle() -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    model.remove_edge("Z", "X")
    assert not model.has_cycle()


# -- validate: the export-time gate --------------------------------------------


def test_validate_passes_when_acyclic() -> None:
    model = _triangle()
    model.validate()  # should not raise


def test_validate_raises_on_cycle() -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    with pytest.raises(GraphValidationError, match="cycle"):
        model.validate()


# -- serialization: to_dict / from_dict ----------------------------------------


def test_to_dict_round_trips_nodes_and_edges() -> None:
    model = DAGModel()
    model.add_node("spend", x=10.0, y=20.0, role="treatment")
    model.add_node("revenue", x=30.0, y=40.0, role="outcome")
    model.add_node("region")
    model.add_edge("spend", "revenue", "+")

    data = model.to_dict()
    assert data["schema_version"] == SCHEMA_VERSION
    restored = DAGModel.from_dict(data)

    assert set(restored.nodes) == {"spend", "revenue", "region"}
    assert restored.position("spend") == (10.0, 20.0)
    assert restored.role("spend") == "treatment"
    assert restored.role("region") is None
    assert restored.position("region") == (None, None)
    assert restored.edges == [("spend", "revenue")]
    assert restored.sign("spend", "revenue") == "+"


def test_from_dict_on_cyclic_data_warns_like_add_edge() -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    data = model.to_dict()

    with pytest.warns(CycleWarning):
        restored = DAGModel.from_dict(data)
    assert restored.has_cycle()


# -- save/resume (session file) vs export (validated) --------------------------


def test_save_and_load_session_round_trips(tmp_path) -> None:
    model = _triangle()
    path = tmp_path / "session.json"
    model.save_session(path)
    loaded = DAGModel.load_session(path)
    assert set(loaded.nodes) == set(model.nodes)
    assert loaded.edges == model.edges


def test_save_session_does_not_require_acyclic(tmp_path) -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    path = tmp_path / "session.json"
    model.save_session(path)  # should not raise despite the cycle
    with pytest.warns(CycleWarning):
        loaded = DAGModel.load_session(path)
    assert loaded.has_cycle()


def test_export_json_raises_on_cycle_and_writes_nothing(tmp_path) -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    path = tmp_path / "export.json"
    with pytest.raises(GraphValidationError):
        model.export_json(path)
    assert not path.exists()


def test_export_json_succeeds_when_acyclic(tmp_path) -> None:
    model = _triangle()
    path = tmp_path / "export.json"
    model.export_json(path)
    loaded = DAGModel.load_session(path)
    assert loaded.edges == model.edges
    assert loaded.sign("Y", "Z") == "-"


def test_export_graphml_raises_on_cycle_and_writes_nothing(tmp_path) -> None:
    model = _triangle()
    with pytest.warns(CycleWarning):
        model.add_edge("Z", "X", "+")
    path = tmp_path / "export.graphml"
    with pytest.raises(GraphValidationError):
        model.export_graphml(path)
    assert not path.exists()


def test_export_graphml_writes_readable_file(tmp_path) -> None:
    model = DAGModel()
    model.add_node("spend", x=10.0, y=20.0, role="treatment")
    model.add_node("revenue", x=30.0, y=40.0, role="outcome")
    model.add_edge("spend", "revenue", "+")
    path = tmp_path / "export.graphml"

    model.export_graphml(path)

    read_back = nx.read_graphml(path)
    assert set(read_back.nodes) == {"spend", "revenue"}
    assert read_back.has_edge("spend", "revenue")
    assert read_back.edges["spend", "revenue"]["sign"] == "+"
    assert read_back.nodes["spend"]["role"] == "treatment"
    assert read_back.nodes["spend"]["x"] == pytest.approx(10.0)
    assert read_back.nodes["spend"]["y"] == pytest.approx(20.0)


def test_export_graphml_omits_unset_position_and_role(tmp_path) -> None:
    model = DAGModel()
    model.add_node("region")  # no x/y/role set
    model.add_node("revenue")
    model.add_edge("region", "revenue", "+")
    path = tmp_path / "bare.graphml"

    model.export_graphml(path)

    read_back = nx.read_graphml(path)
    assert "x" not in read_back.nodes["region"]
    assert "y" not in read_back.nodes["region"]
    assert "role" not in read_back.nodes["region"]
