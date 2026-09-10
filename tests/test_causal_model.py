"""Unit tests for dagshop.causal_model.

Pure-logic module (wraps dowhy.gcm, no UI dependency): see SCOPE.md build
order step 4. Most tests use small hand-built DAGs/data rather than the
full `demo_data.make_csat_demo_data` scenario, to keep fitting/attribution
fast -- one slower end-to-end smoke test at the bottom exercises the real
CSAT scenario instead.

`n_jobs=1` is passed to `attribute_target`/`falsify_causal_graph`
throughout: this session's sandboxed bridge shell hits
`BrokenProcessPool`/`OSError: Too many open files` under `dowhy`'s
default joblib parallelism (see causal_model.py's module docstring) --
not necessarily an issue on a real machine, but forcing sequential
execution keeps the test suite runnable here regardless.

`attribute_target`'s ranking-dependent assertions also pass
`random_state=0`, added after a CI run on Python 3.13 failed
`test_csat_scenario_end_to_end_ranks_friction_severity_highest` --
`gcm.intrinsic_causal_influence` draws from `numpy`'s unseeded global
RNG (see causal_model.py's `random_state` note), so without a seed the
ranking is a fresh Monte Carlo draw every run and can occasionally
disagree with itself. Checked `friction_severity` still wins by a wide,
comfortable margin (roughly 3-4x the runner-up) across several other
seeds too before picking `0` -- this was a reproducibility bug, not a
knife's-edge assertion that needed loosening.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from dowhy import gcm

from dagshop.associate import ColumnTypeError
from dagshop.causal_model import (
    AttributionResult,
    FalsifyResult,
    SignDisagreement,
    _n_jobs_override,
    _sign_disagreements,
    attribute_target,
    falsify_causal_graph,
    fit_causal_model,
)
from dagshop.graph import DAGModel, GraphValidationError


def _root_mid_target_dag_and_data(
    n_rows: int = 300, random_state: int = 0
) -> tuple[DAGModel, pd.DataFrame]:
    """root -> mid -> target, both edges "+", both relationships genuinely positive.

    `mid`'s own noise is deliberately large relative to `root`'s
    contribution (not a near-deterministic `mid = f(root)`): a highly
    collinear root/mid pair makes `mid`'s own intrinsic (Shapley)
    contribution to `target` small and noisy at low Monte Carlo sample
    counts, since most of the variance credit goes to `root` as the
    original variance source -- this shape keeps both nodes' attribution
    reliably positive even with the reduced sample counts the tests use
    to stay fast.
    """
    rng = np.random.default_rng(random_state)
    root = rng.normal(0, 1, size=n_rows)
    mid = 1.0 * root + rng.normal(0, 1.5, size=n_rows)
    target = 2.0 * mid + rng.normal(0, 0.5, size=n_rows)
    data = pd.DataFrame({"root": root, "mid": mid, "target": target})

    dag = DAGModel()
    for name in ("root", "mid", "target"):
        dag.add_node(name)
    dag.add_edge("root", "mid", "+")
    dag.add_edge("mid", "target", "+")
    return dag, data


def _fit(dag: DAGModel, data: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_causal_model(dag, data, **kwargs)


# -- mechanism assignment -------------------------------------------------------


def test_root_node_gets_empirical_distribution() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    assert isinstance(fitted.scm.causal_mechanism("root"), gcm.EmpiricalDistribution)


def test_non_root_node_gets_additive_noise_model_with_regressor() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    mechanism = fitted.scm.causal_mechanism("target")
    assert isinstance(mechanism, gcm.AdditiveNoiseModel)
    assert isinstance(mechanism.prediction_model, gcm.ml.SklearnRegressionModel)


def test_monotonic_cst_follows_alphabetical_parent_order_not_insertion_order() -> None:
    """SCOPE.md-mandated test: parents added in an order that differs from
    alphabetical order must still produce a `monotonic_cst` array in
    alphabetical order, since that is the order `dowhy.gcm` itself
    assembles the parent feature matrix in (see causal_model.py's module
    docstring). "zulu" is added before "alpha" -- insertion order and
    alphabetical order disagree, so a wrong (insertion-order)
    implementation is visibly wrong here, not accidentally right.
    """
    rng = np.random.default_rng(0)
    n = 100
    alpha = rng.normal(0, 1, size=n)
    zulu = rng.normal(0, 1, size=n)
    # child = -alpha + zulu + noise: alpha is asserted "-", zulu is asserted "+".
    child = -alpha + zulu + rng.normal(0, 0.5, size=n)
    data = pd.DataFrame({"alpha": alpha, "zulu": zulu, "child": child})

    dag = DAGModel()
    for name in ("alpha", "zulu", "child"):
        dag.add_node(name)
    dag.add_edge("zulu", "child", "+")
    dag.add_edge("alpha", "child", "-")

    fitted = _fit(dag, data)
    mechanism = fitted.scm.causal_mechanism("child")
    # Alphabetical parent order is ["alpha", "zulu"]: expected [-1, 1].
    # An insertion-order bug would instead produce [1, -1] (zulu, alpha).
    assert list(mechanism.prediction_model.sklearn_model.monotonic_cst) == [-1, 1]


# -- input validation -------------------------------------------------------------


def test_fit_causal_model_rejects_cyclic_dag() -> None:
    dag = DAGModel()
    for name in ("a", "b"):
        dag.add_node(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag.add_edge("a", "b", "+")
        dag.add_edge("b", "a", "+")
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(GraphValidationError):
        fit_causal_model(dag, data)


def test_fit_causal_model_missing_column_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    with pytest.raises(KeyError, match="target"):
        fit_causal_model(dag, data.drop(columns=["target"]))


def test_fit_causal_model_non_numeric_column_raises_column_type_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    data = data.copy()
    data["root"] = data["root"].astype(str)
    with pytest.raises(ColumnTypeError, match="root"):
        fit_causal_model(dag, data)


# -- sign disagreements -----------------------------------------------------------


def test_sign_disagreement_flagged_when_correlation_opposes_asserted_sign() -> None:
    rng = np.random.default_rng(0)
    n = 200
    root = rng.normal(0, 1, size=n)
    # target is strongly *negatively* related to root, but the edge below
    # asserts "+".
    target = -3.0 * root + rng.normal(0, 0.2, size=n)
    data = pd.DataFrame({"root": root, "target": target})

    dag = DAGModel()
    dag.add_node("root")
    dag.add_node("target")
    dag.add_edge("root", "target", "+")

    fitted = _fit(dag, data)
    assert len(fitted.sign_disagreements) == 1
    disagreement = fitted.sign_disagreements[0]
    assert isinstance(disagreement, SignDisagreement)
    assert disagreement.parent == "root"
    assert disagreement.child == "target"
    assert disagreement.asserted_sign == "+"
    assert disagreement.correlation < 0


def test_no_sign_disagreement_when_correlation_agrees() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    assert fitted.sign_disagreements == []


def test_sign_disagreement_respects_min_correlation_floor() -> None:
    rng = np.random.default_rng(0)
    n = 200
    # root and target are essentially unrelated: whatever small sample
    # correlation shows up should not be flagged at the default floor.
    root = rng.normal(0, 1, size=n)
    target = rng.normal(0, 1, size=n)
    data = pd.DataFrame({"root": root, "target": target})

    dag = DAGModel()
    dag.add_node("root")
    dag.add_node("target")
    dag.add_edge("root", "target", "+")

    fitted = _fit(dag, data, min_correlation=0.9)
    assert fitted.sign_disagreements == []


# -- attribution --------------------------------------------------------------


def test_attribute_target_unknown_node_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(KeyError, match="nope"):
        attribute_target(fitted, "nope", n_jobs=1)


def test_attribute_target_ranks_every_ancestor_descending() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    results = attribute_target(
        fitted,
        "target",
        n_jobs=1,
        num_training_samples=200,
        num_samples_randomization=20,
        num_samples_baseline=20,
        random_state=0,
    )
    assert {r.node for r in results} == {"root", "mid", "target"}
    assert all(isinstance(r, AttributionResult) for r in results)
    contributions = [r.contribution for r in results]
    assert contributions == sorted(contributions, reverse=True)
    # root and mid are both genuine (strong, positive) drivers of target;
    # a disconnected node would not turn up at all, but there isn't one
    # in this tiny DAG -- the meaningful assertion is that both ancestors
    # contribute noticeably more than a token amount.
    by_node = {r.node: r.contribution for r in results}
    assert by_node["root"] > 0
    assert by_node["mid"] > 0


def test_attribute_target_random_state_is_reproducible() -> None:
    """Regression test for the flaky-CI fix above: two calls with the
    same `random_state` (and `n_jobs=1`, so there's no worker race over
    numpy's seeded global RNG -- see causal_model.py's `random_state`
    note) must return bit-identical contributions, not just the same
    ranking order.
    """
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    kwargs = dict(
        n_jobs=1,
        num_training_samples=200,
        num_samples_randomization=20,
        num_samples_baseline=20,
        random_state=7,
    )
    first = attribute_target(fitted, "target", **kwargs)
    second = attribute_target(fitted, "target", **kwargs)
    assert [r.contribution for r in first] == [r.contribution for r in second]


# -- private-helper branch coverage ----------------------------------------------


def test_n_jobs_override_is_noop_when_none() -> None:
    from dowhy.gcm import config as gcm_config

    before = gcm_config.default_n_jobs
    with _n_jobs_override(None):
        assert gcm_config.default_n_jobs == before
    assert gcm_config.default_n_jobs == before


def test_attribute_target_uses_dowhy_defaults_when_sample_counts_not_given() -> None:
    """Covers the branches where num_training_samples/num_samples_randomization/
    num_samples_baseline are left `None` (dowhy's own defaults apply) --
    every other attribution test overrides all three to stay fast."""
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    results = attribute_target(fitted, "target", n_jobs=1)
    assert {r.node for r in results} == {"root", "mid", "target"}


def test_sign_disagreements_skips_pair_with_too_few_rows() -> None:
    dag = DAGModel()
    dag.add_node("root")
    dag.add_node("target")
    dag.add_edge("root", "target", "+")
    data = pd.DataFrame({"root": [1.0, np.nan], "target": [np.nan, 2.0]})
    assert _sign_disagreements(dag, data, min_correlation=0.1) == []


def test_sign_disagreements_skips_constant_column_nan_correlation() -> None:
    """A zero-variance column makes Spearman correlation undefined (nan),
    not zero or a real sign -- must be skipped, not misread as agreement."""
    from scipy.stats import ConstantInputWarning

    dag = DAGModel()
    dag.add_node("flat")
    dag.add_node("target")
    dag.add_edge("flat", "target", "+")
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"flat": [1.0] * 50, "target": rng.normal(size=50)})
    with pytest.warns(ConstantInputWarning):
        assert _sign_disagreements(dag, data, min_correlation=0.1) == []


# -- falsify --------------------------------------------------------------------


def test_falsify_causal_graph_returns_result_with_report() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = falsify_causal_graph(fitted, data, n_jobs=1)
    assert isinstance(result, FalsifyResult)
    assert result.significance_level == 0.05
    assert isinstance(result.report, str)
    assert len(result.report) > 0
    # falsified/falsifiable are None only when dowhy can't evaluate at
    # all; with a real, non-degenerate 3-node DAG they should resolve to
    # actual booleans.
    assert result.falsified in (True, False)
    assert result.falsifiable in (True, False)


# -- end-to-end smoke test against the real CSAT scenario -----------------------


def _csat_dag() -> DAGModel:
    from dagshop.demo_data import make_csat_demo_data  # noqa: F401 (docs the pairing)

    dag = DAGModel()
    for name in (
        "age",
        "friction_severity",
        "time_to_respond",
        "num_transfers",
        "num_escalations",
        "num_agents_spoken_to",
        "time_to_resolve",
        "resolved",
        "repeat_contact",
        "csat",
    ):
        dag.add_node(name)
    edges = [
        ("age", "csat", "+"),
        ("friction_severity", "num_transfers", "+"),
        ("friction_severity", "num_escalations", "+"),
        ("friction_severity", "time_to_resolve", "+"),
        ("friction_severity", "resolved", "-"),
        ("friction_severity", "repeat_contact", "+"),
        ("friction_severity", "csat", "-"),
        ("time_to_respond", "time_to_resolve", "+"),
        ("num_transfers", "num_agents_spoken_to", "+"),
        ("num_transfers", "time_to_resolve", "+"),
        ("num_escalations", "num_agents_spoken_to", "+"),
        ("num_escalations", "time_to_resolve", "+"),
        ("num_escalations", "resolved", "-"),
        ("num_agents_spoken_to", "time_to_resolve", "+"),
        ("time_to_resolve", "resolved", "-"),
        ("time_to_resolve", "csat", "-"),
        ("resolved", "repeat_contact", "-"),
        ("resolved", "csat", "+"),
        ("repeat_contact", "csat", "-"),
    ]
    for source, target, sign in edges:
        dag.add_edge(source, target, sign)
    return dag


def test_csat_scenario_end_to_end_ranks_friction_severity_highest() -> None:
    """`friction_severity` is documented (demo_data.py's own docstring,
    and session 12's NOTES.md) as the strongest driver of `csat` in this
    scenario, since nearly every other driver is downstream of it. This
    is the closest thing to a regression test for the whole module
    working together against real, documented ground truth, not just its
    pieces in isolation.
    """
    from dagshop.demo_data import make_csat_demo_data

    data = make_csat_demo_data(n_rows=1000, random_state=0)
    dag = _csat_dag()
    fitted = _fit(dag, data)
    assert fitted.sign_disagreements == []

    results = attribute_target(
        fitted,
        "csat",
        n_jobs=1,
        num_training_samples=500,
        num_samples_randomization=30,
        num_samples_baseline=30,
        random_state=0,
    )
    by_node = {r.node: r.contribution for r in results}
    assert max(by_node, key=by_node.get) == "friction_severity"
