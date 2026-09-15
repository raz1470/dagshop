"""Unit tests for dagshop.causal_model.

Pure-logic module (wraps dowhy.gcm, no UI dependency): see SCOPE.md build
order step 4. Most tests use small hand-built DAGs/data rather than the
full `demo_data.make_csat_demo_data` scenario, to keep fitting/attribution
fast -- one slower end-to-end smoke test at the bottom exercises the real
CSAT scenario instead.

`n_jobs=1` is passed to `attribute_target`/`evaluate_causal_model`
throughout: the sandboxed bridge shell used for development hits
`BrokenProcessPool`/`OSError: Too many open files` under `dowhy`'s
default joblib parallelism (see causal_model.py's module docstring) --
not necessarily an issue on a real machine, but forcing sequential
execution keeps the test suite runnable here regardless.

`attribute_target`'s ranking-dependent assertions also pass
`random_state=0`, added after a CI run on Python 3.13 failed the CSAT
end-to-end test below (`test_csat_scenario_end_to_end_ranks_resolved_highest`,
renamed since when the CSAT demo's own DAG was restructured; the
underlying reproducibility fix and its reasoning are unchanged):
`gcm.intrinsic_causal_influence` draws from `numpy`'s unseeded global
RNG (see causal_model.py's `random_state` note), so without a seed the
ranking is a fresh Monte Carlo draw every run and can occasionally
disagree with itself. Checked the top-ranked ancestor stays top across
several other seeds too before picking `0` -- this was a
reproducibility bug, not a knife's-edge assertion that needed
loosening.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from dowhy import gcm
from scipy.stats import norm

from dagshop.associate import ColumnTypeError
from dagshop.causal_model import (
    ActualVsPredictedPlot,
    AttributionResult,
    InterventionResult,
    MechanismPerformance,
    ModelEvaluation,
    ObservedVsSampledPlot,
    PeriodAttributionResult,
    SignDisagreement,
    _is_binary_coded,
    _n_jobs_override,
    _sign_disagreements,
    attribute_period_change,
    attribute_target,
    build_node_plots,
    evaluate_causal_model,
    fit_causal_model,
    intervene,
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


def _root_binary_target_dag_and_data(
    n_rows: int = 400, random_state: int = 0
) -> tuple[DAGModel, pd.DataFrame]:
    """root -> binary_target, "+", root strongly predictive.

    Strong enough separation that a held-out test fold reliably lands
    both classes and a well-above-chance AUC, without needing a huge
    `n_rows` to keep the test fast.
    """
    rng = np.random.default_rng(random_state)
    root = rng.normal(0, 1, size=n_rows)
    prob = 1 / (1 + np.exp(-3.0 * root))
    binary_target = (rng.uniform(size=n_rows) < prob).astype(float)
    data = pd.DataFrame({"root": root, "binary_target": binary_target})

    dag = DAGModel()
    dag.add_node("root")
    dag.add_node("binary_target")
    dag.add_edge("root", "binary_target", "+")
    return dag, data


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


def test_root_node_noise_override_gaussian_gets_scipy_normal_distribution() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data, noise_models={"root": "gaussian"})
    mechanism = fitted.scm.causal_mechanism("root")
    assert isinstance(mechanism, gcm.ScipyDistribution)
    assert mechanism.scipy_distribution is norm


def test_root_node_noise_unspecified_still_defaults_to_empirical() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data, noise_models={})
    assert isinstance(fitted.scm.causal_mechanism("root"), gcm.EmpiricalDistribution)


def test_root_node_gaussian_noise_actually_changes_sampled_output() -> None:
    """SCOPE.md build order step 7: the noise override has to change what
    gets *sampled*, not just the mechanism's Python type (that part is
    already covered above). Distinguishes the two noise choices by a
    property each guarantees structurally, not by a threshold on the
    samples themselves: `gcm.EmpiricalDistribution` resamples with
    replacement directly from the observed column, so every value it
    draws is necessarily one of the observed values; `gcm.ScipyDistribution
    (scipy.stats.norm)` draws from a fitted continuous Normal instead, so
    a freshly drawn value landing on an exact observed float is
    vanishingly unlikely. If the noise override silently had no effect
    on sampling, the gaussian-fitted mechanism's draws would still all
    match observed values like the empirical one's do.
    """
    dag, data = _root_mid_target_dag_and_data()
    fitted_empirical = _fit(dag, data)
    fitted_gaussian = _fit(dag, data, noise_models={"root": "gaussian"})

    empirical_plot = build_node_plots(fitted_empirical, data, random_state=0)["root"]
    gaussian_plot = build_node_plots(fitted_gaussian, data, random_state=0)["root"]
    assert isinstance(empirical_plot, ObservedVsSampledPlot)
    assert isinstance(gaussian_plot, ObservedVsSampledPlot)

    observed_values = set(empirical_plot.observed)
    assert all(value in observed_values for value in empirical_plot.sampled)
    assert all(value not in observed_values for value in gaussian_plot.sampled)


def test_noise_models_unknown_node_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    with pytest.raises(KeyError):
        _fit(dag, data, noise_models={"nope": "gaussian"})


def test_noise_models_non_root_node_raises_value_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    with pytest.raises(ValueError, match="root nodes"):
        _fit(dag, data, noise_models={"mid": "gaussian"})


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


def test_attribute_target_shares_sum_to_one_including_target_row() -> None:
    """`share` is each row's contribution as a fraction of the total
    across the call -- should sum to ~1.0 across all rows (target's own
    row included, per `AttributionResult`'s docstring), not just across
    the "real driver" rows."""
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
    assert sum(r.share for r in results) == pytest.approx(1.0)
    by_node = {r.node: r.share for r in results}
    # root/mid are genuine drivers; target's own row is its residual
    # noise, not a driver, but still gets a (small, non-negative) share
    # rather than being excluded from the total.
    assert by_node["root"] > 0
    assert by_node["mid"] > 0
    assert by_node["target"] >= 0


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


# -- interventions --------------------------------------------------------------


def test_intervene_unknown_node_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(KeyError, match="nope"):
        intervene(fitted, data, "nope", 1.0, 2.0, "target")


def test_intervene_unknown_target_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(KeyError, match="nope"):
        intervene(fitted, data, "root", 1.0, 2.0, "nope")


def test_intervene_non_descendant_target_raises_value_error() -> None:
    """`target` must be a strict descendant of `node`: an ancestor
    cannot change under `do()` by definition (see causal_model.py's
    `intervene` docstring)."""
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(ValueError, match="not a descendant"):
        intervene(fitted, data, "target", 1.0, 2.0, "root")


def test_intervene_node_as_its_own_target_raises_value_error() -> None:
    """A node is not its own descendant, so `target == node` hits the
    same guard as an ancestor target: both would be a no-op or a
    tautology (see docstring)."""
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(ValueError, match="not a descendant"):
        intervene(fitted, data, "mid", 1.0, 2.0, "mid")


def test_intervene_moves_target_in_asserted_direction() -> None:
    """root -> mid -> target, both edges "+" (see
    `_root_mid_target_dag_and_data`): comparing a "from" well below
    `root`'s observed range against a "to" well above it should raise
    `target`'s mean, not lower it -- the regressor's own
    `monotonic_cst` hard-enforces the asserted sign (see
    causal_model.py's module docstring), so this holds robustly, not
    just on average. v2 (SCOPE.md's "Revised: v2, two-value
    comparison"): both values are synthetic `do()` draws now, neither
    is the real observed baseline."""
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = intervene(fitted, data, "root", -8.0, 8.0, "target")
    assert isinstance(result, InterventionResult)
    assert result.node == "root"
    assert result.target == "target"
    assert result.from_value == -8.0
    assert result.to_value == 8.0
    assert result.to_mean > result.from_mean
    assert result.absolute_change == pytest.approx(result.to_mean - result.from_mean)


def test_intervene_percent_change_is_absolute_change_over_from_mean() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = intervene(fitted, data, "root", 0.0, 5.0, "target")
    assert result.from_mean != pytest.approx(0.0)
    assert result.percent_change == pytest.approx(result.absolute_change / abs(result.from_mean))


def test_intervene_percent_change_is_none_for_near_zero_from_mean() -> None:
    """A near-zero "from" mean makes percent change meaningless (see
    `InterventionResult`'s docstring).

    v1's equivalent test recentered `target` to a zero mean and read
    `baseline_mean` straight off that real column -- exactly zero by
    construction, no model involved. v2's `from_mean` is a *predicted*
    quantity instead (`gcm.interventional_samples`' own resampled
    residual noise, not the real column), so recentering the real data
    and hoping a `do()` prediction lands within `1e-9` of the same
    number no longer works: a fitted mechanism's own residual
    resampling introduces sampling noise several orders of magnitude
    larger than `1e-9` (checked directly -- the recentering approach
    failed this assertion in practice). A `target` column that is
    exactly constant (zero variance) sidesteps this: the fitted
    mechanism's residuals are all exactly zero too, so every `do()`
    prediction -- `from_value` included -- comes back as exactly `0.0`,
    comfortably inside the guard's `1e-9` threshold regardless of which
    two values are compared.
    """
    dag, data = _root_mid_target_dag_and_data()
    data = data.copy()
    data["target"] = 0.0
    fitted = _fit(dag, data)
    result = intervene(fitted, data, "root", 0.0, 5.0, "target")
    assert result.from_mean == pytest.approx(0.0, abs=1e-9)
    assert result.percent_change is None


def test_intervene_csat_scenario_age_lowers_csat() -> None:
    """`age -> csat` is a direct, "-" edge in the CSAT scenario (see
    `_csat_dag` above), and `age` has no other outgoing edge, so `csat`
    is its only descendant -- a clean single-hop check against real
    demo data, not just the small hand-built fixture the rest of this
    file uses. v2: "from" is a typical age (the column's own observed
    mean), "to" is well above the observed range, both synthetic."""
    from dagshop.demo_data import make_csat_demo_data

    data = make_csat_demo_data(n_rows=250, random_state=0)
    dag = _csat_dag()
    fitted = _fit(dag, data)

    typical_age = float(data["age"].mean())
    high_age = float(data["age"].max() + 5 * data["age"].std())
    result = intervene(fitted, data, "age", typical_age, high_age, "csat")
    assert result.to_mean < result.from_mean


# -- private-helper branch coverage ----------------------------------------------


def test_n_jobs_override_is_noop_when_none() -> None:
    from dowhy.gcm import config as gcm_config

    before = gcm_config.default_n_jobs
    with _n_jobs_override(None):
        assert gcm_config.default_n_jobs == before
    assert gcm_config.default_n_jobs == before


# -- attribute_period_change -----------------------------------------------------


def test_attribute_period_change_unknown_target_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(KeyError, match="nope"):
        attribute_period_change(fitted, data, data, "nope", n_jobs=1)


def test_attribute_period_change_missing_column_in_new_data_raises_key_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    with pytest.raises(KeyError, match="target"):
        attribute_period_change(fitted, data, data.drop(columns=["target"]), "target", n_jobs=1)


def test_attribute_period_change_non_numeric_column_raises_column_type_error() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    bad_new_data = data.copy()
    bad_new_data["root"] = bad_new_data["root"].astype(str)
    with pytest.raises(ColumnTypeError, match="root"):
        attribute_period_change(fitted, data, bad_new_data, "target", n_jobs=1)


def test_attribute_period_change_returns_every_ancestor_and_target() -> None:
    dag, data = _root_mid_target_dag_and_data(n_rows=300, random_state=0)
    fitted = _fit(dag, data)
    _, new_data = _root_mid_target_dag_and_data(n_rows=300, random_state=1)
    results = attribute_period_change(fitted, data, new_data, "target", num_samples=50, n_jobs=1)
    assert {r.node for r in results} == {"root", "mid", "target"}
    assert all(isinstance(r, PeriodAttributionResult) for r in results)


def test_attribute_period_change_contributions_sum_to_actual_mean_change() -> None:
    # Shapley efficiency: contributions should sum to close to the
    # target's real observed mean change between the two dataframes.
    dag, old_data = _root_mid_target_dag_and_data(n_rows=1000, random_state=0)
    fitted = _fit(dag, old_data)
    _, new_data = _root_mid_target_dag_and_data(n_rows=1000, random_state=1)
    results = attribute_period_change(
        fitted, old_data, new_data, "target", num_samples=200, n_jobs=1, random_state=0
    )
    total_contribution = sum(r.contribution for r in results)
    actual_change = new_data["target"].mean() - old_data["target"].mean()
    assert total_contribution == pytest.approx(actual_change, abs=0.5)


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


# -- evaluate_causal_model -------------------------------------------------------


def test_evaluate_causal_model_returns_result_with_report() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = evaluate_causal_model(fitted, data, n_jobs=1)
    assert isinstance(result, ModelEvaluation)
    assert isinstance(result.report, str)
    assert len(result.report) > 0
    assert isinstance(result.overall_kl_divergence, float)
    assert result.graph_falsification.significance_level == 0.05
    assert result.graph_falsification.falsified in (True, False)


def test_evaluate_causal_model_root_node_gets_kl_divergence_not_r2() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = evaluate_causal_model(fitted, data, n_jobs=1)
    root_performance = result.mechanism_performances["root"]
    assert isinstance(root_performance, MechanismPerformance)
    assert root_performance.is_root is True
    assert root_performance.kl_divergence is not None
    assert root_performance.r2 is None
    assert root_performance.mse is None
    assert root_performance.nmse is None


def test_evaluate_causal_model_non_root_node_gets_r2_not_kl_divergence() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = evaluate_causal_model(fitted, data, n_jobs=1)
    target_performance = result.mechanism_performances["target"]
    assert target_performance.is_root is False
    assert target_performance.kl_divergence is None
    assert target_performance.r2 is not None
    assert target_performance.mse is not None
    assert target_performance.nmse is not None
    assert target_performance.crps is not None


def test_evaluate_causal_model_significance_level_overrides_dowhy_default() -> None:
    """dowhy's own EvaluateCausalModelConfig defaults
    falsify_graph_significance_level to 0.2; this module's wrapper
    defaults to 0.05 instead, the module's original default for graph
    falsification from its now-deleted `falsify_causal_graph` (see
    module docstring).
    """
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    result = evaluate_causal_model(fitted, data, significance_level=0.01, n_jobs=1)
    assert result.graph_falsification.significance_level == 0.01


# -- build_node_plots -------------------------------------------------------------


def test_is_binary_coded() -> None:
    assert _is_binary_coded(pd.Series([0.0, 1.0, 0.0, 1.0, np.nan])) is True
    assert _is_binary_coded(pd.Series([0.0, 1.0, 2.0])) is False
    assert _is_binary_coded(pd.Series([1.0, 1.0, 1.0])) is False


def test_build_node_plots_root_node_gets_observed_vs_sampled() -> None:
    dag, data = _root_mid_target_dag_and_data()
    fitted = _fit(dag, data)
    plots = build_node_plots(fitted, data)
    root_plot = plots["root"]
    assert isinstance(root_plot, ObservedVsSampledPlot)
    assert len(root_plot.observed) == len(data)
    assert len(root_plot.sampled) == len(root_plot.observed)


def test_build_node_plots_non_root_continuous_node_gets_no_auc() -> None:
    dag, data = _root_mid_target_dag_and_data(n_rows=300)
    fitted = _fit(dag, data)
    plots = build_node_plots(fitted, data, test_size=0.2, random_state=0)
    target_plot = plots["target"]
    assert isinstance(target_plot, ActualVsPredictedPlot)
    expected_test_rows = round(300 * 0.2)
    assert len(target_plot.actual) == len(target_plot.predicted) == expected_test_rows
    assert target_plot.auc is None


def test_build_node_plots_binary_node_gets_auc_well_above_chance() -> None:
    dag, data = _root_binary_target_dag_and_data()
    fitted = _fit(dag, data)
    plots = build_node_plots(fitted, data, random_state=0)
    target_plot = plots["binary_target"]
    assert isinstance(target_plot, ActualVsPredictedPlot)
    assert target_plot.auc is not None
    assert target_plot.auc > 0.7


def test_build_node_plots_shares_one_split_across_nodes() -> None:
    """A different random_state changes which rows land in the test
    fold -- checked indirectly via the actual-vs-predicted plot's own
    values differing, since this module doesn't expose the split
    itself.
    """
    dag, data = _root_mid_target_dag_and_data(n_rows=300)
    fitted = _fit(dag, data)
    plots_a = build_node_plots(fitted, data, random_state=0)
    plots_b = build_node_plots(fitted, data, random_state=1)
    assert plots_a["target"].actual != plots_b["target"].actual


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
        ("friction_severity", "num_transfers", "+"),
        ("friction_severity", "num_escalations", "+"),
        ("num_transfers", "num_agents_spoken_to", "+"),
        ("num_escalations", "num_agents_spoken_to", "+"),
        ("num_transfers", "time_to_resolve", "+"),
        ("num_escalations", "time_to_resolve", "+"),
        ("num_agents_spoken_to", "time_to_resolve", "+"),
        ("time_to_resolve", "resolved", "-"),
        ("num_transfers", "resolved", "-"),
        ("num_escalations", "resolved", "-"),
        ("age", "csat", "-"),
        ("time_to_respond", "csat", "-"),
        ("repeat_contact", "csat", "-"),
        ("time_to_resolve", "csat", "-"),
        ("num_agents_spoken_to", "csat", "-"),
        ("resolved", "csat", "+"),
    ]
    for source, target, sign in edges:
        dag.add_edge(source, target, sign)
    return dag


def test_csat_scenario_end_to_end_ranks_resolved_highest() -> None:
    """`resolved` is documented (demo_data.py's own docstring) as the
    strongest ancestor of `csat` in this scenario -- it is `csat`'s
    largest-coefficient direct parent. This is the closest thing to a
    regression test for the whole module working together against
    real, documented ground truth, not just its pieces in isolation.

    `csat`'s own row (its intrinsic/unexplained variance, not an
    ancestor) is excluded before ranking: at these reduced sample
    settings it is the single largest value in the raw result, which
    would make `max()` pick `csat` itself rather than any real driver
    -- see demo_data.py's docstring for why that row is large here, and
    SCOPE.md's "Requested changes" backlog for the UI-side "Other"
    relabeling this maps to.
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
    by_node = {r.node: r.contribution for r in results if r.node != "csat"}
    assert max(by_node, key=by_node.get) == "resolved"


def test_evaluate_causal_model_csat_scenario_produces_believable_numbers() -> None:
    """SCOPE.md build order step 7: `evaluate_causal_model` against the
    real, documented CSAT scenario, not just the tiny root-mid-target
    fixture the tests above use -- and specifically root-node KL
    divergence at that scale, another step 7 item.

    `n_rows=250` rather than the 1000 used elsewhere in this file: this
    test's `falsify_graph` permutation test scales poorly with row
    count (measured ~150s at 1000 rows with `n_jobs=1` forced, enough
    to blow past this project's per-command timeout in the sandboxed
    bridge; ~20s at 250). `random_state=0` is passed to
    `evaluate_causal_model` itself (not just `make_csat_demo_data`)
    because its R2/CRPS/KL numbers turned out to depend on it too --
    `dowhy`'s internal `KFold(shuffle=True)` draws from numpy's global
    legacy RNG rather than a seeded local one, so two runs with
    identical inputs produced different numbers until this was found
    (see this module's docstring and `evaluate_causal_model`'s own for
    the fix, `_random_state_override`, shared with `attribute_target`'s
    pre-existing use of the same mechanism).

    Thresholds below are the actual measured values at this exact
    `n_rows`/`random_state`, confirmed identical across two back-to-back
    runs (checked directly before writing this test, not guessed):
    `csat`'s R2 came back ~0.312, comfortably clear of the 0.25 floor
    here. Not asserting a floor on `resolved`'s own R2 (~-0.005
    measured, i.e. worse than predicting the mean): `demo_data.py`'s
    docstring only documents `resolved` as `csat`'s strongest
    Shapley-attributed ancestor (already covered by the test above),
    not as itself easy to predict from its own direct parents.
    """
    from dagshop.demo_data import make_csat_demo_data

    data = make_csat_demo_data(n_rows=250, random_state=0)
    dag = _csat_dag()
    fitted = _fit(dag, data)
    result = evaluate_causal_model(fitted, data, n_jobs=1, random_state=0)

    assert set(result.mechanism_performances) == set(dag.nodes)
    root_nodes = {"age", "friction_severity", "time_to_respond", "repeat_contact"}
    for node in root_nodes:
        performance = result.mechanism_performances[node]
        assert performance.is_root is True
        assert performance.kl_divergence is not None
        assert performance.kl_divergence >= 0
        assert performance.r2 is None
        assert performance.crps is None

    for node in set(dag.nodes) - root_nodes:
        performance = result.mechanism_performances[node]
        assert performance.is_root is False
        assert performance.kl_divergence is None
        assert performance.r2 is not None
        assert performance.crps is not None

    csat_performance = result.mechanism_performances["csat"]
    assert csat_performance.r2 > 0.25

    assert result.overall_kl_divergence >= 0
    assert result.graph_falsification.falsified in (True, False)
    assert result.graph_falsification.falsifiable in (True, False)
    assert isinstance(result.report, str) and len(result.report) > 0


def test_build_node_plots_csat_scenario_resolved_gets_auc() -> None:
    """SCOPE.md build order step 7: the AUC path for a binary-coded node,
    against the real CSAT scenario specifically (the small, strongly-
    separable synthetic fixture in `test_build_node_plots_binary_node_
    gets_auc_well_above_chance` above already covers the "well above
    chance" case). `resolved`'s own relationship to its direct parents
    is comparatively weak in this generator (measured AUC ~0.568 at
    this `n_rows`/`random_state`, barely above the 0.5 floor) -- see
    this test's sibling above for why that's expected, not a bug. Only
    asserting the AUC path actually produces a real number here, not a
    strength floor that would be flaky against a genuinely weak signal.

    `n_rows=250`, matching the sibling test above, for the same reason:
    keeps the module's slower CSAT-scale tests under this project's
    per-command timeout in the sandboxed bridge.
    """
    from dagshop.demo_data import make_csat_demo_data

    data = make_csat_demo_data(n_rows=250, random_state=0)
    dag = _csat_dag()
    fitted = _fit(dag, data)
    plots = build_node_plots(fitted, data, random_state=0)
    resolved_plot = plots["resolved"]
    assert isinstance(resolved_plot, ActualVsPredictedPlot)
    assert resolved_plot.auc is not None


def test_attribute_period_change_csat_scenario_finds_both_ground_truth_changes() -> None:
    """`demo_data.make_csat_period_comparison_data`'s docstring documents
    two deliberate, known changes between periods: a genuine mechanism
    change at `num_transfers` (its own `friction_severity` coefficient
    strengthens), and a pure distribution shift at the root
    `time_to_respond` (its mean rises, no edge coefficient changes). A
    correct estimator should flag both -- and only these two -- as
    `mechanism_changed=True`, and rank both among the largest-magnitude
    contributors to `csat`'s mean change.

    `n_rows=500`/`num_samples=100`/`random_state=0` checked directly
    against this exact scenario before picking these numbers: fast
    (~5s) while still reliably surfacing both known changes -- not
    fitted to make the assertion pass, this is the same known-ground-
    truth structure the module docstring documents.
    """
    from dagshop.demo_data import make_csat_period_comparison_data

    data = make_csat_period_comparison_data(n_rows=500, random_state=0)
    old_data = data.loc[data["period"] == "baseline"].drop(columns="period")
    new_data = data.loc[data["period"] == "new"].drop(columns="period")
    dag = _csat_dag()
    fitted = _fit(dag, old_data)

    results = attribute_period_change(
        fitted, old_data, new_data, "csat", num_samples=100, n_jobs=1, random_state=0
    )
    changed_nodes = {r.node for r in results if r.mechanism_changed}
    assert changed_nodes == {"num_transfers", "time_to_respond"}

    by_magnitude = sorted(results, key=lambda r: abs(r.contribution), reverse=True)
    top_two = {r.node for r in by_magnitude[:2]}
    assert top_two == {"num_transfers", "time_to_respond"}
