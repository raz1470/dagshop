"""Tests for `demo_data.py`: the synthetic datasets used by
`dagshop generate-demo-data` (see the module docstring, and each
generator's own docstring, for the intended causal structure and why
each exists)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dagshop.demo_data import (
    make_csat_demo_data,
    make_csat_period_comparison_data,
    make_demo_data,
)


def test_default_shape_and_columns():
    data = make_demo_data()
    assert len(data) == 500
    assert list(data.columns) == [
        "age",
        "prior_engagement",
        "treatment",
        "mediator",
        "outcome",
        "unrelated_score",
        "unrelated_flag",
    ]


def test_n_rows_is_respected():
    data = make_demo_data(n_rows=37)
    assert len(data) == 37


def test_binary_columns_are_zero_or_one():
    data = make_demo_data(n_rows=200, random_state=1)
    assert set(data["treatment"].unique()) <= {0, 1}
    assert set(data["unrelated_flag"].unique()) <= {0, 1}
    # both classes actually occur at this size, or the association scan
    # (which needs 2 classes per binary column) has nothing to score
    assert set(data["treatment"].unique()) == {0, 1}
    assert set(data["unrelated_flag"].unique()) == {0, 1}


def test_same_random_state_is_reproducible():
    first = make_demo_data(n_rows=50, random_state=5)
    second = make_demo_data(n_rows=50, random_state=5)
    pd.testing.assert_frame_equal(first, second)


def test_different_random_state_differs():
    first = make_demo_data(n_rows=50, random_state=5)
    second = make_demo_data(n_rows=50, random_state=6)
    assert not first["age"].equals(second["age"])


def test_treatment_and_outcome_are_actually_associated():
    # Sanity check on the ground truth this module claims to build:
    # treated rows should have a visibly higher mean outcome than
    # untreated rows, at a size and effect this large.
    data = make_demo_data(n_rows=2000, random_state=0)
    treated_mean = data.loc[data["treatment"] == 1, "outcome"].mean()
    untreated_mean = data.loc[data["treatment"] == 0, "outcome"].mean()
    assert treated_mean - untreated_mean > 3.0


def test_unrelated_columns_are_uncorrelated_with_outcome():
    data = make_demo_data(n_rows=2000, random_state=0)
    corr = np.corrcoef(data["unrelated_score"], data["outcome"])[0, 1]
    assert abs(corr) < 0.1


# -- make_csat_demo_data ------------------------------------------------------


def test_csat_default_shape_and_columns():
    data = make_csat_demo_data()
    assert len(data) == 500
    assert list(data.columns) == [
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
    ]


def test_csat_n_rows_is_respected():
    data = make_csat_demo_data(n_rows=37)
    assert len(data) == 37


def test_csat_binary_columns_are_zero_or_one():
    data = make_csat_demo_data(n_rows=500, random_state=1)
    assert set(data["resolved"].unique()) <= {0, 1}
    assert set(data["repeat_contact"].unique()) <= {0, 1}
    # both classes actually occur at this size, or the association scan
    # (which needs 2 classes per binary column) has nothing to score
    assert set(data["resolved"].unique()) == {0, 1}
    assert set(data["repeat_contact"].unique()) == {0, 1}


def test_csat_same_random_state_is_reproducible():
    first = make_csat_demo_data(n_rows=50, random_state=5)
    second = make_csat_demo_data(n_rows=50, random_state=5)
    pd.testing.assert_frame_equal(first, second)


def test_csat_different_random_state_differs():
    first = make_csat_demo_data(n_rows=50, random_state=5)
    second = make_csat_demo_data(n_rows=50, random_state=6)
    assert not first["friction_severity"].equals(second["friction_severity"])


def test_csat_nonnegative_columns_stay_nonnegative():
    # friction_severity/time_to_respond are gamma-distributed (>= 0 by
    # construction); the count columns and time_to_resolve are clipped
    # or built from nonnegative pieces. A negative value here would mean
    # a clip/construction bug, not a modelling choice.
    data = make_csat_demo_data(n_rows=2000, random_state=0)
    for column in [
        "friction_severity",
        "time_to_respond",
        "num_transfers",
        "num_escalations",
        "num_agents_spoken_to",
        "time_to_resolve",
    ]:
        assert (data[column] >= 0).all(), column


def test_csat_is_within_documented_range():
    data = make_csat_demo_data(n_rows=2000, random_state=0)
    assert data["csat"].between(0.0, 10.0).all()


def test_csat_resolved_is_strongly_associated_with_csat():
    # Sanity check on the ground truth the module docstring claims:
    # resolved should be the single strongest correlate of csat, and by
    # a wide margin -- it is csat's largest-coefficient direct parent.
    data = make_csat_demo_data(n_rows=5000, random_state=0)
    corr = data.corr(numeric_only=True)["csat"].drop("csat").abs()
    assert corr.idxmax() == "resolved"
    assert corr["resolved"] > 0.4


def test_csat_indirect_ancestors_are_still_correlated_with_csat():
    # The point of this scenario (SCOPE.md's "Causal attribution
    # feature" section): friction_severity, num_transfers, and
    # num_escalations are NOT direct parents of csat, only ancestors
    # through time_to_resolve/resolved/num_agents_spoken_to. If the
    # generator's multi-hop wiring were broken (e.g. a dropped term),
    # one of these would read as disconnected noise instead.
    data = make_csat_demo_data(n_rows=5000, random_state=0)
    corr = data.corr(numeric_only=True)["csat"].drop("csat").abs()
    for column in ["friction_severity", "num_transfers", "num_escalations"]:
        assert corr[column] > 0.15, column


def test_csat_age_and_friction_severity_are_the_weakest_correlates():
    # Weakest two by raw correlation, per make_csat_demo_data's
    # docstring -- not the same as weakest two by intrinsic causal
    # influence (also documented there): correlation is a marginal,
    # single-column measure, so a driver can rank low here while still
    # carrying a real multi-hop Shapley contribution.
    data = make_csat_demo_data(n_rows=5000, random_state=0)
    corr = data.corr(numeric_only=True)["csat"].drop("csat").abs()
    weakest_two = corr.nsmallest(2).index.tolist()
    assert set(weakest_two) == {"age", "friction_severity"}


# -- make_csat_period_comparison_data ----------------------------------------


def test_period_comparison_shape_and_columns():
    data = make_csat_period_comparison_data(n_rows=50)
    assert len(data) == 100
    assert list(data.columns) == [
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
        "period",
    ]


def test_period_comparison_has_both_labels_in_equal_counts():
    data = make_csat_period_comparison_data(n_rows=50)
    assert data["period"].value_counts().to_dict() == {"baseline": 50, "new": 50}


def test_period_comparison_same_random_state_is_reproducible():
    first = make_csat_period_comparison_data(n_rows=30, random_state=5)
    second = make_csat_period_comparison_data(n_rows=30, random_state=5)
    pd.testing.assert_frame_equal(first, second)


def test_period_comparison_different_random_state_differs():
    first = make_csat_period_comparison_data(n_rows=30, random_state=5)
    second = make_csat_period_comparison_data(n_rows=30, random_state=6)
    baseline_first = first.loc[first["period"] == "baseline", "friction_severity"]
    baseline_second = second.loc[second["period"] == "baseline", "friction_severity"]
    assert not baseline_first.reset_index(drop=True).equals(baseline_second.reset_index(drop=True))


def test_period_comparison_baseline_rows_match_make_csat_demo_data():
    # The "baseline" half must be exactly make_csat_demo_data's own output
    # (same seed, same defaults) -- the whole point is a known, unmodified
    # reference period to compare the deliberately-changed one against.
    combined = make_csat_period_comparison_data(n_rows=40, random_state=7)
    baseline_rows = combined.loc[combined["period"] == "baseline"].drop(columns="period")
    reference = make_csat_demo_data(n_rows=40, random_state=7)
    pd.testing.assert_frame_equal(baseline_rows.reset_index(drop=True), reference)


def test_period_comparison_num_transfers_mean_rises_with_the_mechanism_change():
    # friction_severity -> num_transfers deliberately strengthens
    # (num_transfers_friction_coef 0.18 -> 0.30) in the new period, holding
    # friction_severity's own distribution fixed -- a real mechanism
    # change, not just a moved input. Large n to keep this a stable check,
    # not a seed-dependent coin flip.
    data = make_csat_period_comparison_data(n_rows=5000, random_state=0)
    baseline_mean = data.loc[data["period"] == "baseline", "num_transfers"].mean()
    new_mean = data.loc[data["period"] == "new", "num_transfers"].mean()
    assert new_mean > baseline_mean * 1.2


def test_period_comparison_time_to_respond_mean_rises_with_the_distribution_shift():
    # time_to_respond (a root) shifts from gamma(scale=5.0) to
    # gamma(scale=7.0) -- mean 15 -> 21 -- with no change to its edge into
    # csat. A pure distribution shift on a root, not a mechanism change.
    data = make_csat_period_comparison_data(n_rows=5000, random_state=0)
    baseline_mean = data.loc[data["period"] == "baseline", "time_to_respond"].mean()
    new_mean = data.loc[data["period"] == "new", "time_to_respond"].mean()
    assert new_mean > baseline_mean * 1.2


def test_period_comparison_csat_mean_drops_between_periods():
    # Both deliberate changes push csat down (more transfers, slower
    # response both hurt csat per make_csat_demo_data's documented signs)
    # -- a sanity check that the scenario actually produces a period-over-
    # period change worth attributing, not a wash.
    data = make_csat_period_comparison_data(n_rows=5000, random_state=0)
    baseline_mean = data.loc[data["period"] == "baseline", "csat"].mean()
    new_mean = data.loc[data["period"] == "new", "csat"].mean()
    assert baseline_mean - new_mean > 0.2
