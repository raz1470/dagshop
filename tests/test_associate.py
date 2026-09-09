"""Unit tests for dagshop.associate.scan_associations.

Pure-logic module (pandas/sklearn only, no UI/graph.py dependency): see
SCOPE.md build order step 2. Synthetic DataFrames with known
relationships throughout, not real-data-shaped, so scores stay fast to
compute and their direction/bounds are checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dagshop.associate import (
    AssociationSkippedWarning,
    ColumnTypeError,
    PairResult,
    scan_associations,
)


def _linear_relationship(n: int = 200) -> pd.DataFrame:
    """x evenly spaced; y = 3x - 1 exactly (noiseless); z independent noise."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, n)
    y = 3.0 * x - 1.0
    z = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _separable_binary(n: int = 60) -> pd.DataFrame:
    """x = 0..n-1; y = 1 iff x >= n/2 (perfectly separable step); z independent noise."""
    rng = np.random.default_rng(1)
    x = np.arange(n, dtype=float)
    y = (x >= n / 2).astype(int)
    z = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _result_for(table: list[PairResult], predictor: str, target: str) -> PairResult:
    matches = [r for r in table if r.predictor == predictor and r.target == target]
    assert len(matches) == 1, (
        f"expected exactly one {predictor!r} -> {target!r} result, got {matches}"
    )
    return matches[0]


# -- full pairwise scan, continuous target ---------------------------------------


def test_full_scan_covers_every_ordered_pair() -> None:
    data = _linear_relationship(n=200)
    scan = scan_associations(data)
    assert not scan.scoped
    assert scan.treatment_table == []
    assert scan.outcome_table == []
    # n * (n - 1) ordered pairs across 3 columns: 3 * 2 = 6.
    assert len(scan.full_table) == 6
    pairs = {(r.predictor, r.target) for r in scan.full_table}
    assert pairs == {("x", "y"), ("x", "z"), ("y", "x"), ("y", "z"), ("z", "x"), ("z", "y")}


def test_continuous_target_uses_r2_and_ranks_signal_above_noise() -> None:
    data = _linear_relationship(n=200)
    scan = scan_associations(data)

    signal = _result_for(scan.full_table, "x", "y")
    noise = _result_for(scan.full_table, "z", "y")

    assert signal.score_name == "r2"
    assert noise.score_name == "r2"
    # y = 3x - 1 exactly: a noiseless, monotonic relationship should score
    # well above a model that can only see independent random noise.
    assert signal.score > 0.8
    assert noise.score < 0.3
    assert signal.score > noise.score


# -- full pairwise scan, binary target -------------------------------------------


def test_binary_target_uses_roc_auc_and_predict_proba() -> None:
    data = _separable_binary(n=60)
    scan = scan_associations(data)

    signal = _result_for(scan.full_table, "x", "y")
    noise = _result_for(scan.full_table, "z", "y")

    assert signal.score_name == "roc_auc"
    assert noise.score_name == "roc_auc"
    # y = 1 iff x >= 30 is a clean, monotonic split along x: the model
    # should rank almost every positive above almost every negative on
    # the held-out fold (min_samples_leaf=20 keeps HistGradientBoosting
    # from placing the split exactly at the boundary with only ~48
    # training rows, so this falls just short of the theoretical 1.0).
    assert signal.score > 0.85
    # z carries no information about y: AUC should sit well below the
    # perfect predictor, scattered around the 0.5 no-skill baseline.
    assert 0.15 <= noise.score <= 0.85
    assert signal.score > noise.score


# -- treatment/outcome scoping ----------------------------------------------------


def test_scoped_scan_produces_treatment_and_outcome_tables() -> None:
    data = _linear_relationship(n=200)
    data = data.assign(w=np.linspace(5.0, 15.0, len(data)))  # extra unrelated variable
    scan = scan_associations(data, treatments=["x"], outcomes=["y"])

    assert scan.scoped
    assert scan.full_table == []
    # n * (t + o) with n=4 columns (x, y, z, w), t=1, o=1: fits against x
    # exclude x itself (3 predictors), same for y: 3 + 3 = 6.
    assert len(scan.treatment_table) == 3
    assert len(scan.outcome_table) == 3
    assert {r.target for r in scan.treatment_table} == {"x"}
    assert {r.target for r in scan.outcome_table} == {"y"}
    assert {r.predictor for r in scan.treatment_table} == {"y", "z", "w"}
    assert {r.predictor for r in scan.outcome_table} == {"x", "z", "w"}
    # covariates = columns minus treatment(x)/outcome(y) = {z, w}: every
    # ordered pair among them, c * (c - 1) = 2 * 1 = 2.
    assert len(scan.covariate_table) == 2
    assert {(r.predictor, r.target) for r in scan.covariate_table} == {("z", "w"), ("w", "z")}


def test_scoped_scan_with_only_treatments_leaves_outcome_table_empty() -> None:
    data = _linear_relationship(n=200)
    scan = scan_associations(data, treatments=["x"])
    assert scan.scoped
    assert scan.outcome_table == []
    assert scan.treatment_table != []


def test_covariate_table_empty_when_fewer_than_two_covariates() -> None:
    # Only "z" is left over once x/y are designated: not enough covariates
    # to form an ordered pair.
    data = _linear_relationship(n=200)
    scan = scan_associations(data, treatments=["x"], outcomes=["y"])
    assert scan.covariate_table == []


def test_covariate_table_empty_when_unscoped() -> None:
    # Unscoped mode already covers every pair via full_table.
    data = _linear_relationship(n=200)
    scan = scan_associations(data)
    assert scan.covariate_table == []


def test_unknown_treatment_name_raises_value_error() -> None:
    data = _linear_relationship(n=200)
    with pytest.raises(ValueError, match="treatments"):
        scan_associations(data, treatments=["not_a_column"])


def test_unknown_outcome_name_raises_value_error() -> None:
    data = _linear_relationship(n=200)
    with pytest.raises(ValueError, match="outcomes"):
        scan_associations(data, outcomes=["not_a_column"])


# -- ranking table sort order -----------------------------------------------------


def test_table_sorted_by_score_descending() -> None:
    data = _linear_relationship(n=200)
    scan = scan_associations(data)
    scores = [r.score for r in scan.full_table]
    assert scores == sorted(scores, reverse=True)


# -- subsampling --------------------------------------------------------------------


def test_subsampling_caps_rows_used() -> None:
    data = _linear_relationship(n=500)
    scan = scan_associations(data, max_rows=100)
    for result in scan.full_table:
        assert result.n_used <= 100


def test_no_subsampling_when_under_max_rows() -> None:
    data = _linear_relationship(n=50)
    scan = scan_associations(data, max_rows=5000)
    result = _result_for(scan.full_table, "x", "y")
    assert result.n_used == 50


def test_subsampling_is_deterministic_for_fixed_random_state() -> None:
    data = _linear_relationship(n=500)
    scan_a = scan_associations(data, max_rows=100, random_state=7)
    scan_b = scan_associations(data, max_rows=100, random_state=7)
    scores_a = {(r.predictor, r.target): r.score for r in scan_a.full_table}
    scores_b = {(r.predictor, r.target): r.score for r in scan_b.full_table}
    assert scores_a == scores_b


# -- column type validation ----------------------------------------------------------


def test_rejects_non_numeric_column() -> None:
    data = _linear_relationship(n=50)
    data["label"] = ["a", "b"] * 25
    with pytest.raises(ColumnTypeError, match="label"):
        scan_associations(data)


def test_rejects_constant_target_column() -> None:
    data = _linear_relationship(n=50)
    data["constant"] = 1.0
    with pytest.raises(ColumnTypeError, match="constant"):
        scan_associations(data)


def test_rejects_constant_designated_outcome() -> None:
    data = _linear_relationship(n=50)
    data["constant"] = 1.0
    with pytest.raises(ColumnTypeError, match="constant"):
        scan_associations(data, outcomes=["constant"])


def test_at_least_two_columns_required() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least 2 columns"):
        scan_associations(data)


# -- plot cache ------------------------------------------------------------------------


def test_plot_cache_has_entry_per_scanned_pair() -> None:
    data = _linear_relationship(n=200)
    scan = scan_associations(data, plot_grid_size=10)
    assert set(scan.plot_cache) == {(r.predictor, r.target) for r in scan.full_table}

    plot = scan.plot_cache[("x", "y")]
    assert plot.predictor == "x"
    assert plot.target == "y"
    assert len(plot.grid_x) == 10
    assert len(plot.grid_prediction) == 10
    assert len(plot.x) == len(plot.y)


def test_plot_cache_binary_prediction_is_a_probability() -> None:
    data = _separable_binary(n=60)
    scan = scan_associations(data)
    plot = scan.plot_cache[("x", "y")]
    assert all(0.0 <= p <= 1.0 for p in plot.grid_prediction)


# -- skipped pairs -----------------------------------------------------------------------


def test_too_few_rows_after_dropna_is_skipped_with_reason() -> None:
    data = _linear_relationship(n=50)
    # Blank out all but 3 rows of "z" so every pair involving it has too
    # few non-null rows to fit, while (x, y) is untouched.
    data.loc[3:, "z"] = np.nan

    with pytest.warns(AssociationSkippedWarning):
        scan = scan_associations(data)

    skipped_pairs = {(s.predictor, s.target) for s in scan.skipped}
    assert ("z", "y") in skipped_pairs
    assert ("y", "z") in skipped_pairs
    assert all("rows remain" in s.reason for s in scan.skipped)
    scanned_pairs = {(r.predictor, r.target) for r in scan.full_table}
    assert ("z", "y") not in scanned_pairs
    assert ("x", "y") in scanned_pairs


def test_binary_target_with_one_row_in_a_class_is_skipped() -> None:
    # 29 negatives, 1 positive: too few rows in the minority class to
    # even attempt a train/test split.
    data = pd.DataFrame(
        {
            "p": np.linspace(0.0, 1.0, 30),
            "y_rare": [0.0] * 29 + [1.0],
        }
    )
    with pytest.warns(AssociationSkippedWarning):
        scan = scan_associations(data, outcomes=["y_rare"])
    assert scan.outcome_table == []
    skipped = scan.skipped[0]
    assert skipped.predictor == "p"
    assert skipped.target == "y_rare"
    assert "fewer than 2 rows in one class" in skipped.reason


def test_binary_target_split_leaving_one_class_out_is_skipped() -> None:
    # Minority class has exactly 2 rows out of 42: at the default
    # test_size=0.2, sklearn's stratified split allocates 2 * 0.2 = 0.4
    # (rounds to 0) of the minority class to the test fold, so the test
    # fold ends up with only the majority class present.
    data = pd.DataFrame(
        {
            "p": np.linspace(0.0, 1.0, 42),
            "y_rare": [0.0] * 40 + [1.0] * 2,
        }
    )
    with pytest.warns(AssociationSkippedWarning):
        scan = scan_associations(data, outcomes=["y_rare"])
    assert scan.outcome_table == []
    skipped = scan.skipped[0]
    assert skipped.predictor == "p"
    assert skipped.target == "y_rare"
    assert "one class on one side" in skipped.reason
