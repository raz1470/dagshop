"""Tests for `demo_data.py`: the synthetic dataset used by
`dagshop generate-demo-data` (see its module docstring for the intended
causal structure and why it exists)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dagshop.demo_data import make_demo_data


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
