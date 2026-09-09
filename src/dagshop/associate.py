"""Pre-work: scoped or full pairwise HistGradientBoosting association scan.

Purely exploratory: informs the PM/DS during the workshop, does not set
anything on the DAG automatically. For every relevant ordered pair
`(X, Y)`, fits a single-feature `Y ~ X` model using
`HistGradientBoostingRegressor` (continuous `Y`) or
`HistGradientBoostingClassifier` (binary `Y`), the same model family
`dowhy.gcm`'s auto-assignment uses at its default quality setting. Ranks
by association with designated treatment(s)/outcome(s) (`n * (t + o)`
fits) when designated, or falls back to a full pairwise scan
(`n * (n - 1)`) when not. No threshold-based flagging: this module
produces ranking table(s) and a per-pair plot cache only, per SCOPE.md's
"Sorting, not auto-flagging" -- spotting a variable that ranks high on
both the treatment and outcome tables is left to the PM/DS.

Depends only on pandas/sklearn, not on graph.py or the UI (dagshop has no
dowhy dependency at all -- the DAG this tool exports is loaded into the
user's own dowhy.gcm downstream, per SCOPE.md). See SCOPE.md build order
step 2.

Continuous and binary numeric columns only for v1 (SCOPE.md: categorical
variables are out of scope). `scan_associations` rejects any non-numeric
column with a `ColumnTypeError` up front, before fitting anything, rather
than silently coercing it.

Judgment calls made in this module, not directed by SCOPE.md or asked of
Ryan (flagging per PREFERENCES.md):

- **Held-out scoring.** SCOPE.md/Ryan decided the score *type* (R^2 for
  continuous, ROC AUC for binary, `predict_proba` not the raw class
  prediction) but not whether it is in-sample or held-out.
  `HistGradientBoosting*` can fit training data closely enough that an
  in-sample score would make most pairs look strongly associated
  regardless of the real relationship, which defeats the point of a
  ranking table. Each pair gets its own train/test split (`test_size`,
  default 0.2) and the reported score is computed on the held-out test
  fold. Binary targets are split with `stratify` so both folds keep both
  classes wherever the data allows it.
- **One subsample for the whole scan.** "Subsample rows by default" is
  applied once, at the top of `scan_associations`, rather than
  independently per pair. Every pair scans the same row subset, which
  keeps scores comparable across pairs and avoids resampling `n*(n-1)`
  times. Rows with a missing value in either column of a given pair are
  then dropped from that pair specifically (pairwise deletion), on top of
  the shared subsample.
- **Numeric-with->2-values means continuous.** A column is classified
  "binary" if it has exactly 2 distinct non-null values and "continuous"
  otherwise. v1 has no categorical handling (SCOPE.md), so an
  integer-coded categorical/ordinal column with 3+ levels will be treated
  as continuous rather than rejected -- a known v1 limitation, not a bug.
- **Skip, don't fail, on a degenerate pair.** A pair that has too few
  rows left after dropping missing values, or a binary target that ends
  up single-class in one split fold, is skipped (recorded in
  `AssociationScan.skipped` with a reason, and a `AssociationSkippedWarning`
  is issued) rather than aborting the whole scan. Mirrors graph.py's
  cycle handling: warn and continue, don't except.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

TargetKind = Literal["continuous", "binary"]
ScoreName = Literal["r2", "roc_auc"]

# Numpy dtype.kind codes accepted as "numeric" for v1: bool, signed int,
# unsigned int, float. Anything else (object, category, string,
# datetime, ...) is rejected by `_validate_dtypes`.
_NUMERIC_DTYPE_KINDS = frozenset("biuf")

# Below this many rows (after the shared subsample and pairwise dropna),
# a pair is skipped rather than fit: not enough data left for a
# meaningful train/test split.
_MIN_PAIR_ROWS = 4


class ColumnTypeError(ValueError):
    """A column is not a supported type for this scan.

    v1 only supports continuous and binary numeric columns (SCOPE.md:
    categorical variables are out of scope). Raised up front, before any
    fitting starts, naming every offending column at once.
    """


class AssociationSkippedWarning(UserWarning):
    """A pair was skipped rather than fit or scored.

    Not enough rows remained for that pair after the shared subsample and
    pairwise `dropna`, or a binary target's train/test split left only
    one class on one side. The pair is recorded in
    `AssociationScan.skipped` with a reason; it does not appear in any
    ranking table or the plot cache.
    """


@dataclass(frozen=True)
class PairResult:
    """One ranking-table row: how well `predictor` predicts `target`."""

    predictor: str
    target: str
    score: float
    score_name: ScoreName
    n_used: int


@dataclass(frozen=True)
class PairPlotData:
    """Cached scatter + prediction curve for one pair's plot modal.

    `x`/`y` are the actual (subsampled, pairwise-deleted) data points for
    the scatter. `grid_x` is a linspace across the observed range of `x`;
    `grid_prediction` is the fitted model's prediction at each grid
    point -- for a single-feature model this *is* the partial-dependence
    curve, since there are no other features to average over. For a
    binary target, `grid_prediction` is the predicted probability of the
    larger-valued class in `model.classes_` (sklearn's own ordering),
    not a raw class prediction.
    """

    predictor: str
    target: str
    x: list[float]
    y: list[float]
    grid_x: list[float]
    grid_prediction: list[float]


@dataclass(frozen=True)
class SkippedPair:
    """A pair that was not fit, and why."""

    predictor: str
    target: str
    reason: str


@dataclass
class AssociationScan:
    """Result of `scan_associations`.

    Exactly one of (`treatment_table` and/or `outcome_table`) or
    `full_table` is populated, matching `scoped`: SCOPE.md's "ranked
    association table(s)... split into 'associated with treatment(s)' /
    'associated with outcome(s)' shown side by side when designated,
    otherwise a single table." Each populated table is sorted by score
    descending (ties broken by predictor name, for determinism). No
    field here flags a variable as a confounder -- that reading is left
    to the PM/DS, per SCOPE.md's "Sorting, not auto-flagging."
    """

    scoped: bool
    treatment_table: list[PairResult]
    outcome_table: list[PairResult]
    full_table: list[PairResult]
    plot_cache: dict[tuple[str, str], PairPlotData]
    skipped: list[SkippedPair]


def scan_associations(
    data: pd.DataFrame,
    *,
    treatments: Sequence[str] | None = None,
    outcomes: Sequence[str] | None = None,
    max_rows: int = 5000,
    test_size: float = 0.2,
    random_state: int = 0,
    plot_grid_size: int = 50,
) -> AssociationScan:
    """Run the pre-work association scan described in SCOPE.md.

    Args:
        data: continuous/binary numeric columns only; a non-numeric
            column raises `ColumnTypeError` naming every offender before
            any fitting starts.
        treatments: column names designated as treatment variables. If
            this and/or `outcomes` is non-empty, the scan is scoped:
            every column vs every treatment, and every column vs every
            outcome (`n * (t + o)` fits). Names not found in `data`'s
            columns raise `ValueError`.
        outcomes: column names designated as outcome variables. See
            `treatments`.
        max_rows: if `data` has more rows than this, one random
            subsample of this size (seeded by `random_state`) is used
            for the whole scan. Keeps fitting fast at 50+ variables
            and/or 100k+ rows (SCOPE.md's stated scale target).
        test_size: fraction of each pair's rows held out for scoring.
            See the module docstring's "Held-out scoring" note.
        random_state: seeds the row subsample and every pair's model fit
            and train/test split, for reproducible scans.
        plot_grid_size: number of points in each pair's cached
            prediction curve (`PairPlotData.grid_x`).

    Returns:
        An `AssociationScan`. If neither `treatments` nor `outcomes` is
        given, `full_table` holds every ordered pair `(X, Y)` for `X !=
        Y` across all of `data`'s columns; otherwise `treatment_table`/
        `outcome_table` hold the scoped pairs.
    """
    if data.shape[1] < 2:
        raise ValueError("data must have at least 2 columns to scan pairwise associations")

    columns = list(data.columns)
    _validate_dtypes(data, columns)

    treatment_names = _validate_role_names("treatments", treatments or [], columns)
    outcome_names = _validate_role_names("outcomes", outcomes or [], columns)
    scoped = bool(treatment_names) or bool(outcome_names)

    df_sub = _subsample(data, max_rows=max_rows, random_state=random_state)

    target_columns = list(dict.fromkeys(treatment_names + outcome_names)) if scoped else columns
    target_kinds = {name: _target_kind(df_sub[name]) for name in target_columns}

    skipped: list[SkippedPair] = []
    plot_cache: dict[tuple[str, str], PairPlotData] = {}

    def run_pairs(pairs: list[tuple[str, str]]) -> list[PairResult]:
        results: list[PairResult] = []
        for predictor, target in pairs:
            target_kind = target_kinds[target]
            prepared = _prepare_pair(
                df_sub, predictor, target, target_kind, test_size, random_state
            )
            if isinstance(prepared, str):
                skipped.append(SkippedPair(predictor=predictor, target=target, reason=prepared))
                warnings.warn(
                    f"skipped {predictor!r} -> {target!r}: {prepared}",
                    AssociationSkippedWarning,
                    stacklevel=2,
                )
                continue
            x_all, y_all, x_train, x_test, y_train, y_test = prepared
            model = _fit_model(target_kind, random_state, x_train, y_train)
            score, score_name = _score_model(model, target_kind, x_test, y_test)
            results.append(
                PairResult(
                    predictor=predictor,
                    target=target,
                    score=score,
                    score_name=score_name,
                    n_used=len(x_all),
                )
            )
            plot_cache[(predictor, target)] = _build_plot_data(
                predictor, target, x_all, y_all, model, target_kind, plot_grid_size
            )
        results.sort(key=lambda r: (-r.score, r.predictor))
        return results

    if scoped:
        treatment_table = (
            run_pairs(_scoped_pairs(columns, treatment_names)) if treatment_names else []
        )
        outcome_table = run_pairs(_scoped_pairs(columns, outcome_names)) if outcome_names else []
        full_table: list[PairResult] = []
    else:
        full_table = run_pairs(_full_pairs(columns))
        treatment_table = []
        outcome_table = []

    return AssociationScan(
        scoped=scoped,
        treatment_table=treatment_table,
        outcome_table=outcome_table,
        full_table=full_table,
        plot_cache=plot_cache,
        skipped=skipped,
    )


# -- pair generation ----------------------------------------------------------


def _full_pairs(columns: Sequence[str]) -> list[tuple[str, str]]:
    """Every ordered pair `(X, Y)`, `X != Y`, across all columns: `n * (n - 1)`."""
    return list(itertools.permutations(columns, 2))


def _scoped_pairs(columns: Sequence[str], targets: Sequence[str]) -> list[tuple[str, str]]:
    """Every column vs every designated target, excluding self-pairs."""
    return [
        (predictor, target) for target in targets for predictor in columns if predictor != target
    ]


# -- validation -----------------------------------------------------------------


def _validate_dtypes(data: pd.DataFrame, columns: Sequence[str]) -> None:
    bad = [c for c in columns if data[c].dtype.kind not in _NUMERIC_DTYPE_KINDS]
    if bad:
        offenders = ", ".join(f"{c!r} ({data[c].dtype})" for c in bad)
        raise ColumnTypeError(
            "associate.py v1 only supports continuous and binary numeric columns "
            f"(SCOPE.md: categorical variables are out of scope for v1); non-numeric: {offenders}"
        )


def _validate_role_names(label: str, names: Sequence[str], columns: Sequence[str]) -> list[str]:
    unique = list(dict.fromkeys(names))
    missing = [n for n in unique if n not in columns]
    if missing:
        raise ValueError(f"{label} not found in data columns: {missing}")
    return unique


def _target_kind(series: pd.Series) -> TargetKind:
    n_unique = series.dropna().nunique()
    if n_unique < 2:
        raise ColumnTypeError(
            f"column {series.name!r} has fewer than 2 distinct values after subsampling; "
            "cannot be used as a scan target"
        )
    return "binary" if n_unique == 2 else "continuous"


# -- per-pair fitting -----------------------------------------------------------


def _subsample(data: pd.DataFrame, *, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(data) <= max_rows:
        return data.reset_index(drop=True)
    return data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _prepare_pair(
    df_sub: pd.DataFrame,
    predictor: str,
    target: str,
    target_kind: TargetKind,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | str:
    """Ready one pair's data for fitting, or return a skip reason.

    Returns `(x_all, y_all, x_train, x_test, y_train, y_test)` on
    success (all 1-D float arrays), or a human-readable reason string if
    the pair should be skipped.
    """
    pair = df_sub[[predictor, target]].dropna()
    if len(pair) < _MIN_PAIR_ROWS:
        return (
            f"only {len(pair)} rows remain after dropping missing values (need >= {_MIN_PAIR_ROWS})"
        )

    x_all = pair[predictor].to_numpy(dtype=float)
    y_all = pair[target].to_numpy(dtype=float)

    stratify = y_all if target_kind == "binary" else None
    if target_kind == "binary":
        counts = pd.Series(y_all).value_counts()
        if len(counts) < 2 or counts.min() < 2:
            return "binary target has fewer than 2 rows in one class after dropping missing values"

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if target_kind == "binary" and (len(set(y_train)) < 2 or len(set(y_test)) < 2):
        return "train/test split left only one class on one side"

    return x_all, y_all, x_train, x_test, y_train, y_test


def _fit_model(
    target_kind: TargetKind, random_state: int, x_train: np.ndarray, y_train: np.ndarray
) -> HistGradientBoostingRegressor | HistGradientBoostingClassifier:
    model: HistGradientBoostingRegressor | HistGradientBoostingClassifier
    if target_kind == "binary":
        model = HistGradientBoostingClassifier(random_state=random_state)
    else:
        model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def _score_model(
    model: HistGradientBoostingRegressor | HistGradientBoostingClassifier,
    target_kind: TargetKind,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, ScoreName]:
    x_test_2d = x_test.reshape(-1, 1)
    if target_kind == "binary":
        positive_class = model.classes_[1]
        y_test_indicator = (y_test == positive_class).astype(int)
        proba = model.predict_proba(x_test_2d)[:, 1]
        return float(roc_auc_score(y_test_indicator, proba)), "roc_auc"
    y_pred = model.predict(x_test_2d)
    return float(r2_score(y_test, y_pred)), "r2"


def _build_plot_data(
    predictor: str,
    target: str,
    x: np.ndarray,
    y: np.ndarray,
    model: HistGradientBoostingRegressor | HistGradientBoostingClassifier,
    target_kind: TargetKind,
    grid_size: int,
) -> PairPlotData:
    grid_x = np.linspace(x.min(), x.max(), grid_size)
    grid_2d = grid_x.reshape(-1, 1)
    if target_kind == "binary":
        grid_prediction = model.predict_proba(grid_2d)[:, 1]
    else:
        grid_prediction = model.predict(grid_2d)
    return PairPlotData(
        predictor=predictor,
        target=target,
        x=x.tolist(),
        y=y.tolist(),
        grid_x=grid_x.tolist(),
        grid_prediction=grid_prediction.tolist(),
    )
