"""Wrap `dowhy.gcm` for fitting, model evaluation, and intrinsic-influence attribution.

Builds a `gcm.StructuralCausalModel` off a `graph.DAGModel`, fits one
mechanism per node, and exposes thin wrappers around `dowhy`'s own
`falsify_graph`/`evaluate_causal_model` and `intrinsic_causal_influence`
-- SCOPE.md's "Causal attribution feature" Decided section chose to wrap
`dowhy.gcm` rather than hand-roll either, since both are real statistical
machinery (conditional independence testing, Shapley-value variance
decomposition, k-fold mechanism scoring) that is easy to get subtly wrong.
`evaluate_causal_model` (SCOPE.md build order step 3, "Causal model tab"
section) is meant to eventually replace this module's standalone
`falsify_causal_graph`/`FalsifyResult` wrapper around
`dowhy.gcm.falsify.falsify_graph` (Decided: retire it, since
`evaluate_causal_model` already reruns `falsify_graph` internally as part
of the same call) -- both still coexist here for now; see the
`falsify_causal_graph` docstring for why it hasn't actually been deleted
yet.

Pure logic, no UI dependency: testable standalone against
`demo_data.make_csat_demo_data`, same pattern `graph.py` and `associate.py`
followed. See SCOPE.md build order step 4.

Judgment calls made in this module, not directed by SCOPE.md
(flagging per PREFERENCES.md):

- **`AdditiveNoiseModel` + `HistGradientBoostingRegressor` for every
  node, binary included -- not `ClassifierFCM`/`HistGradientBoostingClassifier`
  for binary nodes.** Tried the classifier path first, since it is the more
  "correct" model family for a binary target. Two problems, both from how
  `dowhy.gcm` wraps sklearn rather than from sklearn itself: (1)
  `ClassifierFCM.fit` requires its target to satisfy `dowhy.gcm.util.general
  .is_categorical`, which only accepts `str`/`bool` values -- an int-coded
  0/1 column has to be cast to `bool` first; (2) once cast, that same
  `is_categorical` check fires again wherever the now-bool column is used
  as a *parent* feature elsewhere in the graph, and `SklearnRegressionModel
  .fit` silently one-hot-encodes any column it flags as categorical before
  handing it to the underlying regressor -- which changes the parent
  feature count out from under `monotonic_cst`, raising a shape mismatch
  on an unrelated node. Rather than threading dtype casts through the
  whole graph to dodge that, every node (binary or continuous) gets
  `AdditiveNoiseModel(SklearnRegressionModel(HistGradientBoostingRegressor(
  monotonic_cst=...)))` uniformly, keeping binary columns as plain 0/1
  numeric the whole way through. This also matches what SCOPE.md's Decided
  section literally names (`HistGradientBoostingRegressor`, monotonic
  constraints) without ever mentioning a classifier variant.
- **Categorical columns are not actually supported yet, despite
  SCOPE.md's Decided section naming `categorical_features="from_dtype"`
  as the mechanism.** That sklearn flag only ever sees a column if
  `dowhy.gcm.ml.SklearnRegressionModel.fit` lets it through unencoded --
  but that wrapper runs its own `is_categorical` check first (see above)
  and auto-one-hot/CatBoost-encodes any `str`/`bool`-valued column before
  the sklearn estimator ever sees it, which reintroduces the exact
  `monotonic_cst` shape-mismatch problem the binary-node decision above
  exists to avoid, this time for string-coded categories instead of
  bools. Correcting the plan rather than leaving it: this module restricts
  parent/target columns to continuous and (numeric-coded) binary for v1,
  same restriction `associate.py` already has and for the same reason --
  reusing its `ColumnTypeError` for the same class of error. A real
  string-valued categorical column would need this addressed (ordinal-
  encoding it first and treating it as numeric, or bypassing
  `SklearnRegressionModel`'s own encoder) before working correctly; no
  driver in the current CSAT demo data is categorical, so nothing asked
  for is blocked by leaving this for later.
- **`dag.graph.copy()` is handed to `StructuralCausalModel`, not
  `dag.graph` directly.** `gcm.StructuralCausalModel.__init__` only
  copies a graph that is already a dowhy-native `CausalModel`/
  `CausalGraph`; given a plain `networkx.DiGraph` it stores the object
  as-is. Handing over `dag.graph` unmodified would let `gcm.fit` (which
  can annotate node attributes) mutate the workshop's own live graph in
  place.
- **What "soft" monotonic constraints actually mean here.**
  Every non-root mechanism's `monotonic_cst` still
  *hard*-enforces the edge's asserted sign at the sklearn level -- there
  is no sklearn option to make that constraint advisory, and dropping it
  would defeat the point of asserting a sign at all. "Soft" instead means
  this module does not *refuse to fit* when the raw data disagrees with
  an asserted sign (a "hard" policy would validate first and raise).
  `fit_causal_model` fits under the constraint regardless, and separately
  reports `sign_disagreements`: for every edge, the Spearman rank
  correlation between the raw (unfitted, marginal) parent/child columns,
  flagged when its sign opposes the asserted one by at least
  `min_correlation`. This is a *marginal* correlation, not the partial/
  conditional effect the asserted sign is actually meant to describe, so
  it can disagree even when the true direct effect matches (confounding
  through a shared ancestor) or agree even when it doesn't (through an
  unblocked path via another parent) -- a cheap heuristic flag for the
  PM/DS to look at, not a statistical test of the constraint itself.
- **`min_correlation=0.1` default for flagging a sign disagreement.**
  Needed some floor -- two genuinely unrelated columns still produce a
  small nonzero sample correlation of essentially random sign, and
  flagging every one of those would bury the disagreements worth a
  second look. Exposed as a parameter rather than fixed, since there's no
  principled value here, just a threshold separating obvious sampling
  noise from a real disagreement worth surfacing.
- **`n_jobs` is a pass-through knob, not a production default -- and
  goes through `dowhy.gcm.config`'s global default, not a per-call
  argument.** `gcm.intrinsic_causal_influence`'s default Shapley
  estimation, and `gcm.falsify.falsify_graph`'s internal kernel-based
  independence test, both use joblib/loky multiprocessing, which failed
  in the sandboxed bridge shell used for development
  (`BrokenProcessPool` / `OSError: Too many open files`) -- a property
  of that specific sandboxed dev environment, not necessarily the
  machine this runs on in production. Tried
  passing `n_jobs` straight through to `falsify_graph`'s own `n_jobs`
  argument first; that only covers one of its two internal parallel
  calls -- the kernel-based test's own bootstrap resampling
  (`kernel_based`'s `bootstrap_n_jobs`) isn't reachable through
  `falsify_graph`'s signature at all. Both `gcm.ShapleyConfig.n_jobs`
  and that bootstrap default fall back to the same place,
  `dowhy.gcm.config.default_n_jobs`, when not given explicitly, so
  `attribute_target`/`falsify_causal_graph` reach every internal joblib
  call at once by temporarily overriding that global for the duration of
  the call (`_n_jobs_override`) instead of threading `n_jobs` through
  each function's own differently-shaped parameter. Left unset (`dowhy`'s
  own default, parallel, `default_n_jobs = -1`) unless a caller passes
  one; pass `n_jobs=1` to force sequential execution if the same error
  shows up elsewhere.
- **Sample-size knobs (`num_training_samples`, `num_samples_randomization`,
  `num_samples_baseline`) are left at `dowhy`'s own defaults** (100000 /
  250 / 1000) unless a caller overrides them. The working prototype for
  this module used much smaller values to iterate quickly against 2000
  demo rows; shipping those as the real defaults would quietly weaken the
  Shapley estimate for everyone. Callers on a real (larger, slower)
  dataset can still lower them explicitly.
- **Attribution results are ranked by signed contribution, descending,
  not by absolute value.** `intrinsic_causal_influence` returns variance-
  based contributions, which are theoretically non-negative; small
  negative values that do turn up are Monte Carlo estimation noise around
  a true contribution near zero, not a real "negative influence." Ranking
  by raw value puts those where they belong, at the bottom, without
  needing to explain a separate abs-value ranking rule in the UI later.
- **A joint `dropna()` across every DAG column, once, before `gcm.fit`.**
  `associate.py` drops rows per-pair, since each pair is fit
  independently. Here every node's mechanism is fit as part of one shared
  model over the same rows, so the drop has to be joint (one row missing
  any DAG column is dropped from the whole fit) rather than per-node.
- **`attribute_target`'s `random_state` (correction, found via a flaky
  CI run on Python 3.13).** `gcm.intrinsic_causal_influence`
  draws its baseline/randomization samples and its Shapley subset/
  permutation sampling from `numpy`'s *global* `np.random` state, not a
  seeded local generator -- confirmed by reading `dowhy.gcm.shapley`/
  `dowhy.gcm.influence`'s source, which call `np.random.choice`/
  `np.random.randint` directly throughout. With `n_jobs=1` (sequential,
  no worker races over that shared state) this makes a given call fully
  reproducible for a fixed seed, but *without* one it is not reproducible
  at all -- two calls back to back, or the same test on two Python
  versions, can rank ancestors differently whenever their true
  contributions are close. `attribute_target` now takes an optional
  `random_state` that seeds `numpy.random` for the duration of the call
  and restores whatever state was there before (`_random_state_override`,
  same pattern as `_n_jobs_override` above). Left `None` (unseeded, matches
  prior behavior) unless a caller passes one -- `server.py`'s
  `/api/causal/attribute/{target_node}` route now passes the same
  `random_state` already used for `fit_causal_model`, so a workshop
  clicking "Show drivers" twice for the same target sees a stable
  ranking rather than one that can shuffle between clicks.
- **`noise_models` validation: unknown node is `KeyError`, a node with
  parents is `ValueError`, not a silent no-op.** SCOPE.md's build order
  names the parameter's shape (`dict[str, Literal["empirical",
  "gaussian"]] | None`) but not what to do with a bad key. Silently
  ignoring an unknown or non-root node would let a typo'd or stale
  noise choice (e.g. after a node is renamed or a root gains a parent)
  pass unnoticed with no effect, the same class of bug
  `_validate_columns` already guards against for `data`'s own columns.
  Reuses `KeyError`/`ValueError` rather than a new exception type, since
  both already mean the same thing elsewhere in this module
  (`attribute_target`'s unknown-node `KeyError`, `_validate_columns`'s
  dtype `ColumnTypeError`).
- **`falsify_causal_graph`/`FalsifyResult` are not actually deleted in
  the same change that adds `evaluate_causal_model`, despite SCOPE.md's
  Decided section calling for their retirement.** Discovered while
  implementing build order step 3: `server.py`'s `POST
  /api/causal/build` (step 5, not yet done) still imports and calls
  `falsify_causal_graph` directly and serializes `FalsifyResult` into
  its response. Deleting both now would leave `server.py` (and its
  tests) broken between this step and step 5, for no benefit -- the
  retirement decision stands, but the actual removal is deferred to
  step 5, done in the same change that migrates `server.py` off it.
- **`EvaluateCausalModelConfig`'s own `n_jobs` is resolved once, at
  construction time, not lazily when `evaluate_causal_model` is
  called.** `EvaluateCausalModelConfig.__init__` runs `n_jobs =
  config.default_n_jobs if n_jobs is None else n_jobs` immediately --
  building the config before entering `_n_jobs_override`'s `with`
  block bakes in whatever the *unmodified* global default was, silently
  ignoring the override (confirmed directly: reproduced the same
  `BrokenProcessPool` the module docstring's other `n_jobs` note
  describes, by constructing the config outside the override first).
  `evaluate_causal_model` (this module's function) builds
  `EvaluateCausalModelConfig` *inside* the `_n_jobs_override` block for
  this reason, not before it.
- **`MechanismPerformance` drops `f1` and every baseline-comparison
  field from `dowhy`'s own `MechanismPerformanceResult`, rather than
  keeping them as always-`None`/always-empty passthroughs.** `f1` never
  fires for this repo's node types regardless of which function
  computes it (session 17's finding: numeric 0/1-coded binary nodes
  always take the `r2` branch -- see the AUC-vs-F1 note above), and the
  baseline fields are always empty too since `evaluate_causal_model` is
  called with its own `compare_mechanism_baselines` default (`False`,
  not asked for in SCOPE.md). Keeping either as dead-but-present fields
  would misrepresent them as live data the frontend might reasonably
  render.
- **`ModelEvaluation.graph_falsification` is `dowhy`'s own
  `EvaluationResult` object, unwrapped, not reshaped into a
  dagshop-specific type the way `FalsifyResult` was.** A second wrapper
  exposing the same `falsified`/`falsifiable`/`significance_level`
  fields `FalsifyResult` already exposes would be pure duplication now
  that there is nothing else for such a wrapper to add (see SCOPE.md's
  Decided section on retiring `FalsifyResult`).
- **`build_node_plots` uses one joint train/test split shared across
  every node, not a per-node split the way `associate.py` does.**
  Matches `fit_causal_model`'s own joint (not per-node) `dropna`: every
  node here is part of one shared model over the same rows, so "held
  out" has to mean the same held-out rows for every node. The
  trade-off: a single joint split can't be stratified per node the way
  `associate.py`'s per-pair splits are, so a binary-coded node's test
  fold can (rarely, at this repo's demo-data scale) land only one
  class -- `ActualVsPredictedPlot.auc` is `None` in that case rather
  than raising, same "degenerate split -> skip, don't crash" instinct
  `associate.py._prepare_pair` already applies to its own per-pair
  splits.
- **`_node_prediction_model` is factored out of `fit_causal_model`'s
  own mechanism-construction loop, shared with `build_node_plots`.**
  Both need the exact same `HistGradientBoostingRegressor`/
  `monotonic_cst` construction -- `fit_causal_model` wraps it in
  `gcm.AdditiveNoiseModel` for the real fit, `build_node_plots` fits it
  directly (no `gcm` wrapping) against a train-only split, since it
  only needs point predictions, not causal sampling. Sharing this one
  function keeps the two from silently drifting apart on a detail this
  module already treats as easy to get subtly wrong (see the
  alphabetical-parent-order note above).
- **`ActualVsPredictedPlot`/`ObservedVsSampledPlot` carry no `kind`
  discriminant field.** A future `GET /api/causal/plot/{node}`
  endpoint (SCOPE.md build order step 5) returns one or the other
  depending on whether `node` is a root; matches this repo's existing
  pattern of letting response shape itself say what it is
  (`associate.py`'s `PairPlotData` vs `SkippedPair` has no discriminant
  either) rather than adding one pre-emptively. Worth revisiting in
  step 5 if the frontend finds branching on field presence awkward.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm import config as gcm_config
from dowhy.gcm.falsify import EvaluationResult, falsify_graph
from dowhy.gcm.model_evaluation import EvaluateCausalModelConfig
from dowhy.graph import get_ordered_predecessors
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from dagshop.associate import ColumnTypeError
from dagshop.graph import DAGModel, Sign

# Same restriction as associate.py, and for the same reason (v1: no
# categorical support -- see module docstring). Redefined here rather than
# importing associate.py's private `_NUMERIC_DTYPE_KINDS`, since that name
# is module-internal there; the two modules happen to want an identical
# check, not a shared one.
_NUMERIC_DTYPE_KINDS = frozenset("biuf")

NoiseModel = Literal["empirical", "gaussian"]
"""Root-node noise distribution choice for `fit_causal_model`'s `noise_models`.

`"empirical"` (default) resamples directly from the observed column
(`gcm.EmpiricalDistribution`), matching today's unconditional behavior.
`"gaussian"` fits a parametric Normal instead
(`gcm.ScipyDistribution(scipy.stats.norm)`). See SCOPE.md's "Causal model
tab" Decided section: per-node, not a workshop-wide default, and only
these two options for v1 (no mixture distribution yet).
"""

_ROOT_NOISE_FACTORIES: dict[NoiseModel, Callable[[], Any]] = {
    "empirical": gcm.EmpiricalDistribution,
    "gaussian": lambda: gcm.ScipyDistribution(norm),
}


@contextlib.contextmanager
def _n_jobs_override(n_jobs: int | None):
    """Temporarily override `dowhy`'s global default parallelism, if given.

    See the module docstring's `n_jobs` note: both `gcm.ShapleyConfig`
    and `falsify_graph`'s internal kernel-based independence test fall
    back to `dowhy.gcm.config.default_n_jobs` when not given a value
    directly, and that internal test's own bootstrap parallelism isn't
    reachable any other way -- overriding this one global for the
    duration of a call is the only mechanism that reaches every internal
    joblib call `attribute_target`/`falsify_causal_graph` can trigger.
    A no-op (nothing saved or restored) when `n_jobs` is `None`.
    """
    if n_jobs is None:
        yield
        return
    previous = gcm_config.default_n_jobs
    gcm_config.set_default_n_jobs(n_jobs)
    try:
        yield
    finally:
        gcm_config.set_default_n_jobs(previous)


@contextlib.contextmanager
def _random_state_override(random_state: int | None):
    """Temporarily seed numpy's global RNG, if given, and restore it after.

    See the module docstring's `random_state` note: `dowhy.gcm`'s Shapley/
    influence code draws from `numpy`'s global `np.random` state directly,
    not a seeded local generator, so this is the only way to make a call
    reproducible without patching `dowhy` itself. Saving/restoring the
    prior state (rather than just calling `np.random.seed`) keeps this
    from leaking into whatever the caller does next, the same reasoning
    `_n_jobs_override` above already applies to the parallelism global. A
    no-op when `random_state` is `None`.
    """
    if random_state is None:
        yield
        return
    previous_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        yield
    finally:
        np.random.set_state(previous_state)


@dataclass(frozen=True)
class SignDisagreement:
    """One edge where the raw data's marginal correlation opposes its asserted sign.

    See the module docstring's "soft monotonic constraints" note: this is
    a cheap, marginal-correlation heuristic, not a check of the fitted
    model or of the true partial/conditional effect.
    """

    parent: str
    child: str
    asserted_sign: Sign
    correlation: float


@dataclass(frozen=True)
class AttributionResult:
    """One ranking-table row: `node`'s intrinsic contribution to the attribution target.

    `share` is `contribution` divided by the sum of every row's
    `contribution` for this call -- by the Shapley efficiency property
    that sum equals the target's total variance, so `share` reads as
    "this node's percentage of the target's variance," summing to ~1.0
    (modulo float error) across the whole result list. That list
    includes a row for the target node itself (its own unexplained
    noise, not a driver) -- callers ranking or displaying "drivers"
    should treat `node == target_node` as a residual bucket, not a
    driver, rather than dropping it (dropping it would make the
    remaining shares no longer sum to 1).
    """

    node: str
    contribution: float
    share: float


@dataclass(frozen=True)
class FalsifyResult:
    """Result of `falsify_causal_graph`, mirroring `dowhy`'s own `EvaluationResult`.

    `falsified`/`falsifiable` are `None` when the permutation test
    couldn't be evaluated at all (`dowhy`'s own `can_evaluate` case, e.g.
    too few nodes to permute meaningfully) rather than a `False`/`False`
    "the graph is fine" reading -- callers should treat `None` as
    "inconclusive," not as a pass.
    """

    falsified: bool | None
    falsifiable: bool | None
    significance_level: float
    report: str


@dataclass(frozen=True)
class MechanismPerformance:
    """One node's per-mechanism evaluation from `dowhy.gcm.evaluate_causal_model`.

    Reshaped down from `dowhy`'s own `MechanismPerformanceResult` to the
    fields SCOPE.md's "Causal model tab" Decided section actually asks
    for: `f1` and the baseline-model-comparison fields are dropped
    entirely, not just left `None` (see module docstring for why).

    `crps` is populated for every node. `kl_divergence` is populated
    only for a root node (`is_root=True`); `mse`/`nmse`/`r2` only for a
    non-root one -- the unpopulated field for a given node is `None`,
    not omitted, so callers can branch on `is_root` rather than probing
    which fields happen to be set.
    """

    node: str
    is_root: bool
    crps: float | None
    kl_divergence: float | None
    mse: float | None
    nmse: float | None
    r2: float | None


@dataclass(frozen=True)
class ModelEvaluation:
    """Result of `evaluate_causal_model`, meant to eventually replace
    `falsify_causal_graph`'s standalone call (see module docstring: not
    yet done, `server.py` still depends on the old path).

    `graph_falsification` is `dowhy`'s own `EvaluationResult` object,
    unwrapped -- see module docstring for why this isn't reshaped into a
    dagshop-specific type the way `FalsifyResult` was. `report` already
    covers a plain-text summary of everything in this result (mechanism
    performances, overall KL divergence, and the graph falsification
    together), same pattern `FalsifyResult.report` used for the plain
    `falsify_graph` call.
    """

    mechanism_performances: dict[str, MechanismPerformance]
    overall_kl_divergence: float
    graph_falsification: EvaluationResult
    report: str


@dataclass(frozen=True)
class ActualVsPredictedPlot:
    """Held-out actual-vs-predicted plot data for one non-root node.

    `actual`/`predicted` are the test-fold's real target values and
    this node's own regressor's prediction for each row, same order --
    paired for a scatter against a y=x reference line (SCOPE.md's
    Decided section: a new plot type, not a reuse of `associate.py`'s
    `PairPlotData`, since a node can have more than one parent).

    `auc` is `roc_auc_score(actual, predicted)` -- the regressor's raw
    continuous output fed directly as the score, no thresholding, no
    probability calibration (matching the "every node is a regressor"
    decision, see module docstring). Populated only for a binary-coded
    (0/1 numeric, exactly 2 distinct values) node whose test fold
    actually landed both classes; `None` for a continuous node, or for
    a binary one where it didn't (see `build_node_plots`'s docstring:
    one joint split, not stratified per node).
    """

    node: str
    actual: list[float]
    predicted: list[float]
    auc: float | None


@dataclass(frozen=True)
class ObservedVsSampledPlot:
    """Observed-vs-sampled plot data for one root node's fitted noise distribution.

    `observed` is this node's own (joint-`dropna`'d) column. `sampled`
    is a fresh draw of the same size from `FittedCausalModel.scm`'s
    already-fitted noise distribution for this node
    (`gcm.EmpiricalDistribution` or `gcm.ScipyDistribution`, whichever
    `noise_models` picked -- see `fit_causal_model`). The two are NOT
    row-aligned: a root node's noise distribution is unconditional, so
    there is no per-row "prediction" to pair against, only two
    independent samples that should look like the same distribution if
    the noise choice fits well (SCOPE.md's Decided section: this
    doubles as a sanity check on the noise dropdown's choice).
    """

    node: str
    observed: list[float]
    sampled: list[float]


@dataclass
class FittedCausalModel:
    """A `gcm.StructuralCausalModel` fitted against `dag`/`data`, plus diagnostics."""

    scm: gcm.StructuralCausalModel
    dag: DAGModel
    sign_disagreements: list[SignDisagreement]


def _node_prediction_model(
    graph: nx.DiGraph, node: str, parents: list[str], random_state: int
) -> HistGradientBoostingRegressor:
    """Build one non-root node's regressor: `HistGradientBoostingRegressor`
    with `monotonic_cst` built from `graph`'s own asserted edge signs, in
    `parents`' order (already `dowhy`-ordered by the caller -- see
    `fit_causal_model`'s docstring for why order matters here).

    Reads sign directly off `graph`'s edge data (`graph.edges[parent,
    node]["sign"]`), not `DAGModel.sign(...)`, so this gives the same
    answer whether `graph` is a live `DAGModel.graph` (as in
    `fit_causal_model`) or an already-fitted `FittedCausalModel.scm.graph`
    that the original `DAGModel` may have since diverged from (as in
    `build_node_plots` -- this repo does not auto-invalidate a build's
    cache on a later graph edit, see server.py's module docstring and
    this module's own `_validate_noise_models` note).

    Shared by `fit_causal_model` (wrapped in `gcm.AdditiveNoiseModel`,
    fit against the whole SCM) and `build_node_plots` (fit directly, no
    `gcm` wrapping, against a train-only split -- see that function's
    docstring for why it needs its own separate fit rather than reusing
    the SCM's already-fitted mechanism).
    """
    monotonic_cst = [1 if graph.edges[parent, node]["sign"] == "+" else -1 for parent in parents]
    return HistGradientBoostingRegressor(
        monotonic_cst=monotonic_cst,
        categorical_features="from_dtype",
        random_state=random_state,
    )


def fit_causal_model(
    dag: DAGModel,
    data: pd.DataFrame,
    *,
    random_state: int = 0,
    min_correlation: float = 0.1,
    noise_models: dict[str, NoiseModel] | None = None,
) -> FittedCausalModel:
    """Fit one `gcm` mechanism per node and return the fitted model.

    `dag` must already be acyclic (`dag.validate()` is called here, and
    raises `graph.GraphValidationError` if not) and every one of its
    nodes must be a continuous or (numeric-coded) binary column in
    `data` (raises `KeyError` for a missing column, `ColumnTypeError`
    for an unsupported dtype -- see module docstring). Rows with a
    missing value in any DAG column are dropped before fitting (a joint
    `dropna`, not per-node).

    Root nodes (no parents) get a noise distribution picked by
    `noise_models` (a `dict[str, NoiseModel]` mapping node name to
    `"empirical"` or `"gaussian"`), defaulting to `"empirical"`
    (`gcm.EmpiricalDistribution()`, today's unconditional behavior) for
    any root not named in the mapping. `"gaussian"` fits a parametric
    Normal instead (`gcm.ScipyDistribution(scipy.stats.norm)`). Raises
    `KeyError` if `noise_models` names a node not in `dag`, or
    `ValueError` if it names a node that has parents (a noise-
    distribution choice only makes sense for a root node -- see module
    docstring). Every other node gets `gcm.AdditiveNoiseModel` over a
    `HistGradientBoostingRegressor` whose `monotonic_cst` is built from
    the node's *asserted* parent signs, in `dowhy`'s own alphabetical
    parent order (`dowhy.graph.get_ordered_predecessors` -- every
    internal `dowhy.gcm` fitting/sampling call assembles a node's parent
    feature matrix in that order, not edge-insertion order, so building
    `monotonic_cst` any other way silently misaligns which constraint
    applies to which feature). See the module docstring for why every
    node uses the regressor, not a classifier, and what "soft"
    constraints mean here.
    """
    dag.validate()
    _validate_columns(dag, data)
    _validate_noise_models(dag, noise_models)

    graph_copy = dag.graph.copy()
    scm = gcm.StructuralCausalModel(graph_copy)

    for node in graph_copy.nodes:
        parents = get_ordered_predecessors(graph_copy, node)
        if not parents:
            choice: NoiseModel = (noise_models or {}).get(node, "empirical")
            scm.set_causal_mechanism(node, _ROOT_NOISE_FACTORIES[choice]())
            continue
        prediction_model = _node_prediction_model(graph_copy, node, parents, random_state)
        scm.set_causal_mechanism(
            node, gcm.AdditiveNoiseModel(gcm.ml.SklearnRegressionModel(prediction_model))
        )

    fit_data = data[dag.nodes].dropna()
    gcm.fit(scm, fit_data)

    disagreements = _sign_disagreements(dag, data, min_correlation=min_correlation)

    return FittedCausalModel(scm=scm, dag=dag, sign_disagreements=disagreements)


def attribute_target(
    fitted: FittedCausalModel,
    target_node: str,
    *,
    num_training_samples: int | None = None,
    num_samples_randomization: int | None = None,
    num_samples_baseline: int | None = None,
    n_jobs: int | None = None,
    random_state: int | None = None,
) -> list[AttributionResult]:
    """Rank every ancestor of `target_node` by its intrinsic causal influence.

    Thin wrapper around `gcm.intrinsic_causal_influence`: any ancestor,
    not just direct parents, gets a Shapley-based variance contribution
    -- the point of this feature over a plain direct-parent ranking (see
    SCOPE.md's `make_csat_demo_data` docstring for why the demo data is
    deliberately multi-hop). Results are sorted by contribution,
    descending (see module docstring for why raw value, not absolute
    value). Each result's `share` is its contribution as a fraction of
    the total across this same call, including the target node's own
    row -- see `AttributionResult`'s docstring for why that row stays in
    rather than getting filtered out.

    `num_training_samples`/`num_samples_randomization`/
    `num_samples_baseline` are left at `dowhy`'s own defaults unless
    given explicitly. `n_jobs` is left unset (`dowhy`'s own default,
    parallel) unless given explicitly -- see module docstring.
    `random_state`, if given, seeds `numpy`'s global RNG for the
    duration of the call (`_random_state_override`); left `None`
    (unseeded) by default. Only actually guarantees a reproducible
    ranking under sequential execution (`n_jobs=1`, or a build small
    enough that `dowhy`'s own default parallelism never kicks in) -- see
    module docstring's `random_state` note. Under real parallelism,
    worker processes can still pull from the shared, seeded state in a
    different order run to run, so `random_state` alone does not
    guarantee reproducibility there; pass `n_jobs=1` too if that
    matters.
    """
    if target_node not in fitted.dag.nodes:
        raise KeyError(f"no node {target_node!r}")

    kwargs: dict[str, Any] = {}
    if num_training_samples is not None:
        kwargs["num_training_samples"] = num_training_samples
    if num_samples_randomization is not None:
        kwargs["num_samples_randomization"] = num_samples_randomization
    if num_samples_baseline is not None:
        kwargs["num_samples_baseline"] = num_samples_baseline

    with _n_jobs_override(n_jobs), _random_state_override(random_state):
        contributions = gcm.intrinsic_causal_influence(
            fitted.scm, target_node=target_node, **kwargs
        )
    ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    # Sum of raw contributions, not the target's actual sample variance:
    # the Shapley efficiency property says they're equal in theory, but
    # dividing by the sum computed here keeps `share` self-consistent
    # with the numbers in this same result list (summing to exactly 1,
    # modulo float error) even if Monte Carlo noise makes the two differ
    # slightly in practice. A near-zero total (e.g. a constant target
    # with no real variance to attribute) would make `share` a division
    # by ~0 blow up into meaningless noise -- 0.0 for every row is the
    # honest answer there, not a NaN or an arbitrarily large ratio.
    total = sum(value for _, value in ranked)
    return [
        AttributionResult(
            node=node,
            contribution=float(value),
            share=float(value / total) if total else 0.0,
        )
        for node, value in ranked
    ]


def falsify_causal_graph(
    fitted: FittedCausalModel,
    data: pd.DataFrame,
    *,
    significance_level: float = 0.05,
    n_jobs: int | None = None,
) -> FalsifyResult:
    """Run `dowhy`'s node-permutation graph refutation test against `data`.

    Thin wrapper around `dowhy.gcm.falsify.falsify_graph`, run against
    the same (copied) graph `fit_causal_model` actually fit, not `dag`
    directly. `n_jobs` is left unset (`dowhy`'s own default) unless
    given explicitly -- see module docstring.

    Superseded by `evaluate_causal_model` below (SCOPE.md's "Causal
    model tab" Decided section: retire this once `server.py` migrates
    off it), but still called directly by `server.py`'s `POST
    /api/causal/build` today -- not yet deleted for that reason, see
    module docstring.
    """
    eval_data = data[fitted.dag.nodes].dropna()
    with _n_jobs_override(n_jobs):
        evaluation = falsify_graph(
            fitted.scm.graph, eval_data, significance_level=significance_level
        )
    return FalsifyResult(
        falsified=evaluation.falsified,
        falsifiable=evaluation.falsifiable,
        significance_level=evaluation.significance_level,
        report=str(evaluation),
    )


def evaluate_causal_model(
    fitted: FittedCausalModel,
    data: pd.DataFrame,
    *,
    significance_level: float = 0.05,
    n_jobs: int | None = None,
) -> ModelEvaluation:
    """Run `dowhy`'s `evaluate_causal_model` against the already-fitted SCM.

    Thin wrapper around `dowhy.gcm.evaluate_causal_model`, run against
    the same (copied, already-fitted) `fitted.scm`, not a fresh fit --
    every non-root node's per-mechanism performance (`crps`/`mse`/
    `nmse`/`r2`) comes from `dowhy`'s own internal 5-fold cross-
    validation (`EvaluateCausalModelConfig.mechanism_evaluation_kfolds`,
    left at its default of 5), re-fitting fresh copies of each node's
    mechanism per fold -- not the single train/test split SCOPE.md's
    Decided section describes for the AUC/actual-vs-predicted-plot work
    (build order step 4, not this function). Root nodes get
    `kl_divergence` instead (also via 5-fold CV, comparing the fitted
    noise distribution's own samples against held-out actual values).

    `significance_level` defaults to `0.05`, matching
    `falsify_causal_graph`'s own default -- `dowhy`'s own
    `EvaluateCausalModelConfig` defaults `falsify_graph_significance_level`
    to `0.2` (see module docstring). `n_jobs` behaves like every other
    `n_jobs` parameter in this module: left unset (`dowhy`'s own
    default) unless given explicitly, via the same global
    `_n_jobs_override` mechanism -- see module docstring for why
    `EvaluateCausalModelConfig` has to be built *inside* that override,
    not before it.
    """
    eval_data = data[fitted.dag.nodes].dropna()
    with _n_jobs_override(n_jobs):
        config = EvaluateCausalModelConfig(falsify_graph_significance_level=significance_level)
        raw = gcm.evaluate_causal_model(fitted.scm, eval_data, config=config)

    mechanism_performances = {
        node: MechanismPerformance(
            node=node,
            is_root=perf.is_root,
            crps=perf.crps,
            kl_divergence=perf.kl_divergence,
            mse=perf.mse,
            nmse=perf.nmse,
            r2=perf.r2,
        )
        for node, perf in raw.mechanism_performances.items()
    }

    # Must be set before `str(raw)`: `CausalModelEvaluationResult.__str__`
    # calls `plot_evaluation_results` (matplotlib) as a side effect
    # whenever this flag is left at its own default of `True` -- see
    # module docstring.
    raw.plot_falsification_histogram = False
    report = str(raw)

    return ModelEvaluation(
        mechanism_performances=mechanism_performances,
        overall_kl_divergence=float(raw.overall_kl_divergence),
        graph_falsification=raw.graph_falsification,
        report=report,
    )


def build_node_plots(
    fitted: FittedCausalModel,
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
) -> dict[str, ActualVsPredictedPlot | ObservedVsSampledPlot]:
    """Build the actual-vs-predicted-popup plot data for every node.

    SCOPE.md build order step 4: one joint train/test split of
    `data[fitted.dag.nodes].dropna()` (same `test_size`/`random_state`
    convention as `associate.py`, single split, not k-fold -- one split
    shared across every node, matching `fit_causal_model`'s own joint
    (not per-node) `dropna`, rather than `associate.py`'s per-pair
    splits: every node's mechanism here is part of one shared model over
    the same rows, so "held out" has to mean the same held-out rows for
    every node, not a different split per node).

    For a non-root node, this refits a *fresh* regressor on the train
    fold only (`_node_prediction_model`, same construction
    `fit_causal_model` uses) and predicts on the test fold --
    deliberately not reusing `fitted.scm`'s already-fitted mechanism
    (which was fit on all rows): scoring a model against rows it was
    already fit on would be in-sample, exactly what SCOPE.md's Decided
    section rejected for this feature. This is a different (single-
    split) held-out estimate than `evaluate_causal_model`'s own 5-fold
    CV numbers (step 3) -- SCOPE.md's Decided section accepts the two
    looking slightly inconsistent for the same node rather than
    discarding the better k-fold estimate to force them to match.

    A root node gets no split-based number at all: `evaluate_causal_
    model`'s `kl_divergence` (step 3) already covers it. Its plot data
    is an `ObservedVsSampledPlot` instead, reusing `fitted.scm`'s
    already-fitted noise distribution directly (no train/test split --
    a root node's distribution is unconditional, so there is nothing
    to hold rows out *from*).

    Node set and per-node parents/signs are read from `fitted.dag.nodes`
    and `fitted.scm.graph` respectively, not re-derived from `data` --
    same precedent `attribute_target`/`falsify_causal_graph` already
    follow, so this reflects what was actually fit even if `fitted.dag`
    has since been edited (this repo does not auto-invalidate a build's
    cache on a later graph edit).
    """
    eval_data = data[fitted.dag.nodes].dropna()
    train_data, test_data = train_test_split(
        eval_data, test_size=test_size, random_state=random_state
    )

    plots: dict[str, ActualVsPredictedPlot | ObservedVsSampledPlot] = {}
    for node in fitted.dag.nodes:
        parents = get_ordered_predecessors(fitted.scm.graph, node)
        if not parents:
            mechanism = fitted.scm.causal_mechanism(node)
            observed = eval_data[node].to_numpy(dtype=float)
            sampled = np.asarray(mechanism.draw_samples(len(observed)), dtype=float).reshape(-1)
            plots[node] = ObservedVsSampledPlot(
                node=node,
                observed=observed.tolist(),
                sampled=sampled.tolist(),
            )
            continue

        x_train = train_data[parents].to_numpy(dtype=float)
        y_train = train_data[node].to_numpy(dtype=float)
        x_test = test_data[parents].to_numpy(dtype=float)
        y_test = test_data[node].to_numpy(dtype=float)

        regressor = _node_prediction_model(fitted.scm.graph, node, parents, random_state)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)

        auc = None
        if _is_binary_coded(eval_data[node]) and len(set(y_test)) == 2:
            auc = float(roc_auc_score(y_test, y_pred))

        plots[node] = ActualVsPredictedPlot(
            node=node,
            actual=y_test.tolist(),
            predicted=y_pred.tolist(),
            auc=auc,
        )

    return plots


def _is_binary_coded(series: pd.Series) -> bool:
    """Same binary detection `associate.py`'s `_target_kind` uses
    (`nunique() == 2` after dropping missing values). Redefined here
    rather than imported, same reason `_NUMERIC_DTYPE_KINDS` is
    redefined above: it's module-private there, and the two modules
    just happen to want an identical check, not a shared one.
    """
    return series.dropna().nunique() == 2


def _validate_noise_models(dag: DAGModel, noise_models: dict[str, NoiseModel] | None) -> None:
    if not noise_models:
        return
    unknown = [n for n in noise_models if n not in dag.nodes]
    if unknown:
        raise KeyError(f"noise_models references unknown node(s): {sorted(unknown)}")
    non_root = [n for n in noise_models if list(dag.graph.predecessors(n))]
    if non_root:
        raise ValueError(
            "noise_models only applies to root nodes (no parents); "
            f"got node(s) with parents: {sorted(non_root)}"
        )


def _validate_columns(dag: DAGModel, data: pd.DataFrame) -> None:
    missing = [n for n in dag.nodes if n not in data.columns]
    if missing:
        raise KeyError(f"data is missing column(s) required by the DAG: {sorted(missing)}")
    bad = [n for n in dag.nodes if data[n].dtype.kind not in _NUMERIC_DTYPE_KINDS]
    if bad:
        raise ColumnTypeError(
            "causal_model.py only supports continuous/binary numeric columns "
            f"for v1 (see module docstring): {sorted(bad)}"
        )


def _sign_disagreements(
    dag: DAGModel, data: pd.DataFrame, *, min_correlation: float
) -> list[SignDisagreement]:
    disagreements: list[SignDisagreement] = []
    for parent, child in dag.edges:
        pair = data[[parent, child]].dropna()
        if len(pair) < 2:
            continue
        correlation, _p_value = spearmanr(pair[parent], pair[child])
        if np.isnan(correlation) or correlation == 0:
            continue
        expected_sign = 1 if dag.sign(parent, child) == "+" else -1
        observed_sign = 1 if correlation > 0 else -1
        if observed_sign != expected_sign and abs(correlation) >= min_correlation:
            disagreements.append(
                SignDisagreement(
                    parent=parent,
                    child=child,
                    asserted_sign=dag.sign(parent, child),
                    correlation=float(correlation),
                )
            )
    disagreements.sort(key=lambda d: abs(d.correlation), reverse=True)
    return disagreements
