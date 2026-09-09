# DAGshop

[![CI](https://github.com/raz1470/dagshop/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/dagshop/actions/workflows/ci.yml)

Interactive DAG-drawing workshop tool for PM and data scientist pairs.
One DS and one PM sit down with a dataset and hypothesize a causal
structure together — visually, on a canvas, grounded in what the data
actually looks like rather than intuition alone. DAGshop's job ends at
a clean, well-formed DAG file; the causal modeling itself is left to
[`dowhy.gcm`](https://www.pywhy.org/dowhy/), which the exported DAG is
built to feed directly.

Runs entirely on one machine via `localhost` — no networked or
multi-device mode, no external calls, no CDN-loaded frontend assets.

## Install

```bash
uv sync
```

## Usage

DAGshop has two subcommands.

### `dagshop launch` — run the workshop

```bash
dagshop launch data.csv --treatment treatment_col --outcome outcome_col
```

`data.csv` must contain only continuous or binary numeric columns — no
categoricals in v1. `--treatment`/`--outcome` are optional and
repeatable (`--treatment X --treatment Y`); designating at least one of
either scopes the pre-work association scan to "what's associated with
treatment(s), what's associated with outcome(s), and how do the
remaining covariates associate with each other," which is much faster
than a full pairwise scan at 50+ variables. Leave both off for a fully
exploratory session — every column gets scanned against every other.

Other flags worth knowing about:

- `--session PATH` — resume a previously saved session file instead of
  starting from a fresh layout. The association scan still runs
  against `data.csv` either way.
- `--max-rows`, `--test-size`, `--random-state`, `--plot-grid-size` —
  tune the association scan (subsample size, held-out fraction per
  pair, seed, points per prediction curve). Defaults are meant to be
  reasonable at 50+ variables and/or 100k+ rows; lower `--max-rows` if
  a session with a lot of columns feels slow to start.
- `--host`, `--port`, `--no-browser` — server binding and whether to
  auto-open a browser tab.

Run `dagshop launch --help` for the full list.

### `dagshop generate-demo-data` — try it without your own data

```bash
dagshop generate-demo-data
```

Writes a synthetic CSV with a known causal structure to `inputs/demo.csv`
by default (pass a path to write elsewhere). The generated columns:

- `age`, `prior_engagement` — confounders, mutually independent
- `treatment` (binary) — depends on both confounders
- `mediator` — depends on `treatment`
- `outcome` — depends on `treatment`, `mediator`, and both confounders
- `unrelated_score`, `unrelated_flag` (binary) — pure noise, connected
  to nothing

Then try the workshop against it:

```bash
dagshop launch inputs/demo.csv --treatment treatment --outcome outcome
```

`--n-rows`/`--random-state` control size and reproducibility;
`--force` overwrites an existing output path.

### `inputs/` and `outputs/`

Two folders are set up as the conventional place for this: point
`generate-demo-data` and `launch` at CSVs under `inputs/`, and exports/
saved sessions default into `outputs/`. Both folders are gitignored
(only tracked via `.gitkeep`), so nothing you generate or export ends
up committed by accident.

One thing worth knowing: the export/save/load path prompts in the UI
default to `outputs/...`, and that path is resolved relative to
whichever directory you ran `dagshop launch` from — not necessarily the
repo root. If you launch from somewhere else, either `cd` into the repo
first or type a full path into the prompt.

## The workshop

Once `launch` is running and your browser opens:

- The left panel shows the association ranking table(s) from the
  pre-work scan. Click a row to open a scatter + model-prediction plot
  for that pair.
- The canvas holds one node per column. If you designated
  treatment(s)/outcome(s), those nodes are pinned in fixed columns and
  distinctly colored; everything else starts scattered — drag to
  position.
- Hold <kbd>Shift</kbd> and drag from one node to another to draw a
  directed edge. A plain drag repositions a node instead.
- Every edge needs a `+`/`−` sign, set entirely by your own domain
  knowledge — nothing is pre-filled from the data. Click an edge to
  open the same plot modal as the ranking tables, so you can check the
  hypothesized link against the data shape.
- Cycles are allowed with a warning while you're brainstorming; export
  will refuse a graph that still has one.
- Use the topbar to export the DAG as JSON or GraphML, or to save/load
  a session so you can pick a workshop back up later (`dagshop launch
  data.csv --session outputs/dagshop_session.json`).

The exported JSON/GraphML is meant to be loaded straight into your own
`dowhy.gcm.ProbabilisticCausalModel`/`StructuralCausalModel`, with
`gcm.auto.assign_causal_mechanisms(causal_model, data)` picking up
model selection, and any treatment/outcome role tags feeding calls like
`gcm.average_causal_effect` directly.

## From DAG to causal analysis with dowhy.gcm

DAGshop's job stops at the exported DAG -- the analysis itself is
`dowhy.gcm`'s (not a DAGshop dependency; `pip install dowhy` in your own
environment). All four examples below share this setup:

```python
import networkx as nx
import pandas as pd
from dowhy import gcm

graph = nx.read_graphml("outputs/dag_export.graphml")
data = pd.read_csv("inputs/demo.csv")

# Role tags carried over from the workshop -- read them back off the
# graph rather than re-typing which columns were treatment/outcome.
treatments = [n for n, role in graph.nodes(data="role") if role == "treatment"]
outcomes = [n for n, role in graph.nodes(data="role") if role == "outcome"]

causal_model = gcm.StructuralCausalModel(graph)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
```

### Refuting the graph

Worth running before trusting anything below -- checks whether the
data's independence structure is actually consistent with the DAG you
drew:

```python
from dowhy.gcm.falsify import falsify_graph

result = falsify_graph(graph, data)
print(result)
```

This only needs `graph` and `data`, not the fitted `causal_model`.

### Interventions

```python
samples = gcm.interventional_samples(
    causal_model,
    {treatments[0]: lambda t: 1},  # do(treatment := 1)
    num_samples_to_draw=1000,
)
```

### Counterfactuals

```python
counterfactual_model = gcm.InvertibleStructuralCausalModel(graph)
gcm.auto.assign_causal_mechanisms(counterfactual_model, data)
gcm.fit(counterfactual_model, data)

gcm.counterfactual_samples(
    counterfactual_model,
    {"age": lambda age: age + 10},
    observed_data=data.iloc[[0]],
)
```

One thing to know: counterfactuals need invertible mechanisms on every
node the query passes through, which is a stronger requirement than the
mechanisms auto-assigned above -- a binary node (this demo's
`treatment`) can't reuse a classifier the same way. Check dowhy's docs
on invertible models if your causal path crosses a discrete/binary
variable.

### Causal intrinsic attribution

```python
contributions = gcm.intrinsic_causal_influence(causal_model, outcomes[0])
```

Returns each ancestor's share of the outcome's own variance/uncertainty
-- a natural next look at exactly the covariates DAGshop's own
association tables already flagged as related.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff format
uv run ruff check
```

There's also a live-server, real-browser smoke test
(`tests/test_frontend_smoke.py`) that drives the actual UI with
Playwright/Chromium. It's marked `frontend` and skipped by default —
kept out of the everyday `dev` sync so day-to-day work doesn't need a
browser download. Run it explicitly:

```bash
uv sync --group frontend
uv run playwright install --with-deps chromium
uv run pytest -m frontend
```
