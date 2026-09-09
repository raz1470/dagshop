"""Synthetic dataset generator for exercising the workshop UI by hand.

`dagshop generate-demo-data` (SCOPE.md's "Manual testing" section) calls
`make_demo_data` and writes the result to a CSV. This exists purely so
someone without a real dataset yet -- or verifying a change -- can
launch DAGshop against data with a *known* causal structure and check
that the ranking tables, canvas layout, and plots surface something
recognisable rather than random noise. It is not part of the
causal-workshop analysis pipeline (`associate.py`/`graph.py`/
`server.py`) and never touches it.

Ground-truth structure (all columns continuous unless noted; matches
SCOPE.md's v1 constraint of continuous/binary only, no categoricals):

    age, prior_engagement            -- confounders, mutually independent
    treatment (binary)                <- age, prior_engagement
    mediator                          <- treatment
    outcome                           <- treatment, mediator, age, prior_engagement
    unrelated_score, unrelated_flag (binary) -- pure noise, connected to nothing

`treatment` and `outcome` are the intended `--treatment`/`--outcome`
column names when launching against this data. Effect sizes are chosen
well above noise so the association scan's ranking tables show a clear
split (confounders/mediator ranking high, `unrelated_*` ranking low)
without needing a large `n_rows`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(n_rows: int = 500, random_state: int = 0) -> pd.DataFrame:
    """Build a synthetic dataset with the causal structure in the module docstring."""
    rng = np.random.default_rng(random_state)

    age = rng.normal(45, 12, size=n_rows)
    prior_engagement = rng.normal(50, 15, size=n_rows)

    treatment_logit = 0.04 * (age - 45) + 0.05 * (prior_engagement - 50)
    treatment_prob = 1.0 / (1.0 + np.exp(-treatment_logit))
    treatment = rng.binomial(1, treatment_prob)

    mediator = 2.0 * treatment + rng.normal(0, 1, size=n_rows)

    outcome = (
        5.0 * treatment
        + 1.5 * mediator
        + 0.3 * age
        + 0.2 * prior_engagement
        + rng.normal(0, 5, size=n_rows)
    )

    unrelated_score = rng.normal(0, 1, size=n_rows)
    unrelated_flag = rng.binomial(1, 0.5, size=n_rows)

    return pd.DataFrame(
        {
            "age": age,
            "prior_engagement": prior_engagement,
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
            "unrelated_score": unrelated_score,
            "unrelated_flag": unrelated_flag,
        }
    )
