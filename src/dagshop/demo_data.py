"""Synthetic dataset generators for exercising the workshop UI by hand.

`dagshop generate-demo-data` (SCOPE.md's "Manual testing" section, and
its "Causal attribution feature" section's build order step 3) calls one
of the two generators below and writes the result to a CSV, picked by
`--scenario`. This exists purely so someone without a real dataset yet
-- or verifying a change -- can launch DAGshop against data with a
*known* causal structure and check that the ranking tables, canvas
layout, and plots surface something recognisable rather than random
noise. Neither generator is part of the causal-workshop analysis
pipeline (`associate.py`/`graph.py`/`server.py`) or the causal
attribution wrapper (`causal_model.py`, build order step 4); they never
touch either.

Two scenarios:

- `make_demo_data` (`--scenario confounder`, the default): a small
  confounder/treatment/mediator/outcome DAG. See its own docstring.
- `make_csat_demo_data` (`--scenario csat`): a multi-hop customer-service
  operations DAG, added for the causal attribution feature so "attribute
  to all ancestors, not just direct parents" has a real, documented
  structure to prove itself against. See its own docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(n_rows: int = 500, random_state: int = 0) -> pd.DataFrame:
    """Build a synthetic dataset for the confounder/treatment/mediator/outcome scenario.

    Ground-truth structure (all columns continuous unless noted; matches
    SCOPE.md's v1 constraint of continuous/binary only, no categoricals):

        age, prior_engagement            -- confounders, mutually independent
        treatment (binary)                <- age, prior_engagement
        mediator                          <- treatment
        outcome                           <- treatment, mediator, age, prior_engagement
        unrelated_score, unrelated_flag (binary) -- pure noise, connected to nothing

    `treatment` and `outcome` are the intended `--treatment`/`--outcome`
    column names when launching against this data. Effect sizes are
    chosen well above noise so the association scan's ranking tables
    show a clear split (confounders/mediator ranking high, `unrelated_*`
    ranking low) without needing a large `n_rows`.
    """
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


def make_csat_demo_data(n_rows: int = 500, random_state: int = 0) -> pd.DataFrame:
    """Build a synthetic customer-service operations dataset, target `csat`.

    Ground-truth structure (all columns continuous unless noted; the
    nine drivers and target are SCOPE.md's "Causal attribution feature"
    brainstorm list verbatim). Deliberately multi-hop: only 5 of the 9
    drivers are direct parents of `csat`, the other 4 reach it only
    through intermediate nodes, so a ranking built from direct parents
    alone would miss them entirely -- the case
    `gcm.intrinsic_causal_influence` (any ancestor, not just parents) is
    meant to cover once `causal_model.py` (build order step 4) exists.

        age                    -- root
        friction_severity      -- root: latent severity/complexity of
                                   the underlying issue
        time_to_respond        -- root: staffing/queue-driven, minutes
                                   to first response

        num_transfers          <- friction_severity
        num_escalations        <- friction_severity
        num_agents_spoken_to   <- num_transfers, num_escalations

        time_to_resolve        <- time_to_respond, friction_severity,
                                   num_transfers, num_escalations,
                                   num_agents_spoken_to
        resolved (binary)      <- time_to_resolve, friction_severity,
                                   num_escalations
        repeat_contact (binary) <- resolved, friction_severity

        csat (target)          <- age, friction_severity,
                                   time_to_resolve, resolved,
                                   repeat_contact

    `friction_severity` sits upstream of `num_transfers`/`num_escalations`
    rather than downstream of them or feeding `csat` directly: SCOPE.md's
    "Causal attribution feature" section flagged this placement as
    undecided (issue complexity driving the operational response, vs.
    the reverse, vs. a parallel direct path), and Ryan picked "issue
    complexity drives response" when asked directly (session 12).

    Every one of the nine drivers is a genuine ancestor of `csat`,
    `time_to_respond`/`num_transfers`/`num_escalations`/
    `num_agents_spoken_to` only indirectly (through `time_to_resolve`,
    and for `num_escalations` also through `resolved`). No pure-noise
    column is included here (unlike `make_demo_data`'s
    `unrelated_score`/`unrelated_flag`): SCOPE.md's brainstorm names
    exactly these nine drivers, nothing extra.

    Approximate expected influence on `csat`, strongest to weakest --
    from this module's own absolute Pearson correlation with `csat` at
    n=5000, not `gcm.intrinsic_causal_influence` (no causal model exists
    yet to compute that against; re-check this ranking once
    `causal_model.py` lands):

        friction_severity > time_to_resolve > resolved > repeat_contact
        > num_transfers > num_agents_spoken_to > num_escalations
        > age > time_to_respond

    `friction_severity` ranks highest despite a modest direct
    coefficient because nearly everything else in the graph is
    downstream of it; `age` and `time_to_respond` rank lowest as the two
    weakest, most indirect drivers, not because either is disconnected
    from `csat` the way `unrelated_score`/`unrelated_flag` are from
    `outcome` in `make_demo_data`.
    """
    rng = np.random.default_rng(random_state)

    age = rng.normal(45, 12, size=n_rows)
    friction_severity = rng.gamma(shape=2.0, scale=2.5, size=n_rows)
    time_to_respond = rng.gamma(shape=3.0, scale=5.0, size=n_rows)

    num_transfers = rng.poisson(0.18 * friction_severity)
    num_escalations = rng.poisson(0.08 * friction_severity)
    num_agents_spoken_to = 1 + rng.poisson(0.6 * num_transfers + 0.4 * num_escalations)

    time_to_resolve = np.clip(
        2.0
        + 0.15 * time_to_respond
        + 0.8 * friction_severity
        + 1.2 * num_transfers
        + 1.5 * num_escalations
        + 0.5 * num_agents_spoken_to
        + rng.normal(0, 2, size=n_rows),
        0.1,
        None,
    )

    resolved_logit = 1.5 - 0.05 * time_to_resolve - 0.1 * friction_severity - 0.15 * num_escalations
    resolved_prob = 1.0 / (1.0 + np.exp(-resolved_logit))
    resolved = rng.binomial(1, resolved_prob)

    repeat_contact_logit = -0.5 - 1.5 * resolved + 0.15 * friction_severity
    repeat_contact_prob = 1.0 / (1.0 + np.exp(-repeat_contact_logit))
    repeat_contact = rng.binomial(1, repeat_contact_prob)

    csat = np.clip(
        8.0
        - 0.4 * friction_severity
        - 0.1 * time_to_resolve
        + 1.5 * resolved
        - 1.0 * repeat_contact
        + 0.02 * (age - 45)
        + rng.normal(0, 1, size=n_rows),
        0.0,
        10.0,
    )

    return pd.DataFrame(
        {
            "age": age,
            "friction_severity": friction_severity,
            "time_to_respond": time_to_respond,
            "num_transfers": num_transfers,
            "num_escalations": num_escalations,
            "num_agents_spoken_to": num_agents_spoken_to,
            "time_to_resolve": time_to_resolve,
            "resolved": resolved,
            "repeat_contact": repeat_contact,
            "csat": csat,
        }
    )
