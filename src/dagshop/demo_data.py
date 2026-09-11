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
    brainstorm list verbatim). Deliberately multi-hop: only 6 of the 9
    drivers are direct parents of `csat`, the other 3 reach it only
    through intermediate nodes, so a ranking built from direct parents
    alone would miss them entirely -- the case
    `gcm.intrinsic_causal_influence` (any ancestor, not just parents) is
    meant to cover.

        age                    -- root
        friction_severity      -- root: latent severity/complexity of
                                   the underlying issue
        time_to_respond        -- root: staffing/queue-driven, minutes
                                   to first response
        repeat_contact (binary) -- root: this customer's own baseline
                                   propensity to contact support again,
                                   independent of this particular issue

        num_transfers          <- friction_severity
        num_escalations        <- friction_severity
        num_agents_spoken_to   <- num_transfers, num_escalations

        time_to_resolve        <- num_transfers, num_escalations,
                                   num_agents_spoken_to
        resolved (binary)      <- time_to_resolve, num_transfers,
                                   num_escalations

        csat (target)          <- age, time_to_respond, repeat_contact,
                                   time_to_resolve, num_agents_spoken_to,
                                   resolved

    Revised from an earlier version of this scenario: `friction_severity`
    used to also feed `csat`, `resolved`, and `repeat_contact` directly,
    and `repeat_contact` used to be caused by `resolved`/
    `friction_severity` rather than being a root. `friction_severity` is
    now purely upstream -- every one of its effects on `csat` runs
    through `num_transfers`/`num_escalations` and what they cascade
    into, nothing direct -- which makes it a cleaner test of "attribute
    to every ancestor, not just direct parents" than the earlier version
    was, since it no longer has a direct edge to lean on. `age` also
    flips from a small positive effect on `csat` to a small negative
    one in this revision.

    Every one of the nine drivers is a genuine ancestor of `csat`.
    `friction_severity`, `num_transfers`, and `num_escalations` reach
    it only indirectly, through `time_to_resolve`, `resolved`, and/or
    `num_agents_spoken_to`. No pure-noise column is included here
    (unlike `make_demo_data`'s `unrelated_score`/`unrelated_flag`):
    SCOPE.md's brainstorm names exactly these nine drivers, nothing
    extra.

    Expected influence on `csat`, strongest to weakest -- from
    `causal_model.attribute_target`'s intrinsic causal influence
    (Shapley-based) against this generator's own output at n=1000,
    `random_state=0` for both the data and the attribution call,
    `n_jobs=1` (see `causal_model.py`'s module docstring on why
    `n_jobs=1` matters for reproducibility). Checked stable across a
    few other attribution seeds against the same data, at both default
    and reduced (faster, noisier) sample settings: `resolved` wins the
    top ancestor spot by a comfortable margin every time.

        resolved > time_to_respond > repeat_contact > age
        > friction_severity > time_to_resolve > num_transfers
        > num_agents_spoken_to > num_escalations

    `csat`'s own row in the raw `intrinsic_causal_influence` output
    (its unexplained/intrinsic variance, not an ancestor at all) is
    larger than any single ancestor's share here -- roughly 43% of the
    total at n=1000, `random_state=0` -- because most of `csat`'s
    variance in this synthetic data is noise, by construction
    (`rng.normal(0, 1)` added at the end). That is expected and is
    exactly the case the UI's "Other" relabeling (see SCOPE.md's
    "Requested changes" backlog) exists to make legible, not a sign
    anything is wrong with the scenario.
    """
    rng = np.random.default_rng(random_state)

    age = rng.normal(45, 12, size=n_rows)
    friction_severity = rng.gamma(shape=2.0, scale=2.5, size=n_rows)
    time_to_respond = rng.gamma(shape=3.0, scale=5.0, size=n_rows)
    repeat_contact = rng.binomial(1, 0.3, size=n_rows)

    num_transfers = rng.poisson(0.18 * friction_severity)
    num_escalations = rng.poisson(0.08 * friction_severity)
    num_agents_spoken_to = 1 + rng.poisson(0.6 * num_transfers + 0.4 * num_escalations)

    time_to_resolve = np.clip(
        2.0
        + 1.2 * num_transfers
        + 1.5 * num_escalations
        + 0.5 * num_agents_spoken_to
        + rng.normal(0, 2, size=n_rows),
        0.1,
        None,
    )

    resolved_logit = 1.5 - 0.05 * time_to_resolve - 0.08 * num_transfers - 0.15 * num_escalations
    resolved_prob = 1.0 / (1.0 + np.exp(-resolved_logit))
    resolved = rng.binomial(1, resolved_prob)

    csat = np.clip(
        8.0
        - 0.02 * (age - 45)
        - 0.05 * time_to_respond
        - 1.0 * repeat_contact
        - 0.1 * time_to_resolve
        - 0.15 * num_agents_spoken_to
        + 1.5 * resolved
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
