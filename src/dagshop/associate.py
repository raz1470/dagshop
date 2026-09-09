"""Pre-work: scoped or full pairwise HistGradientBoosting association scan.

Purely exploratory: informs the PM/DS during the workshop, does not set
anything on the DAG automatically. Ranks variables by association with
designated treatment(s)/outcome(s) (n * (t + o) fits) when designated, or
falls back to a full pairwise scan (n * (n - 1)) when not.

Depends only on pandas/sklearn, not on graph.py or the UI. See SCOPE.md
build order step 2.

Not yet implemented.
"""
