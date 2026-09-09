"""DAG data model: nodes, edges, user-asserted signs, cycle handling.

networkx.DiGraph-backed. Cycle detection warns rather than blocks, to
support free brainstorming; validity is enforced before export. JSON and
graphml serialization, save/resume.

Pure logic, no UI or fitting dependency: first module in the build order,
easiest to unit test in isolation. See SCOPE.md build order step 1.

Not yet implemented.
"""
