"""FastAPI app serving the DAG workshop UI.

Vendored Cytoscape.js frontend (drag/drop nodes, click-drag directional
edges) and Plotly.js for pairwise association plots, both bundled under
static/, no CDN. Ranking tables from associate.py, plot modals, DAG canvas
backed by graph.py. Local-only (localhost), no external network calls, per
SCOPE.md's data handling and security constraints.

Needs graph.py and associate.py working first, since it serves their
outputs. See SCOPE.md build order step 3.

Not yet implemented.
"""
