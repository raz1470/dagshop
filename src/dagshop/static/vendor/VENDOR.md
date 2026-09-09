# Vendored frontend assets

Per `SCOPE.md`'s data handling and security section: no CDN-loaded frontend
assets, no telemetry from any dependency, nothing added without a quick
telemetry check first. All three libraries below are fetched once (via
`npm pack`, network access from the cloud build container, never from the
end user's machine) and committed into this repo as static files. `dagshop`
itself makes no npm/CDN calls at runtime or install time.

## cytoscape.js 3.34.3

- Source: `npm pack cytoscape@3` (dist/cytoscape.min.js)
- License: MIT (The Cytoscape Consortium) -- see `cytoscape/LICENSE`
- Telemetry check: grepped the minified bundle for outbound-request
  patterns (`sendBeacon`, `XMLHttpRequest`, `fetch(`, known analytics
  domains) and for embedded URLs. Only hits are license-header URLs
  (`opensource.org/licenses/MIT`, `engelschall.com`, Wikipedia's MIT
  License page) inside the license banner text. No network code paths.

## cytoscape-edgehandles 4.0.1

- Source: `npm pack cytoscape-edgehandles@4` (cytoscape-edgehandles.js,
  unminified UMD build -- package ships no separate dist/min file)
- License: MIT (The Cytoscape Consortium) -- see
  `cytoscape-edgehandles/LICENSE`
- Peer dependency: `cytoscape@^3.2.0` (already vendored above).
- Runtime dependency: the package's own `package.json` lists
  `lodash.memoize`/`lodash.throttle`, and its UMD browser-global branch
  reads them off a global `_` object (`root["_"]["memoize"]`,
  `root["_"]["throttle"]`) -- i.e. it expects a full lodash global, not
  the two standalone packages its `package.json` names. Rather than
  vendor all of lodash for two functions, `js/compat-shim.js` defines a
  minimal `window._` with just `.memoize`/`.throttle` (our own code, not
  a repackaged third-party library -- no separate telemetry check
  needed). Load order in `index.html`: cytoscape, then
  `compat-shim.js`, then `cytoscape-edgehandles.js`.
- Telemetry check: grepped for the same patterns as above. No hits.

## plotly.js-basic-dist-min 2.35.3

- Source: `npm pack plotly.js-basic-dist-min@2` (plotly-basic.min.js).
  The "basic" partial bundle (not full `plotly.js`) is a deliberate
  choice: it includes the `scatter` trace type (markers + lines), which
  is all `PairPlotData`'s scatter-plus-prediction-curve modal needs, at
  ~1.1 MB minified instead of the full bundle's ~4-5 MB.
- License: MIT (Plotly, Inc) -- see `plotly/LICENSE`
- Telemetry check: grepped for outbound-request patterns. One hit:
  `new XMLHttpRequest` appears once, part of Plotly's optional
  image-export-to-server / Chart Studio upload helpers
  (`Plotly.toImage`-adjacent cloud-export code paths). `dagshop`'s
  frontend never calls those functions -- the plot modal only calls
  `Plotly.newPlot`/`Plotly.react` to render locally-supplied data, so
  this code path is present but dead in our usage. A handful of
  `https://...` string literals are also present: SVG/XML namespace URIs
  (`w3.org/2000/svg` etc, used as string constants when creating SVG
  elements, not fetched) and doc-comment references
  (`plotly.com/javascript/...`, `cdn.plot.ly` mentioned in a comment).
  None are runtime network calls.

## Not yet audited (tracked in SCOPE.md's Open items)

Backend Python dependencies added this slice (`fastapi`, `uvicorn`) are
not yet telemetry-audited, same as the pre-existing open item for
`networkx`/`pandas`/`scikit-learn`. Both are widely-used, telemetry-free
libraries by reputation, but SCOPE.md asks for an explicit check before
go-live, not a reputational assumption -- left as-is for that pass
rather than done piecemeal per session.
