"use strict";

/**
 * DAGshop workshop UI. Rough first pass: working,
 * not polished. Vanilla JS, no build step, no framework -- matches
 * "vendored, no CDN" static assets with nothing to bundle.
 *
 * Talks to server.py's REST API (one endpoint per user action -- see
 * server.py's module docstring for the state-model decision behind
 * that). Cytoscape.js draws the canvas; cytoscape-edgehandles handles
 * click-drag edge creation; Plotly renders the pairwise plot modal from
 * associate.py's cached data, fetched on demand per SCOPE.md.
 */

// -- DOM refs -------------------------------------------------------------

const cyContainer = document.getElementById("cy");
const banner = document.getElementById("banner");
const datasetSummary = document.getElementById("dataset-summary");
const tablesContainer = document.getElementById("tables-container");
const skippedContainer = document.getElementById("skipped-container");

const plotModal = document.getElementById("plot-modal");
const plotModalTitle = document.getElementById("plot-modal-title");
const plotModalBody = document.getElementById("plot-modal-body");

const signModal = document.getElementById("sign-modal");
const signModalSubtitle = document.getElementById("sign-modal-subtitle");
const signPlusBtn = document.getElementById("sign-plus");
const signMinusBtn = document.getElementById("sign-minus");
const signCancelBtn = document.getElementById("sign-cancel");

const pathModal = document.getElementById("path-modal");
const pathModalTitle = document.getElementById("path-modal-title");
const pathModalInput = document.getElementById("path-modal-input");
const pathModalConfirm = document.getElementById("path-modal-confirm");
const pathModalCancel = document.getElementById("path-modal-cancel");

const btnCausalBuild = document.getElementById("btn-causal-build");
const causalBuildResult = document.getElementById("causal-build-result");
const causalAttributeControls = document.getElementById("causal-attribute-controls");
const causalTargetSelect = document.getElementById("causal-target-select");
const btnCausalAttribute = document.getElementById("btn-causal-attribute");
const causalContributionContainer = document.getElementById("causal-contribution-container");

let cy = null;

// -- API helper -------------------------------------------------------------

async function api(path, method = "GET", body) {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = await res.json();
      if (payload && payload.detail) detail = payload.detail;
    } catch (_e) {
      // response body wasn't JSON; fall back to statusText
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  return res.json();
}

function edgeId(source, target) {
  // "=>" rather than "->" to reduce (not eliminate) collision risk with
  // real column names. A column literally containing "=>" is a known,
  // ignored v1 edge case.
  return `${source}=>${target}`;
}

// -- banner ------------------------------------------------------------------

function showBanner(message) {
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function hideBanner() {
  banner.classList.add("hidden");
}

// -- generic modal helpers ----------------------------------------------------

function showModal(modal) {
  modal.classList.remove("hidden");
}

function hideModal(modal) {
  modal.classList.add("hidden");
}

document.querySelectorAll("[data-close]").forEach((btn) => {
  btn.addEventListener("click", () => hideModal(document.getElementById(btn.dataset.close)));
});

// -- dataset summary -----------------------------------------------------------

function renderDatasetSummary(health) {
  const scanKind = health.scoped ? "scoped scan (treatment/outcome-focused)" : "full pairwise scan";
  datasetSummary.textContent =
    `${health.data_path} — ${health.n_rows} rows × ${health.n_columns} columns — ${scanKind}`;
}

// -- ranking tables --------------------------------------------------------------

function formatScore(row) {
  const label = row.score_name === "roc_auc" ? "roc_auc" : "r2";
  return `${row.score.toFixed(3)} (${label})`;
}

function buildTable(title, rows) {
  const table = document.createElement("table");
  table.className = "rank-table";
  const caption = document.createElement("caption");
  caption.textContent = `${title} (${rows.length})`;
  table.appendChild(caption);

  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>Predictor</th><th>Target</th><th>Score</th><th>n</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.predictor)}</td>
      <td>${escapeHtml(row.target)}</td>
      <td class="score-cell">${formatScore(row)}</td>
      <td>${row.n_used}</td>
    `;
    tr.addEventListener("click", () => openPlot(row.predictor, row.target));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return table;
}

// A table's rows can target columns of different kinds (e.g. covariate_table's
// targets are every covariate, continuous and binary alike), which means
// score_name -- and therefore the metric's scale -- isn't the same across every
// row: ROC AUC's floor sits around 0.5 ("no skill") while R2's floor can run
// much lower or negative, so ranking both together by raw score would always
// favor the AUC rows regardless of which relationship is actually stronger.
// Rather than invent a normalized cross-metric score (a modeling judgment call
// SCOPE.md deliberately avoids -- "sorting, not auto-flagging"), split into one
// table per metric whenever a group mixes them; each list is already sorted by
// score, so a stable filter by score_name keeps that order intact.
function renderTableGroup(container, title, rows) {
  if (rows.length === 0) return;
  const scoreNames = [...new Set(rows.map((row) => row.score_name))];
  if (scoreNames.length <= 1) {
    container.appendChild(buildTable(title, rows));
    return;
  }
  for (const scoreName of scoreNames) {
    container.appendChild(
      buildTable(
        `${title} — ${scoreName}`,
        rows.filter((row) => row.score_name === scoreName),
      ),
    );
  }
}

function renderTables(tables) {
  tablesContainer.innerHTML = "";
  if (tables.scoped) {
    renderTableGroup(tablesContainer, "Associated with treatment(s)", tables.treatment_table);
    renderTableGroup(tablesContainer, "Associated with outcome(s)", tables.outcome_table);
    renderTableGroup(tablesContainer, "Associated with covariate(s)", tables.covariate_table);
  } else {
    renderTableGroup(tablesContainer, "Pairwise associations", tables.full_table);
  }

  skippedContainer.innerHTML = "";
  if (tables.skipped.length > 0) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${tables.skipped.length} pair(s) skipped during the scan`;
    details.appendChild(summary);
    const ul = document.createElement("ul");
    for (const s of tables.skipped) {
      const li = document.createElement("li");
      li.textContent = `${s.predictor} → ${s.target}: ${s.reason}`;
      ul.appendChild(li);
    }
    details.appendChild(ul);
    skippedContainer.appendChild(details);
  }
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

// -- plot modal ------------------------------------------------------------------

async function openPlot(predictor, target) {
  plotModalTitle.textContent = `${predictor} → ${target}`;
  plotModalBody.innerHTML = "<p>Loading…</p>";
  showModal(plotModal);
  try {
    const data = await api(`/api/plot/${encodeURIComponent(predictor)}/${encodeURIComponent(target)}`);
    plotModalBody.innerHTML = "";
    const plotDiv = document.createElement("div");
    plotDiv.style.width = "100%";
    plotDiv.style.height = "420px";
    plotModalBody.appendChild(plotDiv);
    Plotly.newPlot(
      plotDiv,
      [
        {
          x: data.x,
          y: data.y,
          mode: "markers",
          type: "scatter",
          name: "observed",
          marker: { size: 5, opacity: 0.55, color: "#6b6b76" },
        },
        {
          x: data.grid_x,
          y: data.grid_prediction,
          mode: "lines",
          type: "scatter",
          name: "model prediction",
          line: { color: "#2e6bd6", width: 2 },
        },
      ],
      {
        margin: { t: 30, r: 10, b: 40, l: 50 },
        xaxis: { title: predictor },
        yaxis: { title: target },
        // Anchored just above the plotting area (not inside it) so the legend
        // never overlaps markers/prediction line regardless of data range.
        legend: { orientation: "h", x: 1, xanchor: "right", y: 1, yanchor: "bottom" },
      },
      { displaylogo: false, responsive: true },
    );
  } catch (err) {
    plotModalBody.innerHTML =
      err.status === 404
        ? "<p>No cached association plot for this pair (only pairs the scan actually ran are cached).</p>"
        : `<p>Could not load plot: ${escapeHtml(err.message)}</p>`;
  }
}

// -- sign modal (every edge needs a user-asserted sign, no default) -----------------

function pickSign(source, target) {
  return new Promise((resolve) => {
    signModalSubtitle.textContent = `${source} → ${target}`;
    showModal(signModal);

    function cleanup() {
      signPlusBtn.removeEventListener("click", onPlus);
      signMinusBtn.removeEventListener("click", onMinus);
      signCancelBtn.removeEventListener("click", onCancel);
      hideModal(signModal);
    }
    function onPlus() {
      cleanup();
      resolve("+");
    }
    function onMinus() {
      cleanup();
      resolve("-");
    }
    function onCancel() {
      cleanup();
      resolve(null);
    }
    signPlusBtn.addEventListener("click", onPlus);
    signMinusBtn.addEventListener("click", onMinus);
    signCancelBtn.addEventListener("click", onCancel);
  });
}

// -- path modal (export/save/load all take a filesystem path) -----------------------

function promptPath(title, defaultValue) {
  return new Promise((resolve) => {
    pathModalTitle.textContent = title;
    pathModalInput.value = defaultValue || "";
    showModal(pathModal);
    pathModalInput.focus();
    pathModalInput.select();

    function cleanup() {
      pathModalConfirm.removeEventListener("click", onConfirm);
      pathModalCancel.removeEventListener("click", onCancel);
      hideModal(pathModal);
    }
    function onConfirm() {
      const value = pathModalInput.value.trim();
      cleanup();
      resolve(value || null);
    }
    function onCancel() {
      cleanup();
      resolve(null);
    }
    pathModalConfirm.addEventListener("click", onConfirm);
    pathModalCancel.addEventListener("click", onCancel);
  });
}

// -- canvas context menu ------------------------------------------------------------

let activeMenu = null;

function closeContextMenu() {
  if (activeMenu) {
    activeMenu.remove();
    activeMenu = null;
  }
}
document.addEventListener("click", closeContextMenu);

function showContextMenu(x, y, items) {
  closeContextMenu();
  const menu = document.createElement("div");
  menu.className = "cy-context-menu";
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  for (const { label, onClick } of items) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", (evt) => {
      evt.stopPropagation();
      closeContextMenu();
      onClick();
    });
    menu.appendChild(btn);
  }
  document.body.appendChild(menu);
  activeMenu = menu;
}

// -- node/edge mutations --------------------------------------------------------------

async function updateNodePosition(node) {
  const { x, y } = node.position();
  try {
    await api(`/api/nodes/${encodeURIComponent(node.id())}/position`, "PUT", { x, y });
  } catch (err) {
    showBanner(`Could not save position for "${node.id()}": ${err.message}`);
  }
}

async function deleteNode(name) {
  if (!window.confirm(`Delete node "${name}"? This also removes any edges touching it.`)) return;
  try {
    await api(`/api/nodes/${encodeURIComponent(name)}`, "DELETE");
    const el = cy.getElementById(name);
    if (el.length > 0) el.remove(); // cytoscape removes incident edges too
  } catch (err) {
    showBanner(`Could not delete node "${name}": ${err.message}`);
  }
}

async function deleteEdge(source, target) {
  try {
    await api(`/api/edges/${encodeURIComponent(source)}/${encodeURIComponent(target)}`, "DELETE");
    const el = cy.getElementById(edgeId(source, target));
    if (el.length > 0) el.remove();
  } catch (err) {
    showBanner(`Could not delete edge "${source} → ${target}": ${err.message}`);
  }
}

async function flipSign(source, target, currentSign) {
  const newSign = currentSign === "+" ? "-" : "+";
  try {
    const result = await api(
      `/api/edges/${encodeURIComponent(source)}/${encodeURIComponent(target)}/sign`,
      "PUT",
      { sign: newSign },
    );
    cy.getElementById(edgeId(source, target)).data("sign", result.sign);
  } catch (err) {
    showBanner(`Could not update sign for "${source} → ${target}": ${err.message}`);
  }
}

async function createEdge(source, target) {
  const sign = await pickSign(source, target);
  if (sign === null) return; // cancelled: no edge, matches "no data-derived default"
  try {
    const result = await api("/api/edges", "POST", { source, target, sign });
    cy.add({
      group: "edges",
      data: { id: edgeId(source, target), source, target, sign: result.sign },
    });
    if (result.cycle_warning) {
      showBanner(`Cycle warning: ${result.cycle_warning}`);
    } else {
      hideBanner();
    }
  } catch (err) {
    showBanner(`Could not add edge "${source} → ${target}": ${err.message}`);
  }
}

// -- cytoscape setup ------------------------------------------------------------------

function toCyElements(graph) {
  const nodes = graph.nodes.map((n) => ({
    group: "nodes",
    data: { id: n.name, role: n.role },
    position: { x: n.x ?? 400, y: n.y ?? 400 },
  }));
  const edges = graph.edges.map((e) => ({
    group: "edges",
    data: { id: edgeId(e.source, e.target), source: e.source, target: e.target, sign: e.sign },
  }));
  return [...nodes, ...edges];
}

function cyStyle() {
  return [
    {
      selector: "node",
      style: {
        "background-color": "#8a8a94",
        label: "data(id)",
        width: 34,
        height: 34,
        "font-size": 10,
        "text-valign": "bottom",
        "text-margin-y": 4,
        "border-width": 1,
        "border-color": "#00000022",
      },
    },
    { selector: 'node[role = "treatment"]', style: { "background-color": "#2e6bd6" } },
    { selector: 'node[role = "outcome"]', style: { "background-color": "#d6642e" } },
    {
      selector: "edge",
      style: {
        width: 2,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "line-color": "#8a8a94",
        "target-arrow-color": "#8a8a94",
        label: "data(sign)",
        "font-size": 12,
        "text-background-color": "#fbfbfc",
        "text-background-opacity": 1,
        "text-background-padding": 2,
      },
    },
    {
      selector: 'edge[sign = "+"]',
      style: { "line-color": "#1f8a4c", "target-arrow-color": "#1f8a4c" },
    },
    {
      selector: 'edge[sign = "-"]',
      style: { "line-color": "#c23b3b", "target-arrow-color": "#c23b3b" },
    },
    // cytoscape-edgehandles 4.0.1's actual classes (see the edge-drawing
    // bug-fix comment in initCytoscape for how these get applied --
    // there is no separate "handle" node in this vendored version, so
    // there used to be a dead ".eh-handle" rule here that never
    // matched anything; removed).
    {
      selector: ".eh-source, .eh-target",
      style: { "border-width": 3, "border-color": "#2e6bd6" },
    },
    { selector: ".eh-hover", style: { "border-width": 3, "border-color": "#1f8a4c" } },
    {
      selector: ".eh-ghost-node",
      style: { "background-color": "#2e6bd6", opacity: 0.6 },
    },
    { selector: ".eh-ghost-edge", style: { "line-style": "dashed", opacity: 0.8 } },
    {
      selector: ".eh-ghost-edge.eh-preview-active",
      style: { "line-color": "#1f8a4c", "target-arrow-color": "#1f8a4c" },
    },
  ];
}

function initCytoscape(graph) {
  cy = cytoscape({
    container: cyContainer,
    elements: toCyElements(graph),
    style: cyStyle(),
    layout: { name: "preset" }, // positions come from server.py's initial layout
    minZoom: 0.2,
    maxZoom: 3,
    // Cytoscape core's own box-selection gesture is bound to Shift+drag
    // by default (regardless of whether the drag starts on a node or
    // the background), which raced against the Shift+drag-to-connect
    // gesture added below and won: a selection box appeared instead of
    // an edge (second bug in the same interaction). Nothing
    // in this app uses multi-select, so disabling it outright is a
    // clean fix rather than trying to out-race it.
    boxSelectionEnabled: false,
  });

  // Treatment/outcome nodes are "pinned" per SCOPE.md step 2: locked by
  // default so a chaotic brainstorm doesn't drag them off their fixed
  // column. Right-click offers "Unlock position" for the rare case
  // someone wants to move one anyway (judgment call -- see NOTES.md).
  cy.nodes().forEach((node) => {
    if (node.data("role")) node.lock();
  });

  cyContainer.addEventListener("contextmenu", (evt) => evt.preventDefault());

  const eh = cy.edgehandles({
    canConnect: (sourceNode, targetNode) => !sourceNode.same(targetNode),
    edgeParams: () => ({}),
    hoverDelay: 150,
    snap: false,
  });

  // Bug fix: Shift+drag couldn't draw an edge. The
  // vendored cytoscape-edgehandles 4.0.1 has no separate "handle" dot
  // to drag (grepped the whole vendored file: the string "eh-handle"
  // -- our own now-removed dead CSS selector -- appears nowhere in the
  // library itself). Its own internal `tapstart` listener only calls
  // `start()` when `drawMode` is true:
  //
  //     this.addListener(cy, 'tapstart', 'node', function (e) {
  //       if (_this.drawMode) { _this.start(node); }
  //     });
  //
  // `drawMode` defaults to false and nothing here ever called
  // `eh.enableDrawMode()`, so plain click-drag on a node could never
  // start an edge -- it just repositioned the node (or did nothing, if
  // locked).
  //
  // Revised once. The first fix called `eh.start()`
  // directly from a `cy.on("tapstart", "node", ...)` listener whenever
  // Shift was already held at mousedown, rather than going through
  // `enableDrawMode()`. That shipped untested and mostly worked by
  // hand, but failed reliably in CI's headless run: calling
  // `eh.start()` from inside `tapstart` is too late to matter, because
  // `enableDrawMode()` (see `toggleDrawMode` above) also runs
  // `cy.autoungrabify(true)` -- specifically so the source node can't
  // *also* be natively grabbed and dragged by cytoscape core while
  // edgehandles is tracking the gesture. Skip that and cytoscape core
  // unconditionally skips emitting `tapdragover`/`tapdragout` (the
  // events edgehandles' `preview()` relies on to ever notice a target
  // node) for as long as any node reports `grabbed() === true`
  // (confirmed by reading cytoscape core's own minified drag handling:
  // `ne&&ne.grabbed()||O==re||(...emit tapdragover...)`). Manual
  // testing never hit this -- drags by hand apparently always involved a
  // locked (pinned treatment/outcome) node, which can't be natively
  // grabbed -- but the Playwright test picked two ordinary grabbable
  // nodes and hit it on every run.
  //
  // Fixed by toggling the library's own `drawMode` on Shift
  // keydown/keyup instead of calling `eh.start()` ourselves: this runs
  // `cy.autoungrabify(true)` *before* the user ever mouses down on the
  // source node, so it's never natively grabbed in the first place,
  // and the library's built-in `tapstart` listener (quoted above)
  // handles starting the gesture. Unmodified drag still moves a node
  // (SCOPE.md step 2's "drag to position"); Shift+drag draws an edge
  // (SCOPE.md step 2's "click-drag to create a directed edge") -- both
  // now go through the code path the library actually tests and
  // documents, rather than one that only worked by the accident of how
  // it happened to be exercised.
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Shift" && !eh.drawMode) eh.enableDrawMode();
  });
  document.addEventListener("keyup", (evt) => {
    if (evt.key === "Shift" && eh.drawMode) eh.disableDrawMode();
  });

  cy.on("ehcomplete", (_evt, sourceNode, targetNode, addedEdge) => {
    addedEdge.remove(); // ephemeral edgehandles edge: not real until signed + saved
    const id = edgeId(sourceNode.id(), targetNode.id());
    if (cy.getElementById(id).length > 0) {
      showBanner(`Edge "${sourceNode.id()} → ${targetNode.id()}" already exists.`);
      return;
    }
    createEdge(sourceNode.id(), targetNode.id());
  });

  cy.on("dragfree", "node", (evt) => updateNodePosition(evt.target));

  cy.on("tap", "edge", (evt) => {
    const edge = evt.target;
    openPlot(edge.data("source"), edge.data("target"));
  });

  cy.on("cxttap", "node", (evt) => {
    const node = evt.target;
    const pos = evt.originalEvent;
    showContextMenu(pos.clientX, pos.clientY, [
      {
        label: node.locked() ? "Unlock position" : "Lock position",
        onClick: () => (node.locked() ? node.unlock() : node.lock()),
      },
      { label: "Delete node", onClick: () => deleteNode(node.id()) },
    ]);
  });

  cy.on("cxttap", "edge", (evt) => {
    const edge = evt.target;
    const pos = evt.originalEvent;
    showContextMenu(pos.clientX, pos.clientY, [
      {
        label: `Flip sign (currently ${edge.data("sign")})`,
        onClick: () => flipSign(edge.data("source"), edge.data("target"), edge.data("sign")),
      },
      { label: "Delete edge", onClick: () => deleteEdge(edge.data("source"), edge.data("target")) },
    ]);
  });

  return eh;
}

async function reloadGraph() {
  const graph = await api("/api/graph");
  cy.elements().remove();
  cy.add(toCyElements(graph));
  cy.nodes().forEach((node) => {
    if (node.data("role")) node.lock();
  });
}

// -- topbar actions ----------------------------------------------------------------

function wireTopbar() {
  document.getElementById("btn-validate").addEventListener("click", async () => {
    try {
      const result = await api("/api/validate");
      if (result.valid) {
        showBanner("Valid: no cycles. Ready to export.");
      } else {
        const formatted = result.cycles.map((c) => [...c, c[0]].join(" → ")).join("; ");
        showBanner(`Not export-ready: ${result.cycles.length} cycle(s): ${formatted}`);
      }
    } catch (err) {
      showBanner(`Could not validate: ${err.message}`);
    }
  });

  document.getElementById("btn-export-json").addEventListener("click", async () => {
    const path = await promptPath("Export DAG as JSON — file path", "outputs/dag_export.json");
    if (!path) return;
    try {
      await api("/api/export", "POST", { path, format: "json" });
      showBanner(`Exported to ${path}`);
    } catch (err) {
      showBanner(`Export failed: ${err.message}`);
    }
  });

  document.getElementById("btn-export-graphml").addEventListener("click", async () => {
    const path = await promptPath("Export DAG as GraphML — file path", "outputs/dag_export.graphml");
    if (!path) return;
    try {
      await api("/api/export", "POST", { path, format: "graphml" });
      showBanner(`Exported to ${path}`);
    } catch (err) {
      showBanner(`Export failed: ${err.message}`);
    }
  });

  document.getElementById("btn-save-session").addEventListener("click", async () => {
    const path = await promptPath("Save session — file path", "outputs/dagshop_session.json");
    if (!path) return;
    try {
      await api("/api/session/save", "POST", { path });
      showBanner(`Session saved to ${path}`);
    } catch (err) {
      showBanner(`Save failed: ${err.message}`);
    }
  });

  document.getElementById("btn-load-session").addEventListener("click", async () => {
    const path = await promptPath("Load session — file path", "outputs/dagshop_session.json");
    if (!path) return;
    try {
      await api("/api/session/load", "POST", { path });
      await reloadGraph();
      showBanner(`Session loaded from ${path}. Ranking tables are unchanged (tied to the data.csv this server was launched with, not to the session file).`);
    } catch (err) {
      showBanner(`Load failed: ${err.message}`);
    }
  });
}


// -- causal attribution panel (SCOPE.md build order step 6) -----------------------

function formatFalsifyBool(value) {
  if (value === null) return "inconclusive";
  return value ? "yes" : "no";
}

function renderCausalBuildResult(result) {
  causalBuildResult.innerHTML = "";
  causalBuildResult.classList.remove("hidden");

  const summary = document.createElement("p");
  summary.className = "causal-falsify-summary";
  summary.textContent =
    `Falsified: ${formatFalsifyBool(result.falsify.falsified)} — falsifiable: ` +
    `${formatFalsifyBool(result.falsify.falsifiable)} (significance level ${result.falsify.significance_level}).`;
  causalBuildResult.appendChild(summary);

  const reportDetails = document.createElement("details");
  const reportSummary = document.createElement("summary");
  reportSummary.textContent = "Full refutation report";
  reportDetails.appendChild(reportSummary);
  const pre = document.createElement("pre");
  pre.className = "causal-falsify-report";
  pre.textContent = result.falsify.report;
  reportDetails.appendChild(pre);
  causalBuildResult.appendChild(reportDetails);

  // Surfaced open (not collapsed) when non-empty: "worth a second look"
  // per SCOPE.md's soft-constraints decision, not something to bury
  // behind a click the way the (usually empty, usually skimmed) full
  // report above is.
  if (result.sign_disagreements.length > 0) {
    const disagreeDetails = document.createElement("details");
    disagreeDetails.open = true;
    const disagreeSummary = document.createElement("summary");
    disagreeSummary.textContent =
      `${result.sign_disagreements.length} edge(s) disagree with the raw correlation`;
    disagreeDetails.appendChild(disagreeSummary);
    const ul = document.createElement("ul");
    for (const d of result.sign_disagreements) {
      const li = document.createElement("li");
      li.textContent =
        `${d.parent} → ${d.child} (asserted "${d.asserted_sign}"): correlation ${d.correlation.toFixed(2)}`;
      ul.appendChild(li);
    }
    disagreeDetails.appendChild(ul);
    causalBuildResult.appendChild(disagreeDetails);
  }
}

function populateCausalTargetSelect(nodes, preferredDefault) {
  causalTargetSelect.innerHTML = "";
  for (const name of nodes) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    causalTargetSelect.appendChild(option);
  }
  if (preferredDefault && nodes.includes(preferredDefault)) {
    causalTargetSelect.value = preferredDefault;
  }
}

// `row.share` (contribution as a fraction of the total across the whole
// call, target's own row included -- see causal_model.py's
// AttributionResult docstring) is already a natural 0-1 range, unlike
// the association-table score column, so no per-metric rescaling is
// needed here the way `scoreBarFraction` does for R2 vs ROC AUC.
// Clamped anyway since Monte Carlo noise can occasionally push a raw
// contribution (and so its share) very slightly negative.
function contributionBarFraction(row) {
  return Math.max(0, Math.min(1, row.share));
}

function contributionCellHtml(row) {
  const pct = (row.share * 100).toFixed(1);
  const barPct = (contributionBarFraction(row) * 100).toFixed(1);
  return `
    <div class="score-cell">
      <span class="score-bar-track"><span class="score-bar-fill" style="width: ${barPct}%"></span></span>
      <span class="score-text">${pct}%</span>
    </div>
  `;
}

function buildContributionTable(targetNode, rows) {
  const table = document.createElement("table");
  table.className = "rank-table";
  const caption = document.createElement("caption");
  caption.textContent = `Drivers of "${targetNode}" (${rows.length})`;
  table.appendChild(caption);

  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>Node</th><th>Share</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    // The target node's own row is its residual/unexplained variance,
    // not a driver of itself (see AttributionResult's docstring for why
    // the API still returns it rather than excluding it from the sum)
    // -- relabeled to "Other" for display only, row.node stays the real
    // name for the click handler below.
    const displayName = row.node === targetNode ? "Other" : row.node;
    tr.innerHTML = `
      <td>${escapeHtml(displayName)}</td>
      <td>${contributionCellHtml(row)}</td>
    `;
    // Row click reuses the existing plot modal (SCOPE.md build order
    // step 6 + the "Considered and set aside" note on PDP-style curves)
    // rather than a new causal-specific plot endpoint: same
    // associate.py plot cache the ranking tables above already use.
    // Only pairs the association scan actually ran are cached --
    // openPlot's existing 404 fallback message already covers a driver
    // the scan never scored against this particular target (e.g. a
    // covariate x covariate pair under a scoped scan), so this can
    // legitimately show "no cached plot" for some rows. Accepted for
    // v1 rather than adding a second plot data source for this one
    // panel.
    tr.addEventListener("click", () => openPlot(row.node, targetNode));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return table;
}

async function attributeCausalTarget(targetNode) {
  causalContributionContainer.innerHTML = "<p>Loading…</p>";
  try {
    const result = await api(`/api/causal/attribute/${encodeURIComponent(targetNode)}`);
    causalContributionContainer.innerHTML = "";
    causalContributionContainer.appendChild(buildContributionTable(targetNode, result.contributions));
  } catch (err) {
    causalContributionContainer.innerHTML = `<p>Could not load contributions: ${escapeHtml(err.message)}</p>`;
  }
}

async function buildCausalModel() {
  btnCausalBuild.disabled = true;
  btnCausalBuild.textContent = "Building…";
  try {
    const result = await api("/api/causal/build", "POST");
    renderCausalBuildResult(result);
    // Re-fetch the graph rather than trusting a stale module-level
    // copy: `dag.outcomes` isn't held anywhere on the frontend between
    // init() and now, and a node could have been removed in between.
    const graph = await api("/api/graph");
    const preferredDefault = graph.outcomes[0] || result.attributable_nodes[0];
    populateCausalTargetSelect(result.attributable_nodes, preferredDefault);
    causalAttributeControls.classList.remove("hidden");
    if (causalTargetSelect.value) {
      await attributeCausalTarget(causalTargetSelect.value);
    }
  } catch (err) {
    showBanner(`Could not build causal model: ${err.message}`);
  } finally {
    btnCausalBuild.disabled = false;
    btnCausalBuild.textContent = "Build causal model";
  }
}

function wireCausalPanel() {
  btnCausalBuild.addEventListener("click", buildCausalModel);
  btnCausalAttribute.addEventListener("click", () => {
    if (causalTargetSelect.value) attributeCausalTarget(causalTargetSelect.value);
  });
}

// -- init ------------------------------------------------------------------------

async function init() {
  try {
    const [health, graph, tables] = await Promise.all([
      api("/api/health"),
      api("/api/graph"),
      api("/api/tables"),
    ]);
    renderDatasetSummary(health);
    renderTables(tables);
    const eh = initCytoscape(graph);
    wireTopbar();
    wireCausalPanel();
    // Exposed for the Playwright smoke test (test_frontend_smoke.py) and
    // for manual debugging in the browser console -- not used by app.js
    // itself, which keeps `cy` as a plain module-level variable above.
    // `eh` (the edgehandles instance) is included so tests can inspect
    // gesture state (`eh.drawMode`, `eh.active`, `eh.targetNode`)
    // directly if a Shift+drag test ever needs to diagnose why an edge
    // did or didn't get created, rather than guessing blind.
    window.__dagshop = { cy, eh };
  } catch (err) {
    showBanner(`Failed to load workshop session: ${err.message}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
