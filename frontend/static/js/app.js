// ── DOM references ──────────────────────────────────────────────────────────
const trainingForm     = document.getElementById("trainingForm");
const resultImage      = document.getElementById("resultImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");

// ── Config sync ──────────────────────────────────────────────────────────────
function syncConfigDisplay() {
  const cfg = gatherPayload();
  document.querySelectorAll("[data-config-mirror]").forEach((el) => {
    const key = el.dataset.configMirror;
    if (key in cfg) el.textContent = cfg[key];
  });
}
trainingForm.addEventListener("input",  syncConfigDisplay);
trainingForm.addEventListener("change", syncConfigDisplay);

// ── Helpers ──────────────────────────────────────────────────────────────────

function showResultImageIfAvailable() {
  if (!resultImage || !imagePlaceholder || !resultImage.getAttribute("src")) return;
  const reveal = () => {
    imagePlaceholder.style.display = "none";
    resultImage.style.display = "block";
  };
  if (resultImage.complete && resultImage.naturalWidth > 0) reveal();
  else resultImage.addEventListener("load", reveal, { once: true });
}

function gatherPayload() {
  return {
    episodes:        parseInt(document.getElementById("episodes").value),
    nodes:           parseInt(document.getElementById("nodes").value),
    model_type:      document.getElementById("model_type").value,
    learning_rate:   parseFloat(document.getElementById("lr").value),
    gamma:           parseFloat(document.getElementById("gamma").value),
    batch_size:      parseInt(document.getElementById("batch_size").value),
    death_threshold: parseFloat(document.getElementById("death_threshold").value) / 100.0,
    max_steps:       parseInt(document.getElementById("max_steps").value),
    seed:            parseInt(document.getElementById("seed").value),
  };
}

/**
 * Populate the metrics KPI grid after a successful training run.
 * Reads from the new schema: data.metrics.{final_coverage, final_avg_soh,
 * network_lifetime, mean_reward}. Falls back to top-level keys for
 * backward-compat with any older response shape.
 */
function applyResult(data) {
  const metricsGrid = document.getElementById("metricsGrid");
  const met = data.metrics ?? {};

  const finalCoverage     = met.final_coverage     ?? null;
  const finalAvgSoH       = met.final_avg_soh       ?? null;
  const networkLifetime   = met.network_lifetime    ?? null;
  const meanReward        = met.mean_reward         ?? data.mean_reward ?? null;

  if (data.image_url) {
    resultImage.src = `${data.image_url}?t=${Date.now()}`;
    resultImage.onload = () => {
      imagePlaceholder.style.display = "none";
      resultImage.style.display = "block";
    };
  } else {
    showResultImageIfAvailable();
  }

  const fmtPct   = (v) => (v !== null && Number.isFinite(Number(v)))
    ? `${(Number(v) * 100).toFixed(1)}%` : "—";
  const fmtNum   = (v) => (v !== null && Number.isFinite(Number(v)))
    ? Number(v).toFixed(2) : "—";
  const fmtInt   = (v) => (v !== null && Number.isFinite(Number(v)))
    ? String(Math.round(Number(v))) : "—";

  document.getElementById("valFinalCoverage").textContent   = fmtPct(finalCoverage);
  document.getElementById("valAvgSoH").textContent          = fmtPct(finalAvgSoH);
  document.getElementById("valNetworkLifetime").textContent = fmtInt(networkLifetime);
  document.getElementById("valMeanReward").textContent      = fmtNum(meanReward);

  metricsGrid.style.display = "grid";
}

// ── Tab switching ────────────────────────────────────────────────────────────

function switchTab(tab) {
  const panels = { current: "panelCurrent", history: "panelHistory", compare: "panelCompare" };
  const tabs   = { current: "tabCurrent",   history: "tabHistory",   compare: "tabCompare"   };

  Object.keys(panels).forEach((key) => {
    const panel = document.getElementById(panels[key]);
    const btn   = document.getElementById(tabs[key]);
    if (key === tab) {
      panel.style.display = "flex";
      btn.classList.add("active-tab");
    } else {
      panel.style.display = "none";
      btn.classList.remove("active-tab");
    }
  });

  if (tab === "history") fetchHistory();
  if (tab === "compare") loadCompareRuns();
}

// ── History: shared helpers ──────────────────────────────────────────────────

function fmt(val, decimals = 2) {
  if (val === null || val === undefined) return "N/A";
  const n = Number(val);
  return Number.isFinite(n) ? n.toFixed(decimals) : "N/A";
}

function fmtPct(val) {
  if (val === null || val === undefined) return "N/A";
  const n = Number(val);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "N/A";
}

function formatTimestamp(isoString) {
  try {
    return new Date(isoString).toLocaleString(undefined, {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch (_) {
    return isoString;
  }
}

function buildKVRow(label, value) {
  const div   = document.createElement("div");
  div.className = "history-kv";
  const lspan = document.createElement("span");
  lspan.className = "history-kv-label";
  lspan.textContent = label;
  const vspan = document.createElement("span");
  vspan.className = "history-kv-value";
  vspan.textContent = String(value);
  div.appendChild(lspan);
  div.appendChild(vspan);
  return div;
}

function buildSection(title, rows) {
  const wrap = document.createElement("div");
  const h    = document.createElement("p");
  h.className = "text-[10px] font-semibold text-on-surface-variant uppercase tracking-widest mb-2";
  h.textContent = title;
  wrap.appendChild(h);
  rows.forEach(([k, v]) => wrap.appendChild(buildKVRow(k, v)));
  return wrap;
}

/**
 * Normalise a run dict to a consistent shape, handling both the old schema
 * (config sub-object) and the new Phase-3 top-level schema.
 */
function normalizeRun(run) {
  const cfg = run.config ?? {};
  return {
    run_id:          run.run_id,
    timestamp:       run.timestamp,
    model_used:      run.model_used      ?? cfg.model_type      ?? "?",
    episodes:        run.episodes        ?? cfg.episodes,
    num_nodes:       run.num_nodes       ?? cfg.nodes,
    learning_rate:   run.learning_rate   ?? cfg.learning_rate,
    gamma:           run.gamma           ?? cfg.gamma,
    batch_size:      run.batch_size      ?? cfg.batch_size,
    death_threshold: run.death_threshold ?? cfg.death_threshold,
    max_steps:       run.max             ?? cfg.max_steps,
    seed:            run.seed            ?? cfg.seed,
    metrics:         run.metrics         ?? {},
    image_url:       run.image_url,
  };
}

// ── History card builder ─────────────────────────────────────────────────────

function buildHistoryCard(run) {
  const r   = normalizeRun(run);
  const met = r.metrics;

  const card = document.createElement("div");
  card.className = "history-card bg-surface-container-low rounded-xl border border-outline-variant/10 overflow-hidden";

  // ── Header ──
  const header = document.createElement("div");
  header.className = "px-5 py-3.5 flex items-center justify-between border-b border-outline-variant/10 bg-surface-container";

  const headLeft = document.createElement("div");
  const tsEl = document.createElement("p");
  tsEl.className = "text-[10px] text-on-surface-variant";
  tsEl.textContent = formatTimestamp(r.timestamp);
  const idEl = document.createElement("h3");
  idEl.className = "font-headline font-bold text-sm text-on-surface mt-0.5";
  idEl.textContent = r.run_id ?? "";
  headLeft.appendChild(tsEl);
  headLeft.appendChild(idEl);

  const badge = document.createElement("span");
  badge.className = "history-badge";
  badge.textContent = r.model_used.toUpperCase();

  header.appendChild(headLeft);
  header.appendChild(badge);
  card.appendChild(header);

  // ── 3-column body ──
  const body = document.createElement("div");
  body.className = "p-5 grid grid-cols-1 md:grid-cols-3 gap-5";

  const cfgVal = (v) => (v === null || v === undefined) ? "N/A" : String(v);
  const deathPct = r.death_threshold != null
    ? `${(r.death_threshold * 100).toFixed(0)}%` : "N/A";

  body.appendChild(buildSection("Configuration", [
    ["Model",         r.model_used.toUpperCase()],
    ["Episodes",      cfgVal(r.episodes)],
    ["Nodes",         cfgVal(r.num_nodes)],
    ["Learning Rate", cfgVal(r.learning_rate)],
    ["Gamma",         cfgVal(r.gamma)],
    ["Max Steps",     cfgVal(r.max_steps)],
    ["Death Thresh.", deathPct],
    ["Seed",          cfgVal(r.seed)],
  ]));

  body.appendChild(buildSection("Metrics", [
    ["Mean Reward",       fmt(met.mean_reward)],
    ["Max Reward",        fmt(met.max_reward)],
    ["Final Coverage",    fmtPct(met.final_coverage)],
    ["Avg SoH",           fmtPct(met.final_avg_soh)],
    ["Network Lifetime",  met.network_lifetime != null ? `${met.network_lifetime} ep` : "N/A"],
    ["Best Episode",      cfgVal(met.best_episode)],
    ["Avg Last 10",       fmt(met.avg_final_10)],
  ]));

  // Plot column
  const plotWrap  = document.createElement("div");
  const plotTitle = document.createElement("p");
  plotTitle.className = "text-[10px] font-semibold text-on-surface-variant uppercase tracking-widest mb-2";
  plotTitle.textContent = "Training Dashboard";
  plotWrap.appendChild(plotTitle);

  if (r.image_url) {
    const img = document.createElement("img");
    img.src   = `${r.image_url}?t=1`;
    img.alt   = `Training dashboard for ${r.run_id}`;
    img.className = "history-card-plot result-fade";
    img.addEventListener("error", () => { img.style.display = "none"; });
    plotWrap.appendChild(img);
  } else {
    const noPlot = document.createElement("div");
    noPlot.className = "flex items-center justify-center h-24 text-on-surface-variant opacity-30 text-xs";
    noPlot.textContent = "No plot available";
    plotWrap.appendChild(noPlot);
  }
  body.appendChild(plotWrap);

  card.appendChild(body);
  return card;
}

// ── History fetch ────────────────────────────────────────────────────────────

// Shared cache so Compare tab can reuse without a second network request
let _historyCache = [];

async function fetchHistory() {
  const historyList = document.getElementById("historyList");
  if (!historyList) return;

  historyList.textContent = "";
  const loadingDiv = document.createElement("div");
  loadingDiv.className = "flex items-center justify-center gap-2 py-12 text-on-surface-variant opacity-50";
  const spinner = document.createElement("div");
  spinner.className = "loader";
  spinner.style.cssText = "display:block; border-top-color:#7bd0ff;";
  const loadLabel = document.createElement("span");
  loadLabel.className = "text-sm";
  loadLabel.textContent = "Loading history\u2026";
  loadingDiv.appendChild(spinner);
  loadingDiv.appendChild(loadLabel);
  historyList.appendChild(loadingDiv);

  try {
    const res  = await fetch("/api/history");
    const runs = await res.json();
    _historyCache = Array.isArray(runs) ? runs : [];
    historyList.textContent = "";

    if (_historyCache.length === 0) {
      const empty = document.createElement("div");
      empty.className = "flex flex-col items-center justify-center gap-3 py-16 text-on-surface-variant opacity-50";
      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined text-5xl";
      icon.textContent = "history";
      const msg = document.createElement("p");
      msg.className = "text-sm";
      msg.textContent = "No training runs yet.";
      empty.appendChild(icon);
      empty.appendChild(msg);
      historyList.appendChild(empty);
      return;
    }

    _historyCache.forEach((run) => historyList.appendChild(buildHistoryCard(run)));
  } catch (err) {
    historyList.textContent = "";
    const errDiv = document.createElement("div");
    errDiv.className = "flex flex-col items-center justify-center gap-3 py-16 text-on-surface-variant";
    const errIcon = document.createElement("span");
    errIcon.className = "material-symbols-outlined text-4xl text-error";
    errIcon.textContent = "error";
    const errMsg = document.createElement("p");
    errMsg.className = "text-sm text-error";
    errMsg.textContent = `Failed to load history: ${err.message}`;
    errDiv.appendChild(errIcon);
    errDiv.appendChild(errMsg);
    historyList.appendChild(errDiv);
  }
}

// ── Compare panel ────────────────────────────────────────────────────────────

async function loadCompareRuns() {
  // Re-fetch history if cache is stale (empty)
  if (_historyCache.length === 0) {
    try {
      const res  = await fetch("/api/history");
      const runs = await res.json();
      _historyCache = Array.isArray(runs) ? runs : [];
    } catch (_) {
      return;
    }
  }

  const selA = document.getElementById("compareRunA");
  const selB = document.getElementById("compareRunB");

  // Rebuild options
  [selA, selB].forEach((sel) => {
    // Keep the first placeholder option
    while (sel.options.length > 1) sel.remove(1);
    _historyCache.forEach((run) => {
      const r   = normalizeRun(run);
      const opt = document.createElement("option");
      opt.value       = r.run_id;
      opt.textContent = `${r.model_used.toUpperCase()} — ${r.run_id} (${r.episodes ?? "?"} ep)`;
      sel.appendChild(opt);
    });
  });
}

async function runComparison() {
  const runIdA = document.getElementById("compareRunA").value;
  const runIdB = document.getElementById("compareRunB").value;

  const statusEl     = document.getElementById("compareStatus");
  const btn          = document.getElementById("compareBtn");
  const btnText      = document.getElementById("compareBtnText");
  const loader       = document.getElementById("compareLoader");
  const compareImg   = document.getElementById("compareImage");
  const placeholder  = document.getElementById("comparePlaceholder");

  statusEl.style.display = "none";
  statusEl.className = "status-message mx-5 mb-4";

  if (!runIdA || !runIdB) {
    statusEl.textContent = "Please select both Run A and Run B.";
    statusEl.classList.add("status-error");
    statusEl.style.display = "block";
    return;
  }
  if (runIdA === runIdB) {
    statusEl.textContent = "Select two different runs.";
    statusEl.classList.add("status-error");
    statusEl.style.display = "block";
    return;
  }

  btn.disabled = true;
  btnText.textContent = "Generating\u2026";
  loader.style.display = "block";

  try {
    const url = `/api/compare?a=${encodeURIComponent(runIdA)}&b=${encodeURIComponent(runIdB)}`;
    const res  = await fetch(url);
    const data = await res.json();

    if (!res.ok || data.status !== "success") {
      throw new Error(data.message ?? "Comparison failed");
    }

    compareImg.src = `${data.image_url}?t=${Date.now()}`;
    compareImg.onload = () => {
      placeholder.style.display = "none";
      compareImg.style.display  = "block";
    };

    statusEl.textContent = "Comparison generated successfully.";
    statusEl.classList.add("status-success");
    statusEl.style.display = "block";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.classList.add("status-error");
    statusEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btnText.textContent = "Generate";
    loader.style.display = "none";
  }
}

// ── Async training poll ──────────────────────────────────────────────────────

async function pollTask(taskId, statusMessage, buttonText, loader, runButton) {
  const POLL_INTERVAL_MS = 2000;

  while (true) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));

    const res  = await fetch(`/api/tasks/${taskId}`);
    const task = await res.json();

    if (task.status === "completed") {
      statusMessage.textContent =
        task.result.message ||
        `Training completed. Mean reward: ${Number(task.result.metrics?.mean_reward ?? task.result.mean_reward).toFixed(2)}`;
      statusMessage.classList.add("status-success");
      statusMessage.style.display = "block";
      applyResult(task.result);
      break;
    }

    if (task.status === "failed") {
      throw new Error(task.error || "Training job failed on server");
    }

    buttonText.textContent = task.status === "running" ? "Training\u2026" : "Queued\u2026";
  }

  runButton.disabled = false;
  buttonText.textContent = "Start Training";
  loader.style.display = "none";
}

// ── Form submit ──────────────────────────────────────────────────────────────

trainingForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const runButton     = document.getElementById("runBtn");
  const buttonText    = document.getElementById("runBtnText");
  const loader        = document.getElementById("loader");
  const statusMessage = document.getElementById("statusMessage");

  runButton.disabled = true;
  buttonText.textContent = "Submitting\u2026";
  loader.style.display = "block";
  statusMessage.style.display = "none";
  statusMessage.className = "status-message";

  try {
    const submitRes = await fetch("/api/train/async", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(gatherPayload()),
    });

    const submitData = await submitRes.json();

    if (!submitRes.ok || submitData.status !== "queued") {
      throw new Error(
        JSON.stringify(submitData.errors ?? submitData.message ?? "Submission failed")
      );
    }

    buttonText.textContent = "Queued\u2026";
    await pollTask(submitData.task_id, statusMessage, buttonText, loader, runButton);

    // Silently pre-load history and compare selects for the new run
    fetchHistory();
  } catch (err) {
    statusMessage.textContent = `Error: ${err.message}`;
    statusMessage.classList.add("status-error");
    statusMessage.style.display = "block";
    runButton.disabled = false;
    buttonText.textContent = "Start Training";
    loader.style.display = "none";
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────

showResultImageIfAvailable();
syncConfigDisplay();
fetchHistory();
