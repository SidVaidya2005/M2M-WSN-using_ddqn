// ── DOM references ──────────────────────────────────────────────────────────
const trainingForm     = document.getElementById("trainingForm");
const resultImage      = document.getElementById("resultImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");

// ── In-flight benchmark tasks: run_id → { taskId, sectionEl } ───────────────
const _benchmarkTasks = new Map();

// ── Config sync ──────────────────────────────────────────────────────────────
// All configuration lives in the single left-panel form. This function reads
// the current form values and updates any data-config-mirror elements so the
// rest of the UI always reflects the same values. Currently there are no
// mirrored display elements (the form IS the display), but this function is
// the single authoritative hook for any future additions.
function syncConfigDisplay() {
  const cfg = gatherPayload();
  document.querySelectorAll("[data-config-mirror]").forEach((el) => {
    const key = el.dataset.configMirror;
    if (key in cfg) el.textContent = cfg[key];
  });
}

// Register sync on every input/change event in the config form
trainingForm.addEventListener("input",  syncConfigDisplay);
trainingForm.addEventListener("change", syncConfigDisplay);

// ── Helpers ─────────────────────────────────────────────────────────────────

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

function applyResult(data) {
  const metricsGrid  = document.getElementById("metricsGrid");
  const gifContainer = document.getElementById("gifContainer");

  const maxReward          = Number(data.max_reward);
  const bestLifetime       = Number(data.results?.best_lifetime ?? maxReward);
  const bestEpisode        = Number(data.results?.best_episode ?? data.episodes ?? 0);
  const meanReward         = Number(data.mean_reward);
  const avgLifetimeFinal10 = Number(data.results?.avg_lifetime_final_10 ?? meanReward);

  if (data.image_url) {
    resultImage.src = `${data.image_url}?t=${Date.now()}`;
    resultImage.onload = () => {
      imagePlaceholder.style.display = "none";
      resultImage.style.display = "block";
    };
  } else {
    showResultImageIfAvailable();
  }

  if (data.gif_url) {
    const resultGif = document.getElementById("resultGif");
    resultGif.src = `${data.gif_url}?t=${Date.now()}`;
    gifContainer.style.display = "block";
  }

  document.getElementById("valBestLifetime").textContent =
    Number.isFinite(bestLifetime) ? bestLifetime.toFixed(2) : "-";
  document.getElementById("valBestEpisode").textContent =
    Number.isFinite(bestEpisode) ? String(Math.trunc(bestEpisode)) : "-";
  document.getElementById("valAvgLifetime").textContent =
    Number.isFinite(avgLifetimeFinal10) ? avgLifetimeFinal10.toFixed(2) : "-";

  metricsGrid.style.display = "grid";
}

// ── Tab switching ────────────────────────────────────────────────────────────

function switchTab(tab) {
  const panelCurrent = document.getElementById("panelCurrent");
  const panelHistory = document.getElementById("panelHistory");
  const tabCurrent   = document.getElementById("tabCurrent");
  const tabHistory   = document.getElementById("tabHistory");

  if (tab === "current") {
    panelCurrent.style.display = "flex";
    panelHistory.style.display = "none";
    tabCurrent.classList.add("active-tab");
    tabHistory.classList.remove("active-tab");
  } else {
    panelCurrent.style.display = "none";
    panelHistory.style.display = "flex";
    tabHistory.classList.add("active-tab");
    tabCurrent.classList.remove("active-tab");
    fetchHistory();
  }
}

// ── History: shared helpers ──────────────────────────────────────────────────

function fmt(val, decimals = 2) {
  const n = Number(val);
  return Number.isFinite(n) ? n.toFixed(decimals) : "-";
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

// ── Evaluation results table ─────────────────────────────────────────────────

function buildBenchmarkTable(benchData) {
  const results  = benchData.results ?? {};
  const episodes = benchData.benchmark_episodes ?? benchData.eval_episodes ?? "?";

  // Sort: trained model first, then baselines sorted descending by mean reward
  const entries = Object.entries(results).sort(([, a], [, b]) => {
    if (a.policy_type === "trained" && b.policy_type !== "trained") return -1;
    if (b.policy_type === "trained" && a.policy_type !== "trained") return  1;
    return b.mean_reward - a.mean_reward;
  });

  const maxReward = Math.max(...entries.map(([, v]) => v.mean_reward));

  const table = document.createElement("table");
  table.className = "bench-table";

  // Header
  const thead = document.createElement("thead");
  const hrow  = document.createElement("tr");
  ["Policy", "Mean Reward", "Relative Performance"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  table.appendChild(thead);

  // Body
  const tbody = document.createElement("tbody");
  entries.forEach(([name, info]) => {
    const tr = document.createElement("tr");
    if (info.policy_type === "trained") tr.classList.add("bench-trained");

    // Policy name
    const tdName = document.createElement("td");
    if (info.policy_type === "trained") {
      const star = document.createElement("span");
      star.textContent = "\u2605 ";
      star.style.fontSize = "0.7rem";
      tdName.appendChild(star);
    }
    tdName.appendChild(document.createTextNode(name));
    tr.appendChild(tdName);

    // Mean reward
    const tdReward = document.createElement("td");
    tdReward.textContent = fmt(info.mean_reward);
    tr.appendChild(tdReward);

    // Bar
    const tdBar = document.createElement("td");
    const barWrap = document.createElement("div");
    barWrap.className = "bench-bar-wrap";
    const barFill = document.createElement("div");
    barFill.className = "bench-bar-fill";
    const pct = maxReward > 0 ? Math.max(0, (info.mean_reward / maxReward) * 100) : 0;
    barFill.style.width = `${pct.toFixed(1)}%`;
    barWrap.appendChild(barFill);
    tdBar.appendChild(barWrap);
    tr.appendChild(tdBar);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  // Wrapper with label
  const wrap = document.createElement("div");
  const label = document.createElement("p");
  label.className = "text-[10px] font-semibold text-on-surface-variant uppercase tracking-widest mb-3";
  label.textContent = `Baseline Comparison \u2014 ${episodes} episodes`;
  wrap.appendChild(label);
  wrap.appendChild(table);
  return wrap;
}

function renderBenchmarkSection(sectionEl, benchData) {
  sectionEl.textContent = "";
  sectionEl.appendChild(buildBenchmarkTable(benchData));
}

// ── Benchmark polling ────────────────────────────────────────────────────────

async function pollBenchmarkTask(taskId, runId, sectionEl, btn) {
  const INTERVAL = 2000;

  while (true) {
    await new Promise((r) => setTimeout(r, INTERVAL));

    let task;
    try {
      const res = await fetch(`/api/tasks/${taskId}`);
      task = await res.json();
    } catch (_) {
      continue; // transient network hiccup — keep polling
    }

    if (task.status === "completed") {
      _benchmarkTasks.delete(runId);
      renderBenchmarkSection(sectionEl, task.result);
      if (btn) { btn.disabled = false; btn.textContent = "Re-evaluate"; }
      return;
    }

    if (task.status === "failed") {
      _benchmarkTasks.delete(runId);
      sectionEl.textContent = "";
      const errMsg = document.createElement("p");
      errMsg.className = "text-xs text-error py-2";
      errMsg.textContent = `Benchmark failed: ${task.error ?? "unknown error"}`;
      sectionEl.appendChild(errMsg);
      if (btn) { btn.disabled = false; btn.textContent = "Retry"; }
      return;
    }
  }
}

async function triggerBenchmark(runId, sectionEl, btn) {
  if (_benchmarkTasks.has(runId)) return; // already in flight

  btn.disabled = true;
  btn.textContent = "Running\u2026";

  // Show inline spinner
  sectionEl.textContent = "";
  const spinWrap = document.createElement("div");
  spinWrap.className = "flex items-center gap-2 py-3 text-on-surface-variant opacity-60";
  const spinner = document.createElement("div");
  spinner.className = "loader";
  spinner.style.cssText = "display:block; border-top-color:#7bd0ff;";
  const spinLabel = document.createElement("span");
  spinLabel.className = "text-xs";
  spinLabel.textContent = "Running baseline comparisons\u2026";
  spinWrap.appendChild(spinner);
  spinWrap.appendChild(spinLabel);
  sectionEl.appendChild(spinWrap);

  try {
    const res  = await fetch("/api/evaluate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ run_id: runId, episodes: 10 }),
    });
    const data = await res.json();

    if (!res.ok || data.status !== "queued") {
      throw new Error(data.message ?? JSON.stringify(data.errors ?? "Submission failed"));
    }

    _benchmarkTasks.set(runId, data.task_id);
    pollBenchmarkTask(data.task_id, runId, sectionEl, btn);
  } catch (err) {
    _benchmarkTasks.delete(runId);
    sectionEl.textContent = "";
    const errMsg = document.createElement("p");
    errMsg.className = "text-xs text-error py-2";
    errMsg.textContent = `Could not start benchmark: ${err.message}`;
    sectionEl.appendChild(errMsg);
    btn.disabled = false;
    btn.textContent = "Retry";
  }
}

// ── History card builder ─────────────────────────────────────────────────────

function buildHistoryCard(run) {
  const cfg = run.config  ?? {};
  const met = run.metrics ?? {};

  const card = document.createElement("div");
  card.className = "history-card bg-surface-container-low rounded-xl border border-outline-variant/10 overflow-hidden";

  // ── Header ──
  const header = document.createElement("div");
  header.className = "px-5 py-3.5 flex items-center justify-between border-b border-outline-variant/10 bg-surface-container";

  const headLeft = document.createElement("div");
  const tsEl = document.createElement("p");
  tsEl.className = "text-[10px] text-on-surface-variant";
  tsEl.textContent = formatTimestamp(run.timestamp);
  const idEl = document.createElement("h3");
  idEl.className = "font-headline font-bold text-sm text-on-surface mt-0.5";
  idEl.textContent = run.run_id ?? "";
  headLeft.appendChild(tsEl);
  headLeft.appendChild(idEl);

  const badge = document.createElement("span");
  badge.className = "history-badge";
  badge.textContent = (cfg.model_type ?? "").toUpperCase();

  header.appendChild(headLeft);
  header.appendChild(badge);
  card.appendChild(header);

  // ── 3-column body ──
  const body = document.createElement("div");
  body.className = "p-5 grid grid-cols-1 md:grid-cols-3 gap-5";

  const deathPct = cfg.death_threshold != null
    ? `${(cfg.death_threshold * 100).toFixed(0)}%` : "-";

  body.appendChild(buildSection("Configuration", [
    ["Episodes",      cfg.episodes      ?? "-"],
    ["Nodes",         cfg.nodes         ?? "-"],
    ["Learning Rate", cfg.learning_rate ?? "-"],
    ["Gamma",         cfg.gamma         ?? "-"],
    ["Batch Size",    cfg.batch_size    ?? "-"],
    ["Max Steps",     cfg.max_steps     ?? "-"],
    ["Death Thresh.", deathPct],
    ["Seed",          cfg.seed          ?? "-"],
  ]));

  body.appendChild(buildSection("Metrics", [
    ["Max Reward",   fmt(met.max_reward)],
    ["Mean Reward",  fmt(met.mean_reward)],
    ["Best Episode", met.best_episode ?? "-"],
    ["Avg Last 10",  fmt(met.avg_final_10)],
  ]));

  // Plot column
  const plotWrap  = document.createElement("div");
  const plotTitle = document.createElement("p");
  plotTitle.className = "text-[10px] font-semibold text-on-surface-variant uppercase tracking-widest mb-2";
  plotTitle.textContent = "Training Curve";
  plotWrap.appendChild(plotTitle);

  if (run.image_url) {
    const img = document.createElement("img");
    img.src   = `${run.image_url}?t=1`;
    img.alt   = `Training curve for ${run.run_id}`;
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

  // ── Evaluation / Benchmark section ──
  const benchSection = document.createElement("div");
  benchSection.className = "bench-section";

  const benchHeader = document.createElement("div");
  benchHeader.className = "flex items-center justify-between mb-3";

  const benchTitle = document.createElement("p");
  benchTitle.className = "text-[10px] font-semibold text-on-surface-variant uppercase tracking-widest";
  benchTitle.textContent = "Baseline Benchmark";
  benchHeader.appendChild(benchTitle);

  const benchBtn = document.createElement("button");
  benchBtn.className = "bench-eval-btn";

  // icon
  const btnIcon = document.createElement("span");
  btnIcon.className = "material-symbols-outlined text-[14px]";
  btnIcon.textContent = "play_circle";
  benchBtn.appendChild(btnIcon);
  benchBtn.appendChild(document.createTextNode(
    run.evaluation ? "Re-evaluate" : "Evaluate Baselines"
  ));
  benchHeader.appendChild(benchBtn);
  benchSection.appendChild(benchHeader);

  // Results container
  const resultsContainer = document.createElement("div");
  benchSection.appendChild(resultsContainer);

  // If we already have results (inlined by /api/history), render them immediately
  if (run.evaluation) {
    renderBenchmarkSection(resultsContainer, run.evaluation);
  } else {
    const hint = document.createElement("p");
    hint.className = "text-xs text-on-surface-variant opacity-50";
    hint.textContent = "Click \u201cEvaluate Baselines\u201d to compare this model against Random, Greedy, EnergyConservative, and BalancedRotation policies.";
    resultsContainer.appendChild(hint);
  }

  benchBtn.addEventListener("click", () => {
    triggerBenchmark(run.run_id, resultsContainer, benchBtn);
  });

  card.appendChild(benchSection);
  return card;
}

// ── History fetch ────────────────────────────────────────────────────────────

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
    historyList.textContent = "";

    if (!Array.isArray(runs) || runs.length === 0) {
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

    runs.forEach((run) => historyList.appendChild(buildHistoryCard(run)));
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
        `Training completed. Mean reward: ${Number(task.result.mean_reward).toFixed(2)}`;
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
  const gifContainer  = document.getElementById("gifContainer");

  runButton.disabled = true;
  buttonText.textContent = "Submitting\u2026";
  loader.style.display = "block";
  statusMessage.style.display = "none";
  statusMessage.className = "status-message";
  gifContainer.style.display = "none";

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

    // Silently pre-load history so the new run is ready when the user switches tabs
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
