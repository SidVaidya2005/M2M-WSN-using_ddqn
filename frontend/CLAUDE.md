# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also:** [`../backend/CLAUDE.md`](../backend/CLAUDE.md) for the Flask API layer, and the root [`../CLAUDE.md`](../CLAUDE.md) for project-wide commands and architecture.

## Structure

```
frontend/
├── templates/index.html   — Single HTML page (all UI lives here)
└── static/
    ├── js/app.js          — All client-side logic (no framework, no build step)
    └── css/style.css      — Custom styles layered on top of Tailwind
```

**No build step.** Tailwind CSS is loaded from CDN (`cdn.tailwindcss.com`). DOMPurify is also CDN-loaded. There is no `package.json`, no bundler, and no transpilation — `app.js` is served as-is.

## UI Layout — Three-Tab Right Panel

The right panel has **three tabs**, toggled by `switchTab(tab)`:

| Tab id | Panel id | Purpose |
|--------|----------|---------|
| `current` | `panelCurrent` | Result image, metrics grid, training status for the most recent run |
| `history` | `panelHistory` | Rendered from `GET /api/history`; one card per run, newest first |
| `compare` | `panelCompare` | DDQN-vs-DQN comparison — pick Run A and Run B, generate a side-by-side plot |

`switchTab("history")` triggers `fetchHistory()`. `switchTab("compare")` triggers `loadCompareRuns()`, which fills the two `<select>` elements (`compareRunA` / `compareRunB`) from the `_historyCache`.

## Key Functions in `app.js`

**`gatherPayload()`** — single source of truth for the `POST /api/train` request body. All form field → API key mappings live here. Note: `death_threshold` is displayed as a percentage (0–100) but sent as a ratio (`/100.0`).

**`applyResult(data)`** — populates the metric KPI cards after a successful training response. Reads from the Phase 3 schema: `data.metrics.{final_coverage, final_avg_soh, network_lifetime, mean_reward}` and `data.image_url`. Coverage and SoH render as percentages; network lifetime as an integer episode count; mean reward to two decimals. Missing values render as `—`.

**`switchTab(tab)`** — see table above.

**`normalizeRun(run)`** — smooths over the old/new metadata schema. Pulls `model_used`, `num_nodes`, `episodes`, etc. from top-level Phase-3 fields first, falling back to `run.config.*` for pre-Phase-3 runs. Use this whenever rendering a history row.

**`loadCompareRuns()` / `runComparison()`** — compare tab flow. `runComparison()` hits `GET /api/compare?a=<id>&b=<id>`, sets `compareImage.src` from `data.image_url` (combined 2×2 plot), then populates four individual-panel `<img>` elements (`cmpCoverage`, `cmpBatteryHealth`, `cmpEnergyConsumption`, `cmpThroughput`) from `data.individual_urls` and unhides `#compareIndividualSection`. All `src` values get `?t=` cache-busters. Rejects identical or missing run IDs client-side before fetching.

**`data-config-mirror` attribute** — any element with `data-config-mirror="<key>"` has its text content synced to the corresponding payload key on every form `input`/`change` event via `syncConfigDisplay()`.

History cards are built entirely in JS via `buildHistoryCard()` / `buildSection()` / `buildKVRow()`. There is **no baseline benchmark table** and no per-card "Run Benchmark" button — baseline evaluation was removed in Phase 0, so the UI exposes only training + compare.

## Gotchas

- **DOMPurify is required** for any HTML rendered from API responses. The history panel uses `innerHTML` assignments — always sanitize with `DOMPurify.sanitize()` before setting.
- **Tailwind config is inline** in `index.html` (the `tailwind.config = { ... }` block). Custom semantic color tokens (e.g., `bg-surface`, `text-on-surface`) are defined there — not in `style.css`.
- **Image cache-busting** — `resultImage.src` and `compareImage.src` get `?t=Date.now()` appended on every new result to force the browser to re-fetch when the filename hasn't changed.
- **Field-name schema drift** — always read run data through `normalizeRun()` so old runs (which nest fields under `config`) render the same as Phase-3 runs.
- **No `/api/evaluate` calls.** If you see code referencing `/api/evaluate`, `_benchmarkTasks`, or `buildBenchmarkTable`, it is dead code from a prior iteration and should be removed.
