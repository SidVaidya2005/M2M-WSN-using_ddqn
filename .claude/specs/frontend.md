# Frontend Spec

## Current Architecture

Single-page app served by Flask (`frontend/templates/index.html` + `frontend/static/`).

**No build step.** Tailwind CSS loaded from CDN. DOMPurify CDN-loaded for XSS safety. `app.js` served as-is — no bundler, no transpilation.

```
frontend/
├── templates/index.html     # Single HTML page
└── static/
    ├── js/app.js            # All client-side logic
    └── css/style.css        # Custom styles on top of Tailwind
```

## Key JS Functions

| Function | Purpose |
|----------|---------|
| `gatherPayload()` | Builds POST `/api/train` request body from form fields. `death_threshold` converted from percentage (0-100) to ratio |
| `applyResult(data)` | Populates metric cards from training response |
| `switchTab(tab)` | Toggles between Current and History panels |
| `buildHistoryCard()` | Renders a run card in the history list |
| `syncConfigDisplay()` | Mirrors form values to `data-config-mirror` elements |

## Current State (Known Issues — Phase 4 Targets)

### Things That Need Removal
- **`_benchmarkTasks` Map** — tracks `/api/evaluate` jobs (endpoint removed)
- **`buildBenchmarkTable()`** — renders baseline comparison (baselines removed)
- **"Run Benchmark" button** on history cards — calls dead `/api/evaluate` endpoint
- All references to baselines / evaluation in HTML and JS

### Things That Need Addition (Phase 4)
- **DDQN vs DQN comparison card** — pick two runs (one DDQN, one DQN), renders side-by-side plot
- **New metrics grid** — show `final_coverage`, `final_avg_soh`, `network_lifetime`, `mean_reward` instead of current metrics
- **Updated history cards** — display `model_used`, `num_nodes`, `episodes`, `final_coverage`, `network_lifetime`
- **4-panel image rendering** — the new training visualization will be 2×2 subplot

## UI Layout

Two-tab right panel:
- **Current** (`panelCurrent`) — training result image, metrics grid, training status
- **History** (`panelHistory`) — rendered from `GET /api/history`; runs displayed newest-first

## Styling Notes

- Tailwind config is inline in `index.html` (`tailwind.config = { ... }` block)
- Custom semantic color tokens (`bg-surface`, `text-on-surface`) defined in Tailwind config
- `resultImage.src` gets `?t=Date.now()` appended for cache-busting
- **DOMPurify required** — always sanitize before `innerHTML` assignments
