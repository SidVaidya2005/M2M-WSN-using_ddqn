# WSN_M2M Restructure Plan

## Context

The WSN RL platform currently mixes learning agents (DDQN, DQN) with 4 baseline
policies, hardcodes `nodes=550` in the backend schema (while config.yaml uses 50),
produces only a single reward-curve plot, and lacks the M2M-specific node behaviors
the project actually needs: per-node coverage handling, charging status, SoH tracking
that influences decisions, and cooperative wake-up when a neighbor's battery drops
to 50%. Legacy files from prior phases (research paper template, migration script,
CLI report, archived results) clutter the repo.

This plan restructures the codebase around **two agents only (DDQN + DQN)**, adds
the missing M2M node logic, upgrades the training output contract (JSON schema + 4-panel
visualization), rewrites the frontend to match, removes dead code, and reorganizes
the directory layout. Post-training, DDQN vs DQN comparison graphs are produced
automatically from saved metrics.

---

## Part 1 — Logical Design of DDQN & DQN in this System

### Shared Foundation (both agents)

**Role.** A single centralized agent observes all N nodes and outputs one binary
action per node: `SLEEP(0)` or `AWAKE(1)`. Training is episodic; an episode ends
when the fraction of dead nodes crosses `death_threshold` or `max_steps` is reached.

**Observation (per node, 6 features — expanded from current 5):**

| Idx | Feature | Range | Purpose |
|---|---|---|---|
| 0 | SoC (normalized) | [0,1] | Energy available now |
| 1 | SoH | [0,1] | Long-term battery health |
| 2 | last_action | {0,1} | Hysteresis / duty-cycle signal |
| 3 | distance_to_sink (norm) | [0,1] | Communication cost proxy |
| 4 | activity_ratio (EMA) | [0,1] | Fairness / load signal |
| 5 | **charging_flag** *(new)* | {0,1} | 1 if node is currently on charge |

`state_dim = N * 6`. The agent network is node-count-agnostic: input is `N*6`,
output is `N*2` (Q-values per node per action), reshaped to `(N, 2)` before
argmax. Node count flows purely from config — never hardcoded.

**Action selection.**
- Training: epsilon-greedy, per node independently, over the (N,2) Q-table.
- Evaluation: pure argmax (eps=0).
- **M2M wake-up override (post-selection, environment-side):** after the agent
  emits its action vector, the environment applies a *deterministic cooperative
  rule* — for any node whose SoC drops to ≤50%, the nearest SLEEP neighbor is
  forcibly switched to AWAKE for the next step. This preserves coverage without
  the agent having to learn the handoff from scratch, and is logged in `info`
  so the reward reflects the post-override configuration.

**Reward (unchanged structure, weights moved to config):**
```
reward = w_cov * r_coverage      # fraction awake & covering sink
       + w_eng * r_energy         # negative drain penalty
       + w_soh * r_soh            # SoH preservation
       + w_bal * r_balance        # fairness across nodes
```
Weights live in `config.yaml` under `environment.reward_weights` so they are
tunable without code changes.

**Replay & target update (shared).**
- Circular replay buffer of size `buffer_size`.
- `min_replay_size` transitions collected before `learn_step()` returns a loss.
- Target network hard-updated every `target_update_frequency` steps.
- Gradient clipping at norm 10.0.

### DDQN — Primary Agent

**Bellman target (decoupled action selection/evaluation):**
```
a*       = argmax_a Q_online(s', a)          # online picks action
y        = r + γ * Q_target(s', a*) * (1-done)   # target evaluates it
```
This reduces the maximization bias DQN suffers from, giving more stable
learning on the noisy, multi-objective WSN reward. This is our **primary**
model and the default for all API calls.

### DQN — Comparison Agent

**Bellman target (single network max):**
```
y = r + γ * max_a Q_target(s', a) * (1-done)
```
Implemented as a subclass of DDQN that overrides only the target computation
in `learn_step()`. Kept solely so the report can plot DDQN-vs-DQN learning
curves on identical seeds and environments — the academic comparison is the
only reason DQN exists in this codebase.

**All baseline policies (Random, Greedy, EnergyConservative, BalancedRotation)
are removed.** The user asked for only two agents; benchmarking is DDQN-vs-DQN.

---

## Part 2 — Implementation Plan (Phased, confirmation gate between phases)

> Rule: at the end of each phase, stop and wait for user confirmation before
> starting the next phase.

### Phase 0 — Cleanup & Directory Restructure  ✅ COMPLETE

**Goal:** shrink the repo to only what's needed and make the layout obvious.

**Deleted:**
- `src/baselines/` (entire package)
- `tests/test_baselines.py`
- `scripts/evaluate_baselines.py`
- `scripts/generate_report.py`
- `scripts/migrate_legacy_runs.py`
- `Research_Paper.md`
- `results/archive/`
- All `.DS_Store` files
- `save_gif` option from `config.yaml` and `config/settings.py` (never implemented)

**Edited:**
- `backend/tasks.py` — dropped baseline imports, `run_baseline_benchmark`,
  `submit_benchmark_task`, `_run_benchmark_background`, `_run_policy_episodes`
- `backend/routes.py` — removed `/api/evaluate` route, `_benchmark_schema`,
  and the inline eval injection loop in `/api/history`
- `backend/schemas.py` — removed `EvaluationRequestSchema`
- `src/__init__.py` — removed baseline re-exports
- `config/config.yaml` + `config/settings.py` — removed `save_gif` field
- `setup.py` — removed `wsn-eval` entry point, repointed `wsn-train` at
  `scripts.train`
- Renamed `scripts/train_model.py` → `scripts/train.py`

**Verification:** `python3 -m compileall -q backend src scripts config tests`
returns clean. Full `pytest tests/` and `python -m backend.app` boot need the
venv and will be rerun once any edits touch runtime behavior.

**Known residual (for Phase 1/4):**
- `scripts/train.py` docstring still shows a `--nodes 550` example.
- `frontend/static/js/app.js` still calls `/api/evaluate` (the button will
  404 until Phase 4 rewrites the UI). Training flow unaffected.

---

### Phase 1 — Config & Hardcode Elimination  ✅ COMPLETE

**Goal:** every tunable lives in `config.yaml`; no literal `550` anywhere.

**Edited:**
- `config/config.yaml` — added three new sub-sections under `environment`:
  `reward_weights` (coverage=10, energy=5, soh=1, balance=2),
  `charging` (enabled=true, rate=0.05, threshold=0.2),
  `wake_cooperation` (low_battery_soc=0.5).
- `config/settings.py` — added `RewardWeightsConfig`, `ChargingConfig`,
  `WakeCooperationConfig` dataclasses; `EnvironmentConfig` now holds them as
  typed fields; `Config.load()` constructs them from nested YAML dicts;
  `to_dict()` serialises them properly.
- `backend/schemas.py` — `nodes` default changed from `550` → `None`
  (with `allow_none=True`); actual default resolved at call time from config.
- `backend/tasks.py` — `nodes` resolved as
  `params.get("nodes") or config.environment.num_nodes`, so an empty POST body
  picks up `num_nodes=50` from config.
- `scripts/train.py` — docstring updated from `--nodes 550` to `--nodes 50`.

**Verification:** `grep 550` across all `.py` and `.yaml` files returns zero
hits. `python3 -m compileall -q config/ backend/schemas.py backend/tasks.py
scripts/train.py` returns clean.

---

### Phase 2 — Environment: Coverage, Charging, SoH, Cooperative Wake-Up  ✅ COMPLETE

**Goal:** implement the per-node behaviors the user asked for.

**Edited:**
- `config/config.yaml` — added `environment.sensing_radius: 100.0`.
- `config/settings.py` — added `sensing_radius: float` to `EnvironmentConfig`;
  updated `Config.load()`, `from_dict()`, and `to_dict()` to include it.
- `src/envs/battery_model.py` — added `charging: bool = False` state;
  rewrote `charge(rate)` to accept a fraction of E_max and apply calendar SoH
  decay while charging; added `is_charging` property and `needs_charge(threshold)`
  helper; `reset_to_health()` now clears `charging`.
- `src/envs/wsn_env.py` — fully rewritten:
  - Observation grows to **6 features per node** (added `charging_flag` at index 5);
    `observation_space` shape is now `(N*6,)`.
  - Constructor accepts `reward_weights`, `charging_enabled`, `charging_rate`,
    `charging_threshold`, `wake_cooperation_soc`, `sensing_radius` with config-
    matching defaults.
  - `step()` applies charging override → cooperative wake-up → physics in that order.
  - **Charging:** node enters `charging=True` when `soc/E_max < charging_threshold`;
    exits when `soc/E_max ≥ 0.95`. Forced SLEEP; `battery.charge(rate)` called.
  - **Cooperative wake-up:** for each AWAKE node whose `soc/E_max ≤ wake_cooperation_soc`,
    the nearest non-charging non-dead SLEEP neighbour is forced AWAKE (deduped).
    Woken node IDs logged in `info["cooperative_wakes"]`.
  - **Coverage:** 20×20 grid-point sampling with `sensing_radius`; replaces the
    old "fraction awake" scalar.
  - **Reward weights** pulled from instance vars (set from config by callers).
  - **Info dict** now includes: `coverage`, `avg_soh`, `alive_fraction`, `dead_count`,
    `mean_soc`, `cooperative_wakes`, `charging_count`, `step_count` (plus
    backward-compat aliases `coverage_ratio` and `dead_nodes`).
- `backend/tasks.py` — updated `WSNEnv()` call to pass all new params from config.
- `scripts/train.py` — same.
- `tests/conftest.py` — `STATE_DIM` updated from `N*5` to `N*6`.
- `tests/test_env.py` — fully rewritten: fixed stale reset-tuple tests and
  broken BatteryModel tests; added 19 new tests covering 6-feature observation,
  coverage metric, charging entry/exit/recovery, and cooperative wake-up.

**Verification:** `pytest tests/ -v` → 55/55 passed.

---

### Phase 3 — Training Output Contract (JSON + 4-Panel Visualization) ✅ COMPLETE

**Goal:** training produces the exact JSON and the exact plot the user asked for.

**`backend/tasks.py` — new metadata schema:**
```json
{
  "run_id": "run_YYYYMMDD_HHMMSS",
  "timestamp": "<ISO-8601>",
  "model_used": "ddqn" | "dqn",
  "episodes": 100,
  "num_nodes": 50,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "death_threshold": 0.3,
  "max": 1000,
  "seed": 42,
  "metrics": {
    "mean_reward": ...,
    "max_reward": ...,
    "best_episode": ...,
    "avg_final_10": ...,
    "final_coverage": ...,
    "final_avg_soh": ...,
    "network_lifetime": ...
  },
  "series": {
    "episode_reward": [...],
    "coverage": [...],
    "avg_soh": [...],
    "alive_fraction": [...],
    "mean_soc": [...]
  },
  "image_url": "/api/visualizations/<run_id>_plot.png",
  "model_path": "..."
}
```
The fields in the top-level block (`model_used`, `episodes`, `num_nodes`,
`learning_rate`, `gamma`, `death_threshold`, `max`, `seed`) exactly match the
user's list. `max` aliases `max_steps` per the user's spec.

**`src/training/trainer.py`:** collect per-episode `coverage`, `avg_soh`,
`alive_fraction`, `mean_soc` from `info` and expose them via
`trainer.episode_series`. Compute `network_lifetime` = episode index at which
`alive_fraction` first drops below `1 - death_threshold` (or total episodes if
never).

**`src/utils/visualization.py` — rewrite `plot_training_curve` into
`plot_training_dashboard`:** a 2×2 figure saved as `{run_id}_plot.png`:

1. **Network Coverage** over episodes (line + 10-ep moving average)
2. **Battery Health (avg SoH)** over episodes
3. **Network Lifetime** — bar of per-episode steps-until-first-death, plus
   a horizontal line at the overall lifetime
4. **Reward & Mean SoC** — dual-axis line plot (reward left, mean SoC right)

Matches the user's "4 graphs" requirement and keeps the reward curve discoverable.

**Verification:**
- Run a 20-episode DDQN job via the web UI; inspect `{run_id}_metadata.json`
  — all required top-level keys present; inspect `{run_id}_plot.png` — 4 panels.
- `tests/test_backend.py::test_train` asserts on the new schema.

---

### Phase 4 — Frontend Alignment ✅ COMPLETE

**Goal:** the single-page UI matches the new backend contract and removes
every mention of baselines.

**`frontend/templates/index.html` + `frontend/static/js/app.js`:**
- Training form: keep `learning_rate` default at `1e-4`; refresh tooltip help
  text for the new metrics and field ranges.
- Result panel: replace the "benchmarks" section (which called `/api/evaluate`
  against baselines) with a **"DDQN vs DQN comparison"** card that lets the
  user pick two historical runs (one DDQN, one DQN) and renders a
  side-by-side plot (served by a new `GET /api/compare?a=<run_id>&b=<run_id>`
  endpoint that produces a 2×2 comparison PNG on demand).
- Metrics grid: show `final_coverage`, `final_avg_soh`, `network_lifetime`,
  `mean_reward` (instead of the current "best lifetime / best episode / mean
  reward / avg final 10").
- History panel: each card displays `model_used`, `num_nodes`, `episodes`,
  `final_coverage`, `network_lifetime`. Remove benchmark-fetch logic from
  `app.js` (`_benchmarkTasks` Map, `buildBenchmarkTable`).
- `app.js::gatherPayload()`: keep `death_threshold` percent→ratio conversion.
  Keep `max_steps` on the wire; the JSON metadata uses `max` as the user
  requested — the backend is responsible for the aliasing.

**Verification:** Open `localhost:5001` in a browser, train a 10-episode run,
confirm the 4-panel image renders, confirm the comparison card lets you pick
two runs and produces a combined plot.

---

### Phase 5 — Scripts & Post-Training Comparison ✅ COMPLETE

**Goal:** a clean CLI path and an automated DDQN-vs-DQN comparison report.

**`scripts/train.py`:** thin wrapper around `backend.tasks.run_training` so the
CLI and API produce identical artifacts. Args mirror the new JSON fields.

**`scripts/compare.py` (new):** takes two run IDs (or auto-picks the most
recent DDQN and DQN runs), loads their metadata, and produces:
- `results/visualizations/compare_{ddqn_id}_vs_{dqn_id}.png` — 2×2 comparison
  (coverage, SoH, lifetime, reward) with both agents overlaid.
- A short printed summary (mean reward, final coverage, lifetime) for each.

**Verification:** run `python scripts/train.py --model-type ddqn --episodes 30`
and `python scripts/train.py --model-type dqn --episodes 30`, then
`python scripts/compare.py` — confirm the comparison PNG exists and is valid.

---

### Phase 6 — Tests, Docs, Final Pass

**Goal:** lock the new behavior in and document the restructure.

- Update `tests/test_agent.py` for the new `state_dim = N*6`.
- Update `tests/test_env.py` for coverage/charging/wake-up behavior.
- Update `tests/test_backend.py` for the new metadata schema and removal of
  `/api/evaluate`.
- Update `.claude/rules/*.md` to match: `rl-environment.md` (6 features,
  reward weights from config, cooperative wake rule), `artifacts.md`
  (new JSON schema, no `evaluation.json`), `api-design.md` (no `/api/evaluate`).
- Update `backend/CLAUDE.md` to drop the `/api/evaluate` row and baseline
  references.
- Update `README.md` and top-level `CLAUDE.md` commands section (remove
  `evaluate_baselines.py`, `generate_report.py`, `migrate_legacy_runs.py`;
  add `compare.py`).

**Verification:** full `pytest tests/` green; manual end-to-end run (train DDQN →
train DQN → compare → history view) works in the browser.

---

## Critical Files (cross-phase)

| File | Phase(s) | Why |
|---|---|---|
| `config/config.yaml` | 0,1,2 | Single source of truth; new reward weights, charging, wake rule |
| `config/settings.py` | 1,2 | Dataclasses extended for new config sections |
| `src/envs/wsn_env.py` | 2,3 | Coverage fn, charging loop, cooperative wake, richer `info` |
| `src/envs/battery_model.py` | 2 | Charging state & method |
| `src/agents/ddqn_agent.py` | 1 | State dim now 6·N; pulled from env, not hardcoded |
| `src/agents/dqn_agent.py` | 1 | Same |
| `src/training/trainer.py` | 3 | Collect per-episode series; compute lifetime |
| `src/utils/visualization.py` | 3,5 | New 4-panel dashboard + compare plot |
| `backend/tasks.py` | 0,3 | Remove baselines, new metadata schema |
| `backend/routes.py` | 0,4 | Remove `/api/evaluate`, add `/api/compare` |
| `backend/schemas.py` | 0,1 | Remove EvaluationRequestSchema; defaults from config |
| `frontend/templates/index.html` | 4 | New metrics + comparison card |
| `frontend/static/js/app.js` | 4 | Remove benchmark logic, add compare flow |
| `scripts/train.py` | 0,5 | Renamed, defaults from config |
| `scripts/compare.py` | 5 | New comparison report |
| `tests/*` | 6 | Updated assertions |

---

## End-to-End Verification (after all phases)

1. `pytest tests/` — all green.
2. `python -m backend.app` — boots on `localhost:5001`.
3. Browser: train DDQN 30 episodes → confirm 4-panel image, metadata JSON has
   all 8 top-level fields (`model_used`, `episodes`, `num_nodes`,
   `learning_rate`, `gamma`, `death_threshold`, `max`, `seed`).
4. Browser: train DQN 30 episodes on the same seed.
5. Browser: comparison card → pick both runs → confirm overlay plot.
6. `python scripts/compare.py` — same comparison, CLI path.
7. `grep -rn "550" src/ backend/ scripts/` — empty.
8. `grep -rn "baseline" src/ backend/ frontend/` — empty (only tests/docs may reference).
