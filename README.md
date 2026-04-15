# WSN DDQN Training Platform

A research-grade deep reinforcement learning platform for optimizing Wireless Sensor Network (WSN) scheduling using Double Deep Q-Networks (DDQN). Trains agents that balance network lifetime, coverage, and energy efficiency — exposed via both a CLI and a Flask REST API with a web dashboard.

---

## Table of Contents

- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Architecture & Layer Rules](#architecture--layer-rules)
- [Configuration System](#configuration-system)
- [RL Core & Mathematics](#rl-core--mathematics)
- [Agents](#agents)
- [Training Loop](#training-loop)
- [Backend API Reference](#backend-api-reference)
- [Frontend Details](#frontend-details)
- [Output Artifacts & Metadata](#output-artifacts--metadata)
- [Testing](#testing)
- [CLI Reference](#cli-reference)
- [Code Quality](#code-quality)

---

## Key Concepts

### Deep Reinforcement Learning Agents

- **Double Deep Q-Network (DDQN)** — The primary agent. Uses two neural networks (policy + target) to reduce Q-value overestimation via a decoupled action selection/evaluation mechanism. Includes experience replay and a decaying epsilon-greedy exploration schedule.
- **Deep Q-Network (DQN)** — The ablation/comparison agent. Subclasses `DDQNAgent`, overriding only the Bellman target computation. Used exclusively for DDQN-vs-DQN benchmarking.
- **Two-Agent Architecture** — The platform operates exclusively on DDQN and DQN. All hardcoded baseline policies have been removed for fairness in comparative deep learning benchmarking.

### WSN Environment (Gymnasium Compliant)

Simulates N sensor nodes, each with a battery tracked by **State of Charge (SoC)** and **State of Health (SoH)**. Influenced by real-world physics:

- Cycle-based and calendar battery degradation via the `BatteryModel`.
- **Charging state machine** — nodes below a SoC threshold enter forced-sleep charging mode.
- **Cooperative wake-ups** — when a low-battery node is awake, its nearest sleeping neighbor is forcefully woken for the next step.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone and Setup

```bash
cd WSN_M2M
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
cp .env.example .env
```

### Step 3: Create Output Directories

```bash
mkdir -p results/models results/metrics results/visualizations logs
```

### Step 4: Run

**Option A — CLI Training (recommended for swept runs):**

```bash
python scripts/train.py \
  --episodes 500 \
  --nodes 50 \
  --lr 1e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --seed 42 \
  --model-type ddqn
```

**Option B — Web Server:**

```bash
python -m backend.app
# Visit http://localhost:5001
```

---

## Project Structure

```
WSN_M2M/
├── config/
│   ├── config.yaml          — YAML source for all hyperparameters and paths
│   └── settings.py          — Dataclass config + get_config() singleton
│
├── src/                     — Pure RL core (no Flask, no HTTP, no file I/O)
│   ├── agents/
│   │   ├── base_agent.py    — BaseAgent ABC (strategy interface for Trainer)
│   │   ├── ddqn_agent.py    — DDQNAgent: policy + target net, decoupled Bellman target
│   │   └── dqn_agent.py     — DQNAgent: subclass of DDQNAgent, ablation only
│   ├── envs/
│   │   ├── wsn_env.py       — Gymnasium WSNEnv: SLEEP/AWAKE per-node actions
│   │   └── battery_model.py — SoC/SoH dynamics, cycle + calendar degradation
│   ├── training/
│   │   └── trainer.py       — Episode loop orchestrator
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── visualization.py
│
├── backend/                 — Flask REST API layer
│   ├── app.py               — Flask app factory (create_app)
│   ├── routes.py            — All route handlers, registered as blueprint api_bp
│   ├── schemas.py           — Marshmallow request validation schemas
│   └── tasks.py             — Sync/async training execution + comparison logic
│
├── frontend/                — Zero-build-step single-page UI
│   ├── templates/
│   │   └── index.html       — SPA entry point (Tailwind via CDN)
│   └── static/
│       ├── js/app.js        — All client-side logic (vanilla JS, no framework)
│       └── css/style.css    — Micro-adjustments layered on Tailwind
│
├── scripts/
│   ├── train.py             — CLI training entry point
│   └── compare.py           — CLI comparison plot generator
│
├── tests/
│   ├── conftest.py          — Shared fixtures + config singleton reset
│   ├── test_agent.py        — DDQNAgent / DQNAgent unit tests
│   ├── test_env.py          — WSNEnv / BatteryModel tests
│   └── test_backend.py      — Flask route tests (test_client)
│
├── results/
│   ├── models/              — Saved .pth model checkpoints
│   ├── metrics/             — Per-run metadata JSON files
│   └── visualizations/      — Training plots and comparison PNGs
│
└── logs/                    — Runtime training logs
```

---

## Architecture & Layer Rules

The repository is strictly layered to prevent circular dependencies. Upward imports are **prohibited** — `src/` has zero knowledge of Flask, HTTP, or the frontend.

```
frontend/          → display only; no business logic
      ↓ HTTP/JSON
backend/routes.py  → HTTP boundary: validate input, call tasks.py, return JSON
backend/tasks.py   → execution engine: construct env/agent, run Trainer, write artifacts
      ↓ Python
src/training/      → training loop only; no I/O, no Flask, no path construction
src/agents/        → Q-network math + replay buffer; no env knowledge
src/envs/          → simulation physics; no agent knowledge
config/            → read-only after startup; no side effects
```

### Design Patterns

| Pattern | Where | Rule |
|---------|-------|------|
| Strategy | `BaseAgent` + agent subclasses | All agents must subclass `BaseAgent`; `Trainer` only calls the abstract interface |
| Singleton | `get_config()` | Import config once via `get_config()`; never instantiate `Config` directly |
| Factory | `create_app()` in `backend/app.py` | Flask app always created via `create_app()`; `app.run()` only in `__main__` |
| Composition | `Trainer` composes `agent + env` | `Trainer` owns the loop; agents and envs do not call each other |

### Data Flow — Training

```
HTTP POST /api/train
  → schemas.py validates + applies defaults
  → tasks.run_training() constructs WSNEnv, DDQNAgent/DQNAgent, Trainer
  → Trainer.train() runs episodes
  → artifacts written: {run_id}_model.pth, {run_id}_plot.png, {run_id}_metadata.json
  → JSON response returned to frontend
```

### Data Flow — Async Training

```
HTTP POST /api/train/async
  → schemas.py validates
  → tasks.submit_training_task() spawns daemon thread → returns task_id (UUID)
  → client polls GET /api/tasks/<task_id> for { status: "queued"|"running"|"completed"|"failed" }
  → on completion, result is identical to sync flow
```

---

## Configuration System

All settings are read from `config/config.yaml` through dataclasses in `config/settings.py`. Access is always via the singleton:

```python
from config.settings import get_config
config = get_config()   # thread-safe singleton; safe to call multiple times
```

Never instantiate `Config` directly. In Flask route handlers, use `current_app.config.get("CONFIG")` instead of calling `get_config()` directly.

### Config Structure

**Training** (`config.training.*`)

| Key | Default | Description |
|-----|---------|-------------|
| `episodes` | 100 | Total training episodes |
| `batch_size` | 64 | Replay buffer sample size |
| `learning_rate` | 1e-4 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 10000 | Steps over which epsilon decays |
| `target_update_frequency` | 100 | Steps between target network syncs |
| `replay_buffer_size` | 50000 | Maximum transitions in buffer |
| `min_replay_size` | 1000 | Steps before learning starts |

**Environment** (`config.environment.*`)

| Key | Default | Description |
|-----|---------|-------------|
| `num_nodes` | 50 | Number of sensor nodes |
| `arena_size` | [500, 500] | Physical arena in metres |
| `sink_position` | [250, 250] | Sink node coordinates |
| `max_steps` | 1000 | Steps per episode |
| `death_threshold` | 0.3 | Dead-node fraction that ends an episode |
| `seed` | 42 | RNG seed (controls positions, weights, noise) |
| `sensing_radius` | 50.0 | Node sensing radius for coverage grid (metres) |
| `timestep_energy_awake` | 1.0 | Energy drained per step when AWAKE |
| `energy_sleep` | 0.01 | Energy drained per step when SLEEP |

**Reward Weights** (`config.environment.reward_weights.*`)

| Key | Default |
|-----|---------|
| `coverage` | 10.0 |
| `energy` | 5.0 |
| `soh` | 1.0 |
| `balance` | 2.0 |

**Charging** (`config.environment.charging.*`)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | true | Master toggle |
| `rate` | 0.05 | SoC fraction recovered per step while charging |
| `threshold` | 0.2 | SoC below which a node enters charging; exits at SoC ≥ 0.95 |

**Cooperative Wake-Up** (`config.environment.wake_cooperation.*`)

| Key | Default | Description |
|-----|---------|-------------|
| `low_battery_soc` | 0.5 | SoC fraction at which a node's nearest sleeping neighbor is force-woken |

**Paths** (`config.paths.*`)

| Key | Value |
|-----|-------|
| `models` | `"results/models"` |
| `metrics` | `"results/metrics"` |
| `visualizations` | `"results/visualizations"` |
| `logs` | `"logs"` |

> `config.paths.*` values are **relative strings**, not `Path` objects. Always wrap with `Path()` before joining. In `backend/`, use the `_abs()` helper to anchor to project root regardless of CWD.

---

## RL Core & Mathematics

### Observation Space

**6 features per node**, flattened into a shape `(N * 6,)` array (`state_dim = 300` for 50 nodes):

| Index (per node) | Feature | Range |
|-----------------|---------|-------|
| 0 | State of Charge (SoC, normalized) | [0, 1] |
| 1 | State of Health (SoH) | [0, 1] — never dynamically recovers |
| 2 | `last_action` | {0, 1} |
| 3 | `distance_to_sink` (normalized) | [0, 1] |
| 4 | `activity_ratio` (EMA) | [0, 1] |
| 5 | `charging_flag` | {0, 1} |

Always derive `state_dim` from the env: `state_dim = env.observation_space.shape[0]`. Never hardcode.

### Action Space

Per-node binary: `{SLEEP=0, AWAKE=1}`. The agent outputs an action vector of length `N`.

### Reward Function

```
reward = (10.0 × r_coverage) + (5.0 × r_energy) + (1.0 × r_soh) + (2.0 × r_balance)
```

All components are positive-encouraging factors to combat gradient saturation:

| Component | Description |
|-----------|-------------|
| `r_coverage` | Fraction of grid cells within sensing radius of at least one AWAKE node |
| `r_energy` | Normalized energy efficiency (lower drain = higher score) |
| `r_soh` | Average battery health (SoH) across all nodes |
| `r_balance` | Fairness metric (low std of SoC levels across nodes) |

### BatteryModel

Located in `src/envs/battery_model.py`. Tracks SoC and SoH per node.

- SoH degrades via **cycle-based + calendar degradation** — it does **not** recover.
- A node is dead when `SoC <= 0.01` or `SoH <= 0.05`.
- An episode ends when `dead_nodes > death_threshold × N`.

### Environment API

```python
# Gymnasium-compliant
state, info = env.reset()                              # returns (obs, info) — 2 values
next_state, reward, done, info = env.step(action)      # returns 4 values — no truncation flag
```

**`info` dict per step:**

```python
{
    "coverage":          float,   # grid coverage fraction (0–1)
    "avg_soh":           float,   # mean SoH across all nodes
    "alive_fraction":    float,   # fraction of nodes still alive
    "dead_count":        int,     # count of dead nodes this step
    "mean_soc":          float,   # mean normalized SoC across all nodes
    "cooperative_wakes": list,    # node IDs woken by cooperative rule this step
    "charging_count":    int,     # number of nodes currently charging
    "step_count":        int,     # current step within the episode
}
```

---

## Agents

### BaseAgent (Abstract Interface)

All agents implement:

```python
select_action(state, eval_mode=False) -> np.ndarray   # action vector length N
store_transition(state, action, reward, next_state, done)
learn_step()                                           -> Optional[float]  # loss or None
save_model(path: str)
load_model(path: str)
```

Pass `eval_mode=True` during evaluation to disable epsilon-greedy exploration.

### DDQNAgent — Primary Agent

| Property | Detail |
|----------|--------|
| Networks | `q_net` (policy) + `target_net` — synced every `target_update_frequency` steps |
| Bellman target | `a* = argmax Q_online(s', a)` then `y = r + γ × Q_target(s', a*) × (1–done)` |
| Replay buffer | Circular, capacity `replay_buffer_size`; learning is a no-op until `min_replay_size` reached |
| Exploration | Epsilon-greedy from `epsilon_start` → `epsilon_end` over `epsilon_decay` steps |
| Gradient clipping | Clip norm `10.0` applied in `learn_step()` |

### DQNAgent — Ablation/Comparison Agent

Subclass of `DDQNAgent` that overrides only the target computation:

```
y = r + γ × max_a Q_target(s', a) × (1–done)
```

Used exclusively for DDQN-vs-DQN comparison graphs. All API calls default to `"ddqn"`.

---

## Training Loop

`src/training/trainer.py` — the `Trainer` class orchestrates the per-episode loop:

```python
trainer = Trainer(agent, env, seed=42)
rewards, metrics = trainer.train(episodes=100)   # returns (list[float], dict)
trainer.save_checkpoint(path)                    # saves agent weights
```

**Per-episode series** (populated during `train()`):

```python
trainer.episode_series = {
    "episode_reward":  [...],
    "coverage":        [...],
    "avg_soh":         [...],
    "alive_fraction":  [...],
    "mean_soc":        [...],
    "step_counts":     [...],
}
trainer.network_lifetime   # int: episode where alive_fraction first dropped below (1 - death_threshold)
```

The loop per episode: `select_action()` → `env.step()` → `store_transition()` → `learn_step()`. Logs every 10 episodes; saves `.pth` checkpoints and metrics JSON at the end.

### Hyperparameter Ranges

| Parameter | Safe Range | Notes |
|-----------|-----------|-------|
| `learning_rate` | 1e-5 – 1e-3 | Start at 1e-4; reduce if loss spikes |
| `gamma` | 0.95 – 0.99 | 0.99 = longer-horizon planning |
| `batch_size` | 32 – 256 | Larger = more stable, slower per step |
| `episodes` | 50 – 1000 | 50-node default trains fast on CPU |

---

## Backend API Reference

Flask REST API with Marshmallow schemas for strict payload validation. App factory: `create_app()` in `backend/app.py`. All routes are registered on the `api_bp` blueprint at the `/api` prefix.

### Endpoints

| Endpoint | Method | Mode | Purpose |
|----------|--------|------|---------|
| `/api/health` | GET | sync | Basic health check |
| `/api/config` | GET | sync | Returns runtime server configuration |
| `/api/train` | POST | **blocking** | Runs training; connection stays open until complete |
| `/api/train/async` | POST | non-blocking | Forks to daemon thread; returns `task_id` immediately |
| `/api/tasks/<task_id>` | GET | sync | Poll in-memory task status |
| `/api/history` | GET | sync | List all training runs, newest first |
| `/api/compare?a=<id>&b=<id>` | GET | sync | Generate DDQN-vs-DQN comparison PNG on-the-fly |
| `/api/results/<path>` | GET | sync | Serve metrics JSON files |
| `/api/visualizations/<path>` | GET | sync | Serve visualization PNGs |

> `/api/evaluate` does not exist — baseline benchmarking was removed entirely.

### Task Lifecycle

```
POST /api/train/async → { "task_id": "<uuid>" }
GET  /api/tasks/<id>  → { "status": "queued" | "running" | "completed" | "failed" }
                       → { "status": "not_found", "task_id": "..." }  (404 on restart)
```

Task state is **in-memory only** — lost on server restart.

### Request Validation

Every route with a body validates through a Marshmallow schema (`backend/schemas.py`):

```python
schema = TrainingRequestSchema()
try:
    params = schema.load(request.json or {})
except ValidationError as e:
    return jsonify({"status": "error", "message": str(e.messages)}), 400
```

**`TrainingRequestSchema` defaults:**

| Field | Default | Notes |
|-------|---------|-------|
| `episodes` | 100 | |
| `nodes` | `None` | Resolved at runtime as `params.get("nodes") or config.environment.num_nodes` (50) |
| `learning_rate` | 1e-4 | |
| `gamma` | 0.99 | |
| `batch_size` | 64 | |
| `death_threshold` | 0.3 | |
| `max_steps` | 1000 | |
| `seed` | 42 | |
| `model_type` | `"ddqn"` | Only `"ddqn"` and `"dqn"` are accepted |

### Response Shape

```json
// Success
{ "status": "success", "run_id": "run_YYYYMMDD_HHMMSS", ... }

// Error
{ "status": "error", "message": "..." }

// Task poll (completed)
{ "status": "completed", "result": { ... } }

// Task poll (failed)
{ "status": "failed", "error": "..." }
```

---

## Frontend Details

A zero-build-step single-page application (`frontend/`). No `package.json`, no bundler, no transpilation.

- **HTML** — `templates/index.html` is the SPA entry point.
- **Styling** — Tailwind CSS loaded from CDN; custom semantic tokens (e.g. `bg-surface`) are defined in the inline `tailwind.config` block in `index.html`. Micro-adjustments in `static/css/style.css`.
- **Security** — `DOMPurify` (CDN) is required for all `innerHTML` assignments from API responses.
- **Image cache-busting** — `?t=Date.now()` is appended to plot `src` attributes on every new result.

### Three-Tab Right Panel

| Tab | Panel | Purpose |
|-----|-------|---------|
| `current` | `panelCurrent` | Result image, metric KPI cards, training status for the most recent run |
| `history` | `panelHistory` | One card per run, rendered from `GET /api/history`, newest first |
| `compare` | `panelCompare` | Pick Run A and Run B from history; generates a side-by-side comparison plot |

### Key JS Functions

| Function | Description |
|----------|-------------|
| `gatherPayload()` | Single source of truth for the `POST /api/train` body; maps form fields to API keys |
| `applyResult(data)` | Populates KPI cards from Phase-3 metadata schema after a successful training run |
| `normalizeRun(run)` | Smooths over old/new metadata schema; always use when rendering history rows |
| `loadCompareRuns()` | Fills the two `<select>` elements in the compare tab from `_historyCache` |
| `runComparison()` | Hits `GET /api/compare?a=<id>&b=<id>`, updates `compareImage.src` with cache-buster |
| `syncConfigDisplay()` | Syncs `data-config-mirror="<key>"` elements to the current payload on every input event |

---

## Output Artifacts & Metadata

### Run ID Format

```
run_YYYYMMDD_HHMMSS
```

Generated at the start of each training job:

```python
run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
```

### File Naming

| File | Path | Written by |
|------|------|-----------|
| Model weights | `results/models/{run_id}_model.pth` | `tasks.run_training()` |
| Training metadata | `results/metrics/{run_id}_metadata.json` | `tasks.run_training()` |
| Training plot | `results/visualizations/{run_id}_plot.png` | `tasks.run_training()` |
| Comparison plot | `results/visualizations/compare_{a}_vs_{b}.png` | `tasks.compare_runs()` |

### Metadata JSON Schema

```json
{
  "run_id":           "run_20260414_080528",
  "timestamp":        "2026-04-14T08:06:10.290101",
  "model_used":       "ddqn",
  "episodes":         100,
  "num_nodes":        50,
  "learning_rate":    0.0001,
  "gamma":            0.99,
  "death_threshold":  0.3,
  "max":              1000,
  "seed":             42,
  "metrics": {
    "mean_reward":       145.32,
    "max_reward":        180.5,
    "best_episode":      73,
    "avg_final_10":      172.4,
    "final_coverage":    0.87,
    "final_avg_soh":     0.94,
    "network_lifetime":  95
  },
  "series": {
    "episode_reward":  [...],
    "coverage":        [...],
    "avg_soh":         [...],
    "alive_fraction":  [...],
    "mean_soc":        [...],
    "step_counts":     [...]
  },
  "image_url":    "/api/visualizations/run_20260414_080528_plot.png",
  "model_path":   "results/models/run_20260414_080528_model.pth"
}
```

**Field glossary:**

| Field | Description |
|-------|-------------|
| `model_used` | `"ddqn"` or `"dqn"` |
| `max` | Max steps per episode (aliased from `max_steps` on the wire) |
| `final_coverage` | Grid coverage fraction at the last episode |
| `final_avg_soh` | Mean SoH across all nodes at the last episode |
| `network_lifetime` | Episode index at which `alive_fraction` first dropped below `1 - death_threshold` |
| `step_counts` | Per-episode count of steps survived |

---

## Testing

The system uses `pytest` for all CI guarantees.

### Test Files

| File | Covers |
|------|--------|
| `tests/test_agent.py` | `DDQNAgent`, `DQNAgent` — unit tests, no env needed |
| `tests/test_env.py` | `WSNEnv`, `BatteryModel` — environment dynamics |
| `tests/test_backend.py` | Flask routes via `app.test_client()` |

### Config Singleton Reset (Critical)

Any test file that imports `config.settings` must reset the singleton before and after the session to prevent test contamination:

```python
import config.settings as settings_module

@pytest.fixture(autouse=True, scope="session")
def reset_config():
    settings_module._config = None
    yield
    settings_module._config = None
```

### Fixtures

Tests use `N_NODES=10` to stay fast. Never use the production default of 50 nodes in tests.

```python
@pytest.fixture
def small_env():
    return WSNEnv(N=10, max_steps=50, death_threshold=0.3)
```

### Running Tests

```bash
pytest tests/                                                   # all tests
pytest tests/test_agent.py                                      # single file
pytest tests/test_agent.py::TestDDQNAgent::test_initialization  # single test
pytest tests/ -x                                                # stop on first failure
pytest tests/ -v                                                # verbose output
```

---

## CLI Reference

| Use-case | Command |
|----------|---------|
| Train DDQN (primary) | `python scripts/train.py --episodes 500 --nodes 50 --seed 42 --model-type ddqn` |
| Train DQN (ablation) | `python scripts/train.py --episodes 500 --nodes 50 --seed 42 --model-type dqn` |
| Compare runs (auto-picks latest DDQN + DQN) | `python scripts/compare.py` |
| Compare specific runs | `python scripts/compare.py --run-a run_20260414_080000 --run-b run_20260414_090000` |
| Run all tests | `pytest tests/ -v` |

All scripts in `scripts/` insert the project root onto `sys.path` via:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Replicate this in any new script, or always run scripts from the project root.

---

## Code Quality

```bash
# Formatting
black src/ backend/

# Linting
flake8 src/ backend/

# Static type checking
mypy src/ --ignore-missing-imports
```

---

## Extending the Platform

| Extension | How |
|-----------|-----|
| New agent | Subclass `BaseAgent` in `src/agents/`; implement all abstract methods; wire into `tasks.py` agent-selection block |
| New environment | Subclass `gym.Env`; implement `step()` and `reset()`; match the existing `info` dict shape |
| New metric | Add computation to `src/utils/metrics.py`; call from `Trainer._run_episode()` |
| New API route | Add handler to `backend/routes.py`; add schema to `backend/schemas.py`; add execution logic to `backend/tasks.py` |
