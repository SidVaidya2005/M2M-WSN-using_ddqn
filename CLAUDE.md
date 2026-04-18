# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Detailed Rules

Precise, topic-scoped rules live in `.claude/rules/`:

| File | Covers |
|------|--------|
| [`architecture.md`](.claude/rules/architecture.md) | Design patterns, layer responsibilities, extension points |
| [`rl-environment.md`](.claude/rules/rl-environment.md) | WSNEnv API, observation space, reward weights, BatteryModel |
| [`agents-training.md`](.claude/rules/agents-training.md) | BaseAgent interface, DDQN internals, Trainer API, hyperparameters |
| [`api-design.md`](.claude/rules/api-design.md) | Sync/async routing, validation, task lifecycle, response shape |
| [`config-paths.md`](.claude/rules/config-paths.md) | Config singleton, YAML structure, path handling |
| [`testing.md`](.claude/rules/testing.md) | Conftest singleton reset, fixture sizes, test patterns |
| [`artifacts.md`](.claude/rules/artifacts.md) | Run ID format, file naming, metadata schema |

Directory-level context: [`frontend/CLAUDE.md`](frontend/CLAUDE.md) · [`backend/CLAUDE.md`](backend/CLAUDE.md) · [`src/CLAUDE.md`](src/CLAUDE.md)

**Restructure complete:** All 6 phases of the restructure described in [`plan.md`](plan.md) are done. The codebase is in its final state: two agents only (DDQN + DQN), 6-feature observation space, charging + cooperative wake-up in WSNEnv, Phase 3 metadata schema, updated frontend with compare tab, and CLI scripts in `scripts/`.

---

## Project Overview

A research-grade Deep Reinforcement Learning (RL) platform for optimizing Wireless Sensor Network (WSN) scheduling using Double Deep Q-Networks (DDQN). Exposes both a CLI and a Flask REST API with a web UI.

## Commands

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p results/models results/metrics results/visualizations logs
cp .env.example .env
```

### Run
```bash
# Web server (localhost:5001)
python -m backend.app

# CLI training (--model-type dqn or ddqn)
python scripts/train.py --episodes 500 --nodes 50 --lr 1e-4 --gamma 0.99 --batch-size 64 --seed 42 --model-type ddqn

# CLI comparison (auto-picks most recent DDQN and DQN runs)
python scripts/compare.py
# Or specify runs explicitly:
python scripts/compare.py --run-a run_20260414_080000 --run-b run_20260414_090000
```

### Tests
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_agent.py

# Run a specific test
pytest tests/test_agent.py::TestDDQNAgent::test_initialization -v
```

Tests use a small `N_NODES=10` fixture to stay fast. The `conftest.py` resets the config singleton (`settings_module._config = None`) before and after each session — replicate this pattern in any new test that imports `config.settings`.

### Code Quality
```bash
black src/ backend/
flake8 src/ backend/
mypy src/ --ignore-missing-imports
```

## Architecture

```
Frontend (frontend/templates/ + frontend/static/) → HTTP/JSON → backend/ (Flask)
                                         ↓
                              src/training/trainer.py
                              ├── src/agents/ddqn_agent.py  (DDQNAgent)
                              ├── src/agents/dqn_agent.py   (DQNAgent)
                              ├── src/agents/base_agent.py  (BaseAgent ABC)
                              ├── src/envs/wsn_env.py    (Gymnasium env, BatteryModel)
                              └── src/utils/             (logging, metrics, visualization)
                                         ↓
                              results/ (models, metrics JSON, PNG plots)
```

### Key Modules

**`config/settings.py`** — Dataclass-based config with validation; use `get_config()` singleton. YAML source: `config/config.yaml`. All components import from here.

**`src/agents/`** — `BaseAgent` is the abstract interface. `DDQNAgent` uses two PyTorch networks (policy + target), epsilon-greedy exploration, and a `ReplayBuffer`. `DQNAgent` uses a single network. Action space is per-node `{SLEEP=0, AWAKE=1}`.

**`src/envs/wsn_env.py`** — Gymnasium-compatible environment. Observation: 6 features per node (SoC, SoH, last_action, distance_to_sink, activity_ratio, charging_flag). Reward balances coverage, energy efficiency, battery health, and fairness. `BatteryModel` (in `src/envs/battery_model.py`) tracks State of Charge (SoC) and State of Health (SoH) with cycle-based and calendar degradation.

**`src/training/trainer.py`** — Orchestrates the training loop. Calls `agent.select_action()` → `env.step()` → `agent.store_transition()` → `agent.learn_step()`. Logs every 10 episodes; saves `.pth` checkpoints and metrics JSON.

**`backend/`** — Flask REST API layer. See [`backend/CLAUDE.md`](backend/CLAUDE.md).

**`frontend/`** — Single-page UI (no build step). See [`frontend/CLAUDE.md`](frontend/CLAUDE.md).

**`src/`** — Pure RL core (envs, agents, training loop). See [`src/CLAUDE.md`](src/CLAUDE.md).

### Output Artifacts
- `results/models/run_{timestamp}_model.pth` — PyTorch policy network weights per run
- `results/metrics/run_{timestamp}_metadata.json` — Per-run config + summary metrics
- `results/visualizations/run_{timestamp}_plot.png` — Combined 2×2 training dashboard (50-ep MA overlay)
- `results/visualizations/{timestamp}/` — Individual PNGs per panel: `coverage.png`, `battery_health.png`, `energy_consumption.png`, `throughput.png`

Run IDs have the format `run_YYYYMMDD_HHMMSS`. The individual-plots directory uses the timestamp portion only (e.g. `20260414_080000`).

## Gotchas

- **`WSNEnv` is Gymnasium-compliant** — `reset()` returns `(obs, info)`; `step()` returns 4 values `(next_state, reward, done, info)`, not 5 (no truncation flag).
- **Scripts require project root on `sys.path`**. All scripts in `scripts/` insert the project root via `Path(__file__).resolve().parent.parent`. If adding a new script, replicate this pattern or it will fail when run from a different directory.
- **`config.paths.*` fields are `str`, not `Path`** — always wrap with `Path()` before using `/` for joining. Artifacts use the `run_{timestamp}_*` naming scheme (see Output Artifacts above).

## Extending the Platform

- **New agent**: Subclass `BaseAgent` in `src/agents/`, implement all abstract methods
- **New environment**: Subclass `gym.Env`, implement `step()` and `reset()`
- **New metric**: Add computation to `src/utils/metrics.py`, call from `Trainer`
