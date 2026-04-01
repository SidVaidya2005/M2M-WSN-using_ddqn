# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python scripts/train_model.py --episodes 500 --nodes 550 --lr 1e-4 --gamma 0.99 --batch-size 64 --seed 42 --model-type ddqn

# Evaluate against baselines
python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth --episodes 10

# Generate research report from saved metrics
python scripts/generate_report.py
```

### Code Quality
```bash
black src/ backend/
flake8 src/ backend/
mypy src/ --ignore-missing-imports
```

## Architecture

```
Frontend (templates/) → HTTP/JSON → backend/ (Flask)
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

**`src/envs/wsn_env.py`** — Gymnasium-compatible environment. Observation: 5 features per node (SoC, SoH, last_action, distance_to_sink, activity_ratio). Reward balances coverage, energy efficiency, battery health, and fairness. `BatteryModel` tracks State of Charge (SoC) and State of Health (SoH) with cycle-based and calendar degradation.

**`src/training/trainer.py`** — Orchestrates the training loop. Calls `agent.select_action()` → `env.step()` → `agent.store_transition()` → `agent.learn_step()`. Logs every 10 episodes; saves `.pth` checkpoints and metrics JSON.

**`src/baselines/`** — Reference policies for benchmarking: `RandomPolicy`, `GreedyPolicy`, `EnergyConservativePolicy`, `BalancedRotationPolicy`.

**`backend/routes.py`** — REST endpoints: `POST /api/train` (sync), `POST /api/train/async` (returns task_id), `GET /api/tasks/<task_id>` (poll), `GET /api/config`, `GET /api/health`, plus static serving of plots and metrics JSON. Input validated via `backend/schemas.py` (marshmallow). Background jobs managed in `backend/tasks.py` (threading, in-memory registry).

### Output Artifacts
- `results/models/trained_model_{model_type}.pth` — PyTorch policy network weights (e.g. `trained_model_ddqn.pth`)
- `results/metrics/training_metrics_{model_type}.json` — Per-episode stats
- `results/metrics/baseline_comparison.json` — Baseline policy comparison (from `evaluate_baselines.py`)
- `results/visualizations/{model_type}_training_curve.png` — Training progress plots

## Gotchas

- **`WSNEnv.reset()` returns a plain `np.ndarray`**, not a `(obs, info)` tuple. Never do `state, _ = env.reset()` — it will crash with `ValueError: too many values to unpack` because the array has 2750 elements (550 nodes × 5 features). Use `state = env.reset()`.
- **Scripts require project root on `sys.path`**. All scripts in `scripts/` insert the project root via `Path(__file__).resolve().parent.parent`. If adding a new script, replicate this pattern or it will fail when run from a different directory.
- **Model and metrics filenames include the model type** (`ddqn` or `dqn`). Don't hardcode `trained_model.pth`; use `config.paths.models / f"trained_model_{model_type}.pth"`.
- **`POST /api/train` is synchronous** (blocks until training finishes) for frontend compatibility. Use `POST /api/train/async` for non-blocking invocation; poll `GET /api/tasks/<task_id>` for status.

## Extending the Platform

- **New agent**: Subclass `BaseAgent` in `src/agents/`, implement all abstract methods
- **New baseline**: Add policy to `src/baselines/`, register in the evaluation script
- **New environment**: Subclass `gym.Env`, implement `step()` and `reset()`
- **New metric**: Add computation to `src/utils/metrics.py`, call from `Trainer`
