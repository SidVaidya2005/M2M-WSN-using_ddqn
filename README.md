# WSN DDQN Training Platform

A research-grade deep reinforcement learning platform for optimizing Wireless Sensor Network (WSN) scheduling using Double Deep Q-Networks (DDQN). Trains agents that balance network lifetime, coverage, and energy efficiency — exposed via both a CLI and a Flask REST API with a web dashboard.

---

## Table of Contents

- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Common Tasks](#common-tasks)
- [Configuration](#configuration)
- [Training Guide](#training-guide)
- [System Architecture](#system-architecture)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Key Concepts

### Double Deep Q-Network (DDQN)

The agent uses two neural networks (policy + target) to reduce Q-value overestimation, an experience replay buffer for sample efficiency, and a decaying epsilon-greedy exploration schedule.

### WSN Environment

Simulates N sensor nodes, each with a battery tracked by State of Charge (SoC) and State of Health (SoH). At each step the agent decides which nodes sleep or wake. The episode ends when too many nodes die (SoC < `death_threshold`). Reward balances coverage, energy efficiency, battery health, and fairness.

### Two-Agent Benchmarking

DDQN (primary) is benchmarked exclusively against DQN (ablation). Both agents share the same
environment, seed, and training budget so the comparison is controlled. Use
`scripts/compare.py` or the web UI Compare tab to generate a 2×2 overlay plot.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone and Setup

```bash
cd m2m_ddqn
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env if needed for custom settings
```

### Step 4: Create Results Directories

```bash
mkdir -p results/models results/metrics results/visualizations logs
```

### First Training Run

#### Option A: Command Line (Recommended for First Run)

```bash
python scripts/train.py \
  --episodes 10 \
  --nodes 50 \
  --lr 1e-4 \
  --seed 42 \
  --model-type ddqn
```

**Parameters:**

- `--episodes`: Number of training episodes (start small: 10-50)
- `--nodes`: Number of sensor nodes (default: 50 from config)
- `--lr`: Learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--batch-size`: Replay batch size (default: 64)
- `--death-threshold`: Fraction of dead nodes ending an episode (default: 0.3)
- `--seed`: Random seed for reproducibility
- `--model-type`: `ddqn` (default) or `dqn`

**Expected Output:**

```
INFO:src.utils.logger:Training configuration: ...
INFO:src.utils.logger:Creating WSN environment with 50 nodes...
INFO:src.utils.logger:Starting training for 10 episodes...
Episode 1/10 - Reward: 45.32, 10-ep MA: 45.32
...
Training completed!
Model saved to results/trained_model.pth
Metrics saved to results/training_metrics.json
```

#### Option B: Web Server

```bash
python -m backend.app
```

Then visit `http://localhost:5001` in your browser.

**Form Fields:**

- **Episodes**: Training episodes (1-1000)
- **Nodes**: Number of nodes (10-10000)
- **Learning Rate**: 0.0001 to 0.1
- **Gamma**: 0.0 to 1.0

### Understanding Results

#### Generated Files

After training, check `results/`:

```
results/
├── models/run_{timestamp}_model.pth              # Neural network weights
├── metrics/run_{timestamp}_metadata.json          # Per-run config + summary metrics
└── visualizations/run_{timestamp}_plot.png        # Training progress plot
```

Run IDs have the format `run_YYYYMMDD_HHMMSS` (e.g. `run_20260406_080528`).

#### Metrics Explained

**Metrics JSON** (`run_{timestamp}_metadata.json`):

```json
{
  "run_id": "run_20260414_080528",
  "timestamp": "2026-04-14T08:06:10.290101",
  "model_used": "ddqn",
  "episodes": 10,
  "num_nodes": 50,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "death_threshold": 0.3,
  "max": 1000,
  "seed": 42,
  "metrics": {
    "mean_reward": 145.32,
    "max_reward": 180.5,
    "best_episode": 7,
    "avg_final_10": 172.4,
    "final_coverage": 0.87,
    "final_avg_soh": 0.94,
    "network_lifetime": 9
  },
  "series": { "episode_reward": [...], "coverage": [...], ... },
  "image_url": "/api/visualizations/run_20260414_080528_plot.png"
}
```

**Key Metrics:**

- `mean_reward`: Average reward per episode (higher is better)
- `final_coverage`: Grid coverage fraction at the last episode (0-1, higher is better)
- `final_avg_soh`: Average battery health at the last episode (0-1, higher is better)
- `network_lifetime`: Episode number when alive fraction first dropped below `1 - death_threshold`

---

## Project Structure

```
WSN_M2M/
├── config/              # config.yaml + settings singleton
├── src/
│   ├── agents/          # BaseAgent, DDQNAgent, DQNAgent
│   ├── envs/            # WSNEnv (Gymnasium), BatteryModel
│   ├── training/        # Trainer (training loop)
│   └── utils/           # logging, metrics, visualization
├── backend/             # Flask API (app.py, routes.py, tasks.py, schemas.py)
├── frontend/            # templates/index.html + static/ (CDN-based, no build step)
├── scripts/
│   ├── train.py         # CLI training wrapper (same artifacts as web API)
│   └── compare.py       # CLI DDQN-vs-DQN comparison report
├── tests/               # pytest suite
└── results/             # Generated output (gitignored)
    ├── models/          # run_{timestamp}_model.pth
    ├── metrics/         # run_{timestamp}_metadata.json
    └── visualizations/  # run_{timestamp}_plot.png, compare_*_vs_*.png
```

---

## Common Tasks

| Task | Command |
|------|---------|
| Train DDQN (CLI) | `python scripts/train.py --episodes 500 --nodes 50 --seed 42 --model-type ddqn` |
| Train DQN (CLI) | `python scripts/train.py --episodes 500 --nodes 50 --seed 42 --model-type dqn` |
| Compare runs (CLI) | `python scripts/compare.py` (auto-picks most recent DDQN + DQN) |
| Compare specific runs | `python scripts/compare.py --run-a <id> --run-b <id>` |
| Web dashboard | `python -m backend.app` → http://localhost:5001 |
| Run tests | `pytest tests/ -v` |
| Format code | `black src/ backend/` |
| Lint | `flake8 src/ backend/` |
| Type-check | `mypy src/ --ignore-missing-imports` |

---

## Configuration

All settings in [`config/config.yaml`](config/config.yaml):

```yaml
training:
  episodes: 100
  batch_size: 64
  learning_rate: 1.0e-4
  gamma: 0.99

environment:
  num_nodes: 550
  arena_size: [500, 500]
  sink_position: [250, 250]
  max_steps: 1000
  death_threshold: 0.3
```

Override any value via CLI flags or the web form — CLI flags take precedence over the YAML.

### Change Training Parameters

Edit `config/config.yaml`:

```yaml
training:
  episodes: 200 # More episodes
  batch_size: 128 # Larger batches
  learning_rate: 5e-5 # Lower learning rate
```

### Change Environment

Edit `config/config.yaml`:

```yaml
environment:
  num_nodes: 1000 # More nodes
  max_steps: 20000 # Longer episodes
  death_threshold: 0.5 # End when 50% dead
```

### Custom Agent Architecture

Edit `src/agents/ddqn_agent.py`:

```python
agent = DDQNAgent(
    ...,
    hidden_dims=[1024, 512, 256],  # Deeper network
    lr=1e-5,                         # Lower learning rate
)
```

---

## Training Guide

### Training Modes

#### 1. Command Line Training (Recommended)

Best for reproducible, parameter-swept training runs.

```bash
python scripts/train_model.py [OPTIONS]
```

**Common Options:**

```bash
# Quick test run
python scripts/train_model.py --episodes 10 --nodes 50

# Full training
python scripts/train_model.py --episodes 500 --nodes 50 --seed 42

# Custom hyperparameters
python scripts/train_model.py \
  --episodes 200 \
  --nodes 50 \
  --lr 5e-5 \
  --gamma 0.995 \
  --batch-size 32 \
  --seed 123
```

**Output:**

- `results/models/run_{timestamp}_model.pth` - Neural network weights
- `results/metrics/run_{timestamp}_metadata.json` - Per-run config + summary metrics
- `results/visualizations/run_{timestamp}_plot.png` - Training progress plot

#### 2. Web Interface Training

Best for interactive exploration and non-technical users.

```bash
python -m backend.app
# Visit http://localhost:5001
```

**Advantages:**

- Visual form for parameters
- Real-time progress updates
- Download results directly
- No command line needed

**Disadvantages:**

- Slower for multiple runs
- No easy parameter sweep
- Session-based (loses progress if connection drops)

#### 3. Programmatic Training

Best for custom workflows and research.

```python
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from config.settings import get_config

config = get_config()

# Create environment
env = WSNEnv(
    N=50,
    max_steps=1000,
    seed=42,
)

# Create agent
agent = DDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=2,
    node_count=50,
    lr=1e-4,
    gamma=0.99,
    batch_size=64,
)

# Create trainer
trainer = Trainer(agent, env)

# Train
rewards, metrics = trainer.train(episodes=100)

# Evaluate
eval_rewards, eval_metrics = trainer.evaluate(episodes=10)

# Save
trainer.save_checkpoint('results/my_model.pth')
```

### Training Workflow

#### Step 1: Configuration

Before training, decide on hyperparameters:

| Parameter     | Typical Range | Notes                                           |
| ------------- | ------------- | ----------------------------------------------- |
| Episodes      | 50-1000       | More = better learning but longer training      |
| Learning Rate | 1e-5 to 1e-3  | Too high = instability, too low = slow learning |
| Gamma         | 0.95-0.99     | Higher = agent plans further ahead              |
| Batch Size    | 32-256        | Larger = more stable gradients                  |
| Nodes         | 50-1000       | Larger = harder learning problem                |

**Quick Recommendation:**

```bash
python scripts/train_model.py \
  --episodes 100 \
  --nodes 100 \
  --lr 1e-4 \
  --gamma 0.99 \
  --batch-size 64
```

#### Step 2: Start Training

```bash
python scripts/train_model.py --episodes 100 --nodes 50 --seed 42
```

**Monitor Progress:**

- Watch for increasing rewards over time
- 10-episode moving average should increase
- ~5-10 minutes per 100 episodes on CPU

#### Step 3: Check Results

```bash
# View metrics (use the actual run_id from your training output)
cat results/metrics/run_{timestamp}_metadata.json
```

#### Step 4: Evaluate Model

```bash
python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth
```

**Outputs:**

- Mean reward vs baselines
- Coverage comparison
- Energy efficiency metrics

### Hyperparameter Tuning

#### Grid Search

Try multiple parameter combinations:

```bash
#!/bin/bash
for lr in 1e-5 1e-4 1e-3; do
  for gamma in 0.95 0.99; do
    python scripts/train_model.py \
      --episodes 100 \
      --lr $lr \
      --gamma $gamma \
      --output-dir results/lr_${lr}_gamma_${gamma}
  done
done
```

#### Manual Tuning Guidelines

**Learning Rate is too high if:**

- Loss spikes randomly
- Rewards oscillate wildly
- Training becomes unstable

**Solution:** Reduce to 1e-5

**Learning Rate is too low if:**

- Rewards increase very slowly
- Training takes forever
- Loss stays high

**Solution:** Increase to 1e-3

**Gamma (discount factor) tuning:**

- Gamma = 0.95: Agent focuses on immediate rewards
- Gamma = 0.99: Agent plans further ahead
- Gamma = 0.999: Very long-term planning

**Batch Size tuning:**

- Small (8-16): Faster training, noisier updates
- Large (128-256): Stable gradients, slower

### Advanced Training

#### Continue Training from Checkpoint

```python
agent = DDQNAgent(...)
agent.load_model('results/trained_model.pth')

trainer = Trainer(agent, env)
rewards, metrics = trainer.train(episodes=100)  # Train 100 more
trainer.save_checkpoint('results/trained_model_v2.pth')
```

#### Custom Reward Function

Modify `src/envs/wsn_env.py` step() method:

```python
# Default
reward = 10.0 * r_coverage + 5.0 * r_energy + 1.0 * r_soh + 2.0 * r_balance

# Custom: prioritize coverage more
reward = 15.0 * r_coverage + 3.0 * r_energy + 1.0 * r_soh + 1.0 * r_balance

# Custom: prioritize energy efficiency
reward = 5.0 * r_coverage + 10.0 * r_energy + 1.0 * r_soh + 1.0 * r_balance
```

#### Distributed Training (Parallel Seeds)

```bash
#!/bin/bash
for seed in 42 123 456; do
  python scripts/train_model.py \
    --episodes 100 \
    --seed $seed \
    --output-dir results/seed_$seed &
done
wait
```

Then aggregate results:

```python
import json
import numpy as np
from pathlib import Path

results = []
for seed_dir in Path('results').glob('seed_*'):
    with open(seed_dir / 'training_metrics.json') as f:
        results.append(json.load(f))

mean_reward = np.mean([r['training']['mean_reward'] for r in results])
std_reward = np.std([r['training']['mean_reward'] for r in results])
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

### Best Practices

1. **Use meaningful output directories:**
   ```bash
   --output-dir results/experiment_name
   ```

2. **Log your experiments:**
   ```bash
   tee results/experiment.log >>(python scripts/train_model.py ...)
   ```

3. **Version your code:**
   ```bash
   git commit "WIP: Testing lr=1e-5"
   ```

4. **Keep configs separate:**
   ```bash
   cp config/config.yaml config/config.baseline.yaml
   # Edit config.yaml for experiment
   ```

5. **Always evaluate baselines:**
   ```bash
   python scripts/evaluate_baselines.py
   ```

---

## System Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────────────┐
│                      Frontend (Web UI)                         │
│                   templates/, static/                          │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/JSON
┌────────────────────────▼─────────────────────────────────────┐
│                    Backend (Flask API)                        │
│              backend/routes.py, backend/app.py                │
└────────────────────────┬─────────────────────────────────────┘
                         │ Python
┌────────────────────────▼─────────────────────────────────────┐
│                 Training Controller                           │
│             src/training/trainer.py                           │
└────────────────────┬───────────────────┬─────────────────────┘
                     │                   │
        ┌────────────▼──┐        ┌───────▼────────┐
        │   RL Agent    │        │  Environment   │
        │ src/agents/   │        │  src/envs/     │
        │ ddqn_agent.py │        │  wsn_env.py    │
        └───────────────┘        └────────────────┘
                     │                   │
        ┌────────────▼──┐        ┌───────▼────────┐
        │  Q-Networks   │        │  Battery Model │
        │  (PyTorch)    │        │  Physics Sim   │
        └───────────────┘        └────────────────┘
```

### Core Modules

#### 1. config/ - Configuration Management

**Files:**

- `config.yaml` - YAML configuration file
- `settings.py` - Configuration loader and validation
- `logging_config.yaml` - Logging setup

**Usage:**

```python
from config.settings import get_config
config = get_config()
print(config.training.episodes)  # Access settings via attributes
```

---

#### 2. src/agents/ - Reinforcement Learning Agents

**Files:**

- `base_agent.py` - Abstract base class defining agent interface
- `ddqn_agent.py` - Double Deep Q-Network implementation (policy + target networks)
- `dqn_agent.py` - Single-network DQN implementation (for ablation comparisons)

**Design Pattern:** Strategy pattern via base class

**Key Classes:**

- `BaseAgent` - Abstract interface
- `DDQNAgent` - Concrete DDQN implementation with:
  - `select_action()` - Epsilon-greedy policy
  - `store_transition()` - Experience replay buffer
  - `learn_step()` - Training update
  - `save_model()` / `load_model()` - Persistence

**Architecture:**

```
DDQNAgent
├── Q-Network (policy)
├── Target Network
├── Replay Buffer (experience)
├── Optimizer (Adam)
└── Epsilon schedule (exploration)
```

---

#### 3. src/envs/ - Environment Simulation

**Files:**

- `battery_model.py` - Battery physics simulation
- `wsn_env.py` - Gym-compatible WSN environment

**Key Classes:**

**BatteryModel:**

```python
- discharge(energy) - Drain battery
- charge(energy) - Charge battery
- is_dead() - Check if failed
```

**WSNEnv (Gym.Env):**

- `reset()` - Initialize episode
- `step(action)` - Execute one simulation step
- `observation_space` - State shape
- `action_space` - Action shape

**Reward Function:**

```
reward = 10.0 * r_coverage + 5.0 * r_energy + 1.0 * r_soh + 2.0 * r_balance

Where (all terms are positive-good, clipped to bounded ranges):
- r_coverage = fraction of nodes awake [0, 1]
- r_energy   = -normalized_energy_usage (inverted so lower drain → higher score)
- r_soh      = average battery health [−1, 1]
- r_balance  = fairness penalty, −std of charge levels [−1, 0]

A heavy penalty of −10 is applied if the network fails (too many dead nodes).
```

---

#### 4. src/baselines/ - Reference Policies

**Policies:**

1. **RandomPolicy** - Random decisions
2. **GreedyPolicy** - Wake highest (SoC × SoH) nodes
3. **EnergyConservativePolicy** - Minimize energy (wake only best nodes)
4. **BalancedRotationPolicy** - Rotate awake sets periodically

**Design:** All inherit from `BaseAgent` for consistent interface

---

#### 5. src/training/ - Training Loop

**Files:**

- `trainer.py` - Generic training orchestrator

**Key Methods:**

- `train(episodes)` - Supervised training loop
- `evaluate(episodes)` - Evaluation without learning
- `_run_episode()` - Single episode execution
- `save_checkpoint()` / `load_checkpoint()` - Persistence

**Workflow:**

```python
trainer = Trainer(agent, env)
rewards, metrics = trainer.train(episodes=100)
trainer.save_checkpoint('models/best.pth')
```

---

#### 6. src/utils/ - Utilities

**logger.py:**

- Structured logging via YAML configuration
- Automatic file and console output
- Log levels per module

**metrics.py:**

- `compute_episode_metrics()` - Per-episode stats
- `aggregate_metrics()` - Multi-episode aggregation
- `compute_lifetime_metrics()` - Network lifetime

**visualization.py:**

- `save_metrics_json()` - Metric persistence
- `plot_training_curve()` - matplotlib plots
- `plot_comparison()` - Baseline comparison charts

---

#### 7. backend/ - Web Server (Flask)

**Files:**

- `app.py` - Flask app factory
- `routes.py` - API endpoints
- `schemas.py` - Input validation (marshmallow)
- `tasks.py` - Sync/async training execution and baseline benchmark logic

**Architecture:**

- **app.py**: Creates Flask app instance, registers blueprints
- **routes.py**: REST endpoints
- **schemas.py**: Marshmallow schemas for validation

**Request Flow:**

```
HTTP Request
    ↓
Schemas.py (Validation)
    ↓
routes.py (Handler)
    ↓
Trainer + Agent + Environment
    ↓
Results saved to results/
    ↓
HTTP Response JSON
```

---

#### 8. scripts/ - CLI Tools

- `train_model.py` - Standalone training from CLI
- `evaluate_baselines.py` - Benchmark against reference policies
- `generate_report.py` - Generate research report from saved metrics JSON
- `migrate_legacy_runs.py` - Migrate pre-run_id artifacts into the `run_{timestamp}_*` naming scheme (safe, idempotent)

---

#### 9. frontend/ - Web UI

```
frontend/
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/app.js
    └── images/
```

### Data Flow

#### Training Cycle

```
1. User Request (Web/CLI)
           ↓
2. Config Load → Validation
           ↓
3. Environment Reset
           ↓
4. Agent Selects Action (Epsilon-greedy)
           ↓
5. Env Step → Reward, Done, Next State
           ↓
6. Store Transition (Replay Buffer)
           ↓
7. Sample Batch from Replay Buffer
           ↓
8. Compute Q-targets (DDQN logic)
           ↓
9. Backward Pass → Update Networks
           ↓
10. Update Target Network (periodic)
           ↓
11. Continue or End Episode
           ↓
12. Save Metrics & Visualizations
```

### Dependency Graph

```
Agent (DDQN)
├── Base Agent
├── Replay Buffer
├── Q-Networks (PyTorch)
└── Optimizer

Environment (WSN)
├── Battery Model
└── Reward Function

Trainer
├── Agent interface
├── Environment interface
└── Metrics utils

Backend
├── Trainer
├── Config
└── Validation (Schemas)

Scripts
├── Trainer
├── Agent
├── Environment
├── Baselines
└── Utils
```

### Design Patterns Used

1. **Strategy Pattern**: BaseAgent with multiple implementations
2. **Factory Pattern**: `create_app()` in backend
3. **Singleton Pattern**: `get_config()` for global configuration
4. **Observer Pattern**: Trainer callbacks (extensible)
5. **Composition**: Trainer composes Agent + Environment

### Extension Points

**Add New Agent:** Inherit from `BaseAgent`, implement required methods, register in training pipeline.

**Add New Baseline:** Inherit from `BaseAgent`, implement `select_action()`, add to baselines registry.

**Add New Environment:** Inherit from `gym.Env`, implement `step()` and `reset()`, define observation/action spaces.

**Add New Metric:** Add function to `src/utils/metrics.py`, call from `Trainer._run_episode()`, aggregate in post-training analysis.

### Testing Strategy

- **Unit Tests**: Test agents in isolation, environment dynamics, utility functions
- **Integration Tests**: Test trainer with agents/envs, Flask endpoints, config loading
- **Regression Tests**: Compare benchmarks against baselines, validate reward calculations

### Future Improvements

1. **Distributed Training**: Use Ray or Horovod for parallel agents
2. **Hyperparameter Tuning**: Integrate Optuna or Ray Tune
3. **Model Zoo**: Pre-trained models for different scenarios
4. **Live Monitoring**: Real-time dashboard with TensorBoard/Weights&Biases
5. **Multi-Agent RL**: Decentralized agent training
6. **Transfer Learning**: Domain adaptation across network sizes

---

## API Reference

All endpoints are at `/api/` base path. Currently no authentication required.

### API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/train` | Start training (blocking) |
| POST | `/train/async` | Start training (non-blocking, returns `task_id`) |
| GET | `/tasks/<task_id>` | Poll async job status |
| GET | `/history` | List all training runs, newest first |
| POST | `/evaluate` | Run baseline benchmark for a `run_id` |
| GET | `/config` | Current configuration |
| GET | `/health` | Health check |

### GET /api/health

Health check endpoint to verify server is running.

**Response:**

```json
{
  "status": "healthy"
}
```

**Status Code:** 200

```bash
curl http://localhost:5001/api/health
```

---

### GET /api/config

Retrieve current configuration.

**Response:**

```json
{
  "training": {
    "episodes": 100,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "gamma": 0.99
  },
  "environment": {
    "num_nodes": 50,
    "arena_size": [500, 500],
    "max_steps": 1000
  },
  "paths": {
    "models": "results/models",
    "metrics": "results/metrics"
  }
}
```

**Status Code:** 200

```bash
curl http://localhost:5001/api/config
```

---

### POST /api/train

Start training a new model (blocks until complete).

**Request Body:**

```json
{
  "episodes": 100,
  "nodes": 50,
  "model_type": "ddqn",
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "batch_size": 64,
  "seed": 42
}
```

| Field         | Type  | Default | Range        |
| ------------- | ----- | ------- | ------------ |
| episodes      | int   | 100     | 1-10000      |
| nodes         | int   | 50      | 10-10000     |
| model_type    | str   | ddqn    | dqn or ddqn  |
| learning_rate | float | 1e-4    | 1e-6 to 1e-1 |
| gamma         | float | 0.99    | 0.0-1.0      |
| batch_size    | int   | 64      | 8-512        |
| seed          | int   | 42      | any int      |

**Response (Success):**

```json
{
  "status": "success",
  "message": "Training completed successfully with DDQN.",
  "episodes": 100,
  "nodes": 50,
  "model_type": "ddqn",
  "mean_reward": 145.32,
  "max_reward": 180.5,
  "results": {
    "best_lifetime": 180.5,
    "best_episode": 73,
    "avg_lifetime_final_10": 172.4
  },
  "model_path": "results/models/run_20260406_080528_model.pth"
}
```

- `results.best_episode` is the 1-based episode number where the maximum reward occurred.
- `results.avg_lifetime_final_10` is the average reward over the last up to 10 episodes.

**Response (Error):**

```json
{
  "status": "error",
  "message": "Invalid batch_size: must be >= 8"
}
```

**Status Code:** 200 (success), 400 (bad request), or 500 (server error)

```bash
curl -X POST http://localhost:5001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": 50,
    "nodes": 100,
    "model_type": "dqn",
    "learning_rate": 0.0001
  }'
```

---

### POST /api/train/async

Start training without blocking. Returns a `task_id` immediately; poll for status.

**Request Body:** Same fields as `POST /api/train`.

**Response:**

```json
{ "task_id": "550e8400-e29b-41d4-a716-446655440000" }
```

**Status Code:** 202

---

### GET /api/tasks/\<task_id\>

Poll the status of an async training or benchmark job.

**Response:**

```json
{ "status": "queued" }
{ "status": "running" }
{ "status": "completed", "result": { ... } }
{ "status": "failed",    "error": "..." }
{ "status": "not_found" }
```

`"not_found"` is returned (not a 404) if the task_id is unknown or the server was restarted (task registry is in-memory only).

**Status Code:** 200

---

### GET /api/history

Return all training run metadata, newest first. Scans `results/metrics/` for `*_metadata.json` files. If a matching `*_evaluation.json` exists for a run, it is inlined as `run["evaluation"]`.

**Response:**

```json
[
  {
    "run_id": "run_20260406_080528",
    "timestamp": "2026-04-06T08:06:10.290101",
    "config": { "model_type": "ddqn", "episodes": 100, "nodes": 550 },
    "metrics": { "mean_reward": 145.32, "max_reward": 180.5 },
    "image_url": "/api/visualizations/run_20260406_080528_plot.png",
    "evaluation": { ... }
  }
]
```

**Status Code:** 200

---

### POST /api/evaluate

Submit an async baseline benchmark job for a completed training run.

**Request Body:**

```json
{ "run_id": "run_20260406_080528", "episodes": 10 }
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| run_id | str | required | From `GET /api/history` |
| episodes | int | 10 | 1–100 |

**Response:**

```json
{ "task_id": "550e8400-e29b-41d4-a716-446655440000" }
```

Poll `GET /api/tasks/<task_id>` for results. On completion, writes `results/metrics/{run_id}_evaluation.json`.

**Status Code:** 202

---

### GET /api/results/\<filename\>

Retrieve output files (models, metrics, visualizations).

**Response:** File content (binary for .pth, JSON for .json, PNG for .png)

**Status Code:** 200 (found), 404 (not found)

```bash
# Download JSON metrics
curl http://localhost:5001/api/results/training_metrics.json > metrics.json

# Download trained model
curl http://localhost:5001/api/results/trained_model.pth > model.pth
```

### Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

| Code | Meaning      | Example               |
| ---- | ------------ | --------------------- |
| 200  | Success      | Training completed    |
| 400  | Bad request  | Invalid parameter     |
| 404  | Not found    | Results file missing  |
| 500  | Server error | Crash during training |

### Input Validation (Schemas)

The API validates all inputs using Marshmallow schemas defined in `backend/schemas.py`.

```python
class TrainingRequestSchema(Schema):
    episodes = fields.Int(
        required=False,
        validate=validate.Range(min=1, max=10000),
        missing=100,
    )
    nodes = fields.Int(
        required=False,
        validate=validate.Range(min=10, max=10000),
        missing=50,
    )
    learning_rate = fields.Float(
        required=False,
        validate=validate.Range(min=1e-6, max=1e-1),
        missing=1e-4,
    )
```

Invalid requests return:

```json
{
  "error": "Validation error",
  "details": {
    "learning_rate": ["Must be between 1e-6 and 1e-1"]
  }
}
```

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:5001/api"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())  # {"status": "healthy"}

# Get config
response = requests.get(f"{BASE_URL}/config")
config = response.json()
print(f"Current episodes: {config['training']['episodes']}")

# Start training
training_config = {
    "episodes": 100,
    "nodes": 50,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 64,
}

response = requests.post(
    f"{BASE_URL}/train",
    json=training_config,
)

if response.status_code == 200:
    result = response.json()
    print(f"Training complete!")
    print(f"Mean reward: {result['mean_reward']:.2f}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript Client Example

```javascript
const BASE_URL = "http://localhost:5001/api";

// Health check
async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  const data = await response.json();
  console.log(data); // {status: "healthy"}
}

// Start training
async function startTraining(trainingConfig) {
  const response = await fetch(`${BASE_URL}/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(trainingConfig),
  });

  if (response.ok) {
    const result = await response.json();
    console.log(`Training complete!`);
    console.log(`Mean reward: ${result.mean_reward.toFixed(2)}`);
  } else {
    console.error(`Error: ${await response.text()}`);
  }
}
```

### Deployment

#### Running with Gunicorn (Production)

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5001 backend.app:app
```

#### Running with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["gunicorn", "-w 4", "-b 0.0.0.0:5001", "backend.app:app"]
```

```bash
docker build -t wsn-ddqn .
docker run -p 5001:5001 wsn-ddqn
```

### Monitoring & Logging

All API calls are logged. Configure in `config/logging_config.yaml`:

```yaml
loggers:
  backend:
    level: INFO
    handlers: [console, file]
```

```bash
tail -f logs/app.log
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch
```

### Issue: `CUDA out of memory`

Use CPU instead:

```python
agent = DDQNAgent(..., device='cpu')
```

### Issue: Training is very slow

- Reduce number of nodes: `--nodes 100`
- Reduce episodes: `--episodes 10`
- Reduce batch size in config: edit `config/config.yaml`

### Issue: `Config file not found`

```bash
# Make sure you're in the project root
cd m2m_ddqn
python scripts/train_model.py ...
```

### Problem: Rewards Not Increasing

**Likely causes:**

1. Learning rate too low
2. Not enough episodes
3. Bad hyperparameters

**Solutions:**

```bash
# Try higher learning rate
python scripts/train_model.py --lr 1e-3 --episodes 200

# Try different gamma
python scripts/train_model.py --gamma 0.95 --episodes 200
```

### Problem: Memory Error

**Likely causes:**

1. Too many nodes
2. Replay buffer too large
3. Batch size too large

**Solutions:**

```bash
# Reduce problem size
python scripts/train_model.py --nodes 100 --batch-size 32

# Edit config.yaml:
# replay_buffer_size: 100000  (was 200000)
```

### Problem: NaN in Loss

**Likely causes:**

1. Learning rate too high
2. Exploding gradients

**Solutions:**

```bash
# Lower learning rate
python scripts/train_model.py --lr 1e-5

# Edit ddqn_agent.py:
nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)  # Lower from 10.0
```

---

## Performance Benchmarks

### On CPU (Intel i7, 8GB RAM):

| Nodes | Episodes | Time   | Mean Reward |
| ----- | -------- | ------ | ----------- |
| 50    | 100      | 2 min  | ~120        |
| 100   | 100      | 5 min  | ~100        |
| 550   | 100      | 45 min | ~150        |  ← full-scale

### On GPU (NVIDIA GTX 1080):

| Nodes | Episodes | Time   | Mean Reward |
| ----- | -------- | ------ | ----------- |
| 550   | 100      | 8 min  | ~150        |
| 550   | 500      | 40 min | ~180        |

---

## Reproducibility

```bash
python scripts/train_model.py \
  --episodes 500 --nodes 50 \
  --lr 1e-4 --gamma 0.99 \
  --batch-size 64 --seed 42
```

The `--seed` controls initial weights, node positions, and exploration randomness — the same seed always produces the same results.

---

## References

- Deep Reinforcement Learning with Double Q-learning — [Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)
- Gymnasium: A Standard API for Reinforcement Learning Environments — [Farama Foundation](https://gymnasium.farama.org)
