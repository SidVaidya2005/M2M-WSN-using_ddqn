# WSN DDQN Training Platform

A research-grade deep reinforcement learning platform for optimizing Wireless Sensor Network (WSN) scheduling using Double Deep Q-Networks (DDQN). Trains agents that balance network lifetime, coverage, and energy efficiency — exposed via both a CLI and a Flask REST API with a web dashboard.

---

## Key Concepts

### Double Deep Q-Network (DDQN)

The agent uses two neural networks (policy + target) to reduce Q-value overestimation, an experience replay buffer for sample efficiency, and a decaying epsilon-greedy exploration schedule.

### WSN Environment

Simulates N sensor nodes, each with a battery tracked by State of Charge (SoC) and State of Health (SoH). At each step the agent decides which nodes sleep or wake. The episode ends when too many nodes die (SoC < `death_threshold`). Reward balances coverage, energy efficiency, battery health, and fairness.

### Baseline Policies

| Policy | Strategy |
|--------|----------|
| Random | Random sleep/wake per node |
| Greedy | Wake the highest (SoC × SoH) nodes |
| EnergyConservative | Wake only the healthiest 20% of nodes |
| BalancedRotation | Rotate awake sets periodically |

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p results/models results/metrics results/visualizations logs
cp .env.example .env

# 2. Train via CLI
python scripts/train_model.py --episodes 100 --nodes 50 --lr 1e-4 --gamma 0.99 --batch-size 64 --seed 42 --model-type ddqn

# 3. Or launch the web dashboard
python -m backend.app        # → http://localhost:5001
```

---

## Project Structure

```
m2m_ddqn/
├── config/              # config.yaml + settings singleton
├── src/
│   ├── agents/          # BaseAgent, DDQNAgent, DQNAgent
│   ├── envs/            # WSNEnv (Gymnasium), BatteryModel
│   ├── baselines/       # baseline_policies.py
│   ├── training/        # Trainer (training loop)
│   └── utils/           # logging, metrics, visualization
├── backend/             # Flask API (app.py, routes.py, tasks.py, schemas.py)
├── frontend/            # templates/index.html + static/ (CDN-based, no build step)
├── scripts/             # CLI tools (train, evaluate, report, migrate)
├── tests/               # pytest suite
├── docs/                # Detailed reference guides
└── results/             # Generated output (gitignored)
    ├── models/          # run_{timestamp}_model.pth
    ├── metrics/         # run_{timestamp}_metadata.json, _evaluation.json
    └── visualizations/  # run_{timestamp}_plot.png
```

---

## Common Tasks

| Task | Command |
|------|---------|
| Train (CLI) | `python scripts/train_model.py --episodes 500 --nodes 50 --seed 42 --model-type ddqn` |
| Web dashboard | `python -m backend.app` → http://localhost:5001 |
| Evaluate baselines | `python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth --episodes 10` |
| Generate report | `python scripts/generate_report.py` |
| Migrate legacy artifacts | `python scripts/migrate_legacy_runs.py` |
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

---

## API Overview

The REST API runs at `http://localhost:5001/api`:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/train` | Start training (blocking) |
| POST | `/train/async` | Start training (non-blocking, returns `task_id`) |
| GET | `/tasks/<task_id>` | Poll async job status |
| GET | `/history` | List all training runs, newest first |
| POST | `/evaluate` | Run baseline benchmark for a `run_id` |
| GET | `/config` | Current configuration |
| GET | `/health` | Health check |

See [`docs/api.md`](docs/api.md) for full request/response schemas.

---

## Reproducibility

```bash
python scripts/train_model.py \
  --episodes 500 --nodes 50 \
  --lr 1e-4 --gamma 0.99 \
  --batch-size 64 --seed 42 \
  --model-type ddqn
```

The `--seed` controls initial weights, node positions, and exploration randomness — the same seed always produces the same results.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [`docs/getting_started.md`](docs/getting_started.md) | Installation, first run, understanding results |
| [`docs/architecture.md`](docs/architecture.md) | System design, data flow, design patterns |
| [`docs/training_guide.md`](docs/training_guide.md) | Hyperparameter tuning, advanced training, troubleshooting |
| [`docs/api.md`](docs/api.md) | Full REST API reference |

---

## References

- Deep Reinforcement Learning with Double Q-learning — [Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)
- Gymnasium: A Standard API for Reinforcement Learning Environments — [Farama Foundation](https://gymnasium.farama.org)
