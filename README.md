# WSN DDQN Training Platform

A comprehensive deep reinforcement learning system for optimizing wireless sensor network (WSN) scheduling strategies. Uses Double Deep Q-Networks (DDQN) to train agents that balance network lifetime, coverage, and energy efficiency.

## Project Overview

This project implements a research-grade RL training platform with:

- **Modular Architecture**: Clean separation of RL logic, environments, and web interface
- **Scalable Web Interface**: Flask-based REST API for remote training and monitoring
- **Comprehensive Benchmarking**: Baseline policies and metrics for comparison
- **Production-Ready**: Async background jobs, logging, configuration management

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

### 3. Train a Model via CLI

```bash
python scripts/train_model.py --episodes 100 --nodes 50 --lr 1e-4
```

### 4. Run Web Server

```bash
python -m backend.app
```

Then visit `http://localhost:5001` in your browser.

### 5. Evaluate Against Baselines

```bash
python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth --episodes 10
```

## Project Structure

```
m2m_ddqn/
├── config/                    # Configuration management
├── src/                       # Core application logic
│   ├── agents/               # RL agent implementations
│   ├── envs/                 # Gym environments
│   ├── baselines/            # Baseline policies
│   ├── training/             # Training loop
│   └── utils/                # Logging, metrics, visualization
├── backend/                   # Flask web server
├── frontend/                  # Web UI templates & assets
├── scripts/                   # Standalone training/evaluation scripts
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── results/                   # Output directory
```

## Key Components

### Environment

[src/envs/wsn_env.py](src/envs/wsn_env.py) — Gym-compatible environment simulating WSN with battery degradation

### Agent

[src/agents/ddqn_agent.py](src/agents/ddqn_agent.py) — DDQN agent with experience replay and target network

### Trainer

[src/training/trainer.py](src/training/trainer.py) — Generic training loop supporting multiple agents/environments

### Web API

[backend/routes.py](backend/routes.py) — REST endpoints for training without CLI

## Configuration

All settings in [config/config.yaml](config/config.yaml):

```yaml
training:
  episodes: 100
  batch_size: 64
  learning_rate: 1.0e-4
  gamma: 0.99

environment:
  num_nodes: 50
  arena_size: [500, 500]
  max_steps: 1000
```

## Results

Models and metrics saved to `results/`:

```
results/
├── models/              # Trained weights
├── metrics/             # JSON metrics
└── visualizations/      # Training curves & plots
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/ backend/
flake8 src/ backend/
mypy src/ --ignore-missing-imports
```

### Documentation

See [docs/](docs/) for detailed guides:

- [Architecture](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [API Reference](docs/api.md)

## Reproduction

To reproduce published results:

```bash
python scripts/train_model.py \
  --episodes 500 \
  --nodes 50 \
  --lr 1e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --seed 42
```

## References

This implementation is based on:

- Deep Reinforcement Learning with Double Q-learning ([Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))
- Gym: A Toolkit for Developing and Comparing RL Algorithms ([Brockman et al., 2016](https://arxiv.org/abs/1606.01540))

## License

[Add your license here]

## Contact

[Add contact information]
