# System Architecture

## High-Level Design

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

## Core Modules

### 1. **config/** - Configuration Management

**Files:**

- `config.yaml` - YAML configuration file
- `settings.py` - Configuration loader and validation
- `logging_config.yaml` - Logging setup

**Purpose:** Centralized configuration for entire application

**Usage:**

```python
from config.settings import get_config
config = get_config()
print(config.training.episodes)  # Access settings via attributes
```

---

### 2. **src/agents/** - Reinforcement Learning Agents

**Files:**

- `base_agent.py` - Abstract base class defining agent interface
- `ddqn_agent.py` - Double Deep Q-Network implementation

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

### 3. **src/envs/** - Environment Simulation

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
reward = 10 * coverage - 5 * energy + 1 * soh + 2 * balance

Where:
- coverage = fraction of nodes awake
- energy = normalized energy usage
- soh = battery health
- balance = fairness (low std of charge levels)
```

---

### 4. **src/baselines/** - Reference Policies

**Policies:**

1. **RandomPolicy** - Random decisions
2. **GreedyPolicy** - Wake highest (SoC × SoH) nodes
3. **EnergyConservativePolicy** - Minimize energy (wake only best nodes)
4. **BalancedRotationPolicy** - Rotate awake sets periodically

**Design:** All inherit from `BaseAgent` for consistent interface

---

### 5. **src/training/** - Training Loop

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

### 6. **src/utils/** - Utilities

**Submodules:**

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

### 7. **backend/** - Web Server (Flask)

**Files:**

- `app.py` - Flask app factory
- `routes.py` - API endpoints
- `schemas.py` - Input validation

**Architecture:**

- **app.py**: Creates Flask app instance, registers blueprints
- **routes.py**: REST endpoints
  - `POST /api/train` - Start training
  - `GET /api/config` - Current configuration
  - `GET /api/results/<file>` - Download results
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

### 8. **scripts/** - CLI Tools

**Scripts:**

- `train_model.py` - Standalone training from CLI
- `evaluate_baselines.py` - Benchmark against reference policies

**Design:** Importable modules, not just scripts

---

### 9. **frontend/** - Web UI

**Structure:**

```
frontend/
├── templates/
│   ├── base.html
│   └── index.html
└── static/
    ├── css/
    ├── js/
    └── images/
```

---

## Data Flow

### Training Cycle

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

---

## Dependency Graph

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

---

## Design Patterns Used

1. **Strategy Pattern**: BaseAgent with multiple implementations
2. **Factory Pattern**: `create_app()` in backend
3. **Singleton Pattern**: `get_config()` for global configuration
4. **Observer Pattern**: Trainer callbacks (extensible)
5. **Composition**: Trainer composes Agent + Environment

---

## Extension Points

### Add New Agent

1. Inherit from `BaseAgent`
2. Implement required methods
3. Register in training pipeline

### Add New Baseline

1. Inherit from `BaseAgent`
2. Implement `select_action()`
3. Add to baselines registry

### Add New Environment

1. Inherit from `gym.Env`
2. Implement `step()`, `reset()`
3. Define observation/action spaces

### Add New Metric

1. Add function to `src/utils/metrics.py`
2. Call from `Trainer._run_episode()`
3. Aggregate in post-training analysis

---

## Performance Considerations

- **Batch Processing**: Experience replay enables efficient batching
- **Target Network**: Reduces computational cost of Bellman backup
- **GPU Acceleration**: PyTorch auto-selects CUDA when available
- **Lazy Evaluation**: State evaluation only when needed

---

## Testing Strategy

### Unit Tests (tests/)

- Test agents in isolation
- Test environment dynamics
- Test utility functions

### Integration Tests

- Test trainer with agents/envs
- Test Flask endpoints
- Test config loading

### Regression Tests

- Compare benchmarks against baselines
- Validate reward calculations

---

## Future Improvements

1. **Distributed Training**: Use Ray or Horovod for parallel agents
2. **Hyperparameter Tuning**: Integrate Optuna or Ray Tune
3. **Model Zoo**: Pre-trained models for different scenarios
4. **Live Monitoring**: Real-time dashboard with TensorBoard/Weights&Biases
5. **Multi-Agent RL**: Decentralized agent training
6. **Transfer Learning**: Domain adaptation across network sizes
