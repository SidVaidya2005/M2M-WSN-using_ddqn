# Project Restructuring Guide: WSN DDQN Training Platform

## New Directory Structure

```
m2m_ddqn/
├── README.md                          # Project overview & quick start
├── requirements.txt                   # Python dependencies (pinned versions)
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── setup.py                           # Package installation config
│
├── config/                            # Configuration management
│   ├── __init__.py
│   ├── config.yaml                    # Global config (training params, paths)
│   │── settings.py                    # Config loader + validation
│   └── logging_config.yaml            # Logging configuration
│
├── src/                               # Core application logic (NOT test/demo code)
│   ├── __init__.py
│   ├── agents/                        # RL Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Abstract base class
│   │   └── ddqn_agent.py              # DDQN implementation (from ddqn_agent.py)
│   │
│   ├── envs/                          # Gym environments
│   │   ├── __init__.py
│   │   ├── battery_model.py           # Battery model (extracted from env_wsn.py)
│   │   └── wsn_env.py                 # WSN environment (from env_wsn.py)
│   │
│   ├── baselines/                     # Baseline policies for comparison
│   │   ├── __init__.py
│   │   └── baseline_policies.py       # All baselines (from baselines.py)
│   │
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py                  # Structured logging
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── visualization.py           # Plot generation utilities
│   │
│   └── training/                      # Training utilities
│       ├── __init__.py
│       └── trainer.py                 # Extracted training logic (from train_final_ddqn.py)
│
├── backend/                           # Flask web server & API
│   ├── __init__.py
│   ├── app.py                         # Flask app factory & config
│   ├── routes.py                      # API endpoints (refactored from app.py)
│   ├── schemas.py                     # Input validation (Marshmallow or Pydantic)
│   ├── tasks.py                       # Background job handling (Celery)
│   └── models.py                      # DB models for tracking runs/results
│
├── frontend/                          # Web UI
│   ├── templates/
│   │   ├── base.html                  # Base template
│   │   ├── index.html                 # Main page (from templates/index.html)
│   │   ├── training.html              # Training control panel
│   │   ├── results.html               # Results visualization
│   │   └── components/
│   │       ├── navbar.html
│   │       └── form.html
│   │
│   └── static/
│       ├── css/
│       │   ├── style.css
│       │   └── dashboard.css
│       ├── js/
│       │   ├── app.js                 # Main frontend logic
│       │   ├── training.js            # Training control
│       │   └── charts.js              # Chart.js visualizations
│       └── images/
│
├── scripts/                           # Standalone scripts (one-off training, analysis)
│   ├── train_model.py                 # Refactored train_final_ddqn.py
│   ├── evaluate_baselines.py          # Refactored compare_realistic_metrics.py
│   └── generate_report.py             # New: generate research reports
│
├── notebooks/                         # Jupyter notebooks (exploration, analysis)
│   ├── analysis.ipynb                 # Data analysis & visualization
│   └── hyperparameter_tuning.ipynb    # Hyperparameter exploration
│
├── tests/                             # Unit & integration tests
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_agent.py                  # Test DDQN agent
│   ├── test_env.py                    # Test WSN environment
│   ├── test_baselines.py              # Test baseline policies
│   └── test_backend.py                # Test Flask endpoints
│
├── results/                           # Output directory (generated, not tracked)
│   ├── models/                        # Trained model checkpoints
│   │   └── .gitkeep
│   ├── metrics/                       # Metrics JSON files
│   │   └── .gitkeep
│   └── visualizations/                # Generated plots & animations
│       └── .gitkeep
│
├── docs/                              # Documentation
│   ├── architecture.md                # System design
│   ├── getting_started.md             # Setup & first run
│   ├── api.md                         # API documentation
│   ├── training_guide.md              # How to train models
│   └── contributing.md                # Development guidelines
│
└── .vscode/                           # VS Code settings (optional)
    └── settings.json

```

---

## File Migration Guide

### **SRC/ - Core Logic (Reusable Modules)**

| Old File        | New Location                         | Changes                           |
| --------------- | ------------------------------------ | --------------------------------- |
| `ddqn_agent.py` | `src/agents/ddqn_agent.py`           | ✅ Keep as-is, add docstrings     |
| `env_wsn.py`    | Split into two:                      | **SPLIT** (see below)             |
| —               | `src/envs/battery_model.py`          | Extract `BatteryModel` class      |
| —               | `src/envs/wsn_env.py`                | Keep `WSNEnv` class               |
| `baselines.py`  | `src/baselines/baseline_policies.py` | ✅ Rename, add missing docstrings |
| **NEW**         | `src/agents/base_agent.py`           | Create abstract base class        |
| **NEW**         | `src/utils/logger.py`                | Centralized logging setup         |
| **NEW**         | `src/utils/metrics.py`               | Extract metric calculations       |
| **NEW**         | `src/utils/visualization.py`         | Extract plot generation           |
| **NEW**         | `src/training/trainer.py`            | Extract training loop logic       |

### **BACKEND/ - Web Server & API**

| Old File | New Location         | Changes                               |
| -------- | -------------------- | ------------------------------------- |
| `app.py` | Split into:          | **REFACTOR**                          |
| —        | `backend/app.py`     | Flask app factory only                |
| —        | `backend/routes.py`  | HTTP endpoints (from app.py handlers) |
| **NEW**  | `backend/tasks.py`   | Background job execution (async)      |
| **NEW**  | `backend/schemas.py` | Input validation                      |
| **NEW**  | `backend/models.py`  | Database models (optional)            |

### **FRONTEND/ - Web UI**

| Old File               | New Location                    | Changes                         |
| ---------------------- | ------------------------------- | ------------------------------- |
| `templates/index.html` | `frontend/templates/index.html` | ✅ Keep, maybe split components |
| **NEW**                | `frontend/templates/base.html`  | Base template for inheritance   |
| **NEW**                | `frontend/static/css/style.css` | Extracted styles                |
| **NEW**                | `frontend/static/js/app.js`     | Extracted JS logic              |

### **SCRIPTS/ - Standalone Tools**

| Old File                       | New Location                    | Changes                                   |
| ------------------------------ | ------------------------------- | ----------------------------------------- |
| `train_final_ddqn.py`          | `scripts/train_model.py`        | Refactor to use `src/training/trainer.py` |
| `compare_realistic_metrics.py` | `scripts/evaluate_baselines.py` | Refactor to use src modules               |
| **NEW**                        | `scripts/generate_report.py`    | Create structured reports                 |

### **CONFIG/ - Configuration**

| File                         | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `config/config.yaml`         | **NEW**: Central config (training params, env params, paths) |
| `config/settings.py`         | **NEW**: Loads & validates config, provides defaults         |
| `config/logging_config.yaml` | **NEW**: Logging setup (levels, formats, handlers)           |

### **TESTS/ - Test Suite**

| File                      | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| `tests/test_agent.py`     | **NEW**: Unit tests for DDQNAgent              |
| `tests/test_env.py`       | **NEW**: Unit tests for WSNEnv & BatteryModel  |
| `tests/test_baselines.py` | **NEW**: Unit tests for baseline policies      |
| `tests/test_backend.py`   | **NEW**: Integration tests for Flask endpoints |

---

## Key Refactorings & Improvements

### **1. Split `env_wsn.py` into Two Files**

**Current problem:** BatteryModel and WSNEnv mixed together

**Solution:**

- `src/envs/battery_model.py` → Just `BatteryModel` class
- `src/envs/wsn_env.py` → Just `WSNEnv` class
- `src/envs/__init__.py` → Export both for convenience

**Benefit:** Can test battery dynamics independently of environment.

---

### **2. Create Base Agent Class**

**File:** `src/agents/base_agent.py`

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, state, eval_mode=False):
        pass

    @abstractmethod
    def learn_step(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass
```

**Benefit:** All agents (DDQN, future agents) inherit from same interface.

---

### **3. Extract Training Logic → `src/training/trainer.py`**

**Currently:** Training hardcoded in `train_final_ddqn.py`

**New:** Create reusable `Trainer` class:

```python
class Trainer:
    def __init__(self, agent, env, config, logger):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger

    def train(self, episodes, callbacks=None):
        """Generic training loop, reusable for different agents/envs."""
        pass

    def evaluate(self, episodes=10):
        """Evaluation loop."""
        pass

    def save_checkpoint(self, path):
        """Save agent and metrics."""
        pass
```

**Usage in scripts:**

```python
trainer = Trainer(agent, env, config, logger)
trainer.train(episodes=100)
trainer.save_checkpoint('results/models/best.pth')
```

**Benefit:** Training logic reusable across scripts and web backend.

---

### **4. Refactor Flask App → Async Tasks**

**Current problem:** `app.py` blocks on training requests

**Solution:**

**File:** `backend/tasks.py`

```python
from celery import Celery

celery = Celery('wsn_training')
celery.config_from_object('config.settings')

@celery.task
def train_model_task(config_dict):
    """Background task: train model asynchronously."""
    config = Config.from_dict(config_dict)
    trainer = Trainer(...)
    trainer.train(config.episodes)
    return {"status": "completed", "model_path": "..."}
```

**File:** `backend/routes.py`

```python
@app.route('/run_training', methods=['POST'])
def run_training():
    data = request.json
    task = train_model_task.delay(data)  # Returns immediately
    return jsonify({
        'status': 'queued',
        'task_id': task.id
    })

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = train_model_task.AsyncResult(task_id)
    return jsonify({
        'status': task.status,
        'result': task.result if task.ready() else None
    })
```

**Benefit:** Multiple training jobs can run in parallel; UI doesn't freeze.

---

### **5. Add Configuration Management**

**File:** `config/config.yaml`

```yaml
training:
  episodes: 100
  batch_size: 64
  lr: 1e-4
  gamma: 0.99

environment:
  num_nodes: 550
  arena_size: [500, 500]
  max_steps: 10000

paths:
  models: results/models
  metrics: results/metrics
  visualizations: results/visualizations

logging:
  level: INFO
  format: json
```

**File:** `config/settings.py`

```python
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class Config:
    training: dict
    environment: dict
    paths: dict

    @classmethod
    def load(cls, config_path='config/config.yaml'):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Benefit:** No hardcoded values; easy to change across entire app.

---

### **6. Add Input Validation**

**File:** `backend/schemas.py`

```python
from marshmallow import Schema, fields, validate

class TrainingRequestSchema(Schema):
    episodes = fields.Int(validate=validate.Range(min=1, max=1000))
    nodes = fields.Int(validate=validate.Range(min=10, max=10000))
    lr = fields.Float(validate=validate.Range(min=1e-6, max=1e-1))
    batch_size = fields.Int(validate=validate.Range(min=8, max=512))
```

**Usage:**

```python
schema = TrainingRequestSchema()
try:
    data = schema.load(request.json)
except ValidationError as e:
    return jsonify({'errors': e.messages}), 400
```

**Benefit:** Invalid requests rejected before training starts.

---

### **7. Centralized Logging**

**File:** `src/utils/logger.py`

```python
import logging
import logging.config
import yaml

def setup_logging(config_path='config/logging_config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    return logging.getLogger(__name__)

logger = setup_logging()
```

**Usage everywhere:**

```python
from src.utils.logger import logger

logger.info(f"Training started: {config}")
logger.debug(f"Episode 5, reward: {reward}")
logger.error(f"Training failed: {exc}")
```

**Benefit:** Structured logs; can be sent to file, cloud, monitoring systems.

---

### **8. Add Tests**

**File:** `tests/test_agent.py`

```python
import pytest
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv

def test_agent_initialization():
    agent = DDQNAgent(state_dim=100, action_dim=2, node_count=50)
    assert agent.state_dim == 100

def test_action_selection():
    agent = DDQNAgent(state_dim=100, action_dim=2, node_count=50)
    state = np.random.randn(100)
    action = agent.select_action(state)
    assert action.shape == (50,)
```

**Run tests:**

```bash
pytest tests/ -v
```

**Benefit:** Catch regressions; safe refactoring.

---

### **9. Documentation Structure**

| Document                  | Purpose                            |
| ------------------------- | ---------------------------------- |
| `README.md`               | Quick start, overview              |
| `docs/architecture.md`    | System design, module interactions |
| `docs/getting_started.md` | Installation, environment setup    |
| `docs/training_guide.md`  | How to run training (CLI & web)    |
| `docs/api.md`             | REST API endpoints & schemas       |
| `docs/contributing.md`    | Development workflow               |

---

## Migration Checklist

- [ ] Create new folder structure
- [ ] Create `src/agents/base_agent.py` with abstract class
- [ ] Move & refactor `ddqn_agent.py` to `src/agents/`
- [ ] Split `env_wsn.py` → `battery_model.py` and `wsn_env.py`
- [ ] Move `baselines.py` → `src/baselines/baseline_policies.py`
- [ ] Create `src/training/trainer.py` with reusable logic
- [ ] Create `src/utils/logger.py`, `metrics.py`, `visualization.py`
- [ ] Create `config/config.yaml` and `config/settings.py`
- [ ] Refactor `app.py` → `backend/app.py`, `routes.py`, `tasks.py`, `schemas.py`
- [ ] Add `requirements.txt` with all dependencies & versions
- [ ] Create `tests/` with unit tests
- [ ] Create `docs/` with documentation
- [ ] Update `scripts/train_model.py` to use new structure
- [ ] Update `scripts/evaluate_baselines.py` to use new structure
- [ ] Add `.env.example` template
- [ ] Update `.gitignore` to exclude results/, **pycache**/, .env

---

## Dependencies to Add (`requirements.txt`)

```
# Core ML
torch==2.0.1
gymnasium==0.28.1
numpy==1.24.3

# Web
flask==2.3.2
marshmallow==3.19.0
celery==5.3.0
redis==4.5.5

# Config
pyyaml==6.0

# Utilities
matplotlib==3.7.1
pandas==2.0.2

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Development
black==23.3.0
flake8==6.0.0
isort==5.12.0
```

---

## Benefits of This Structure

✅ **Modularity**: Each module has single responsibility  
✅ **Reusability**: Core logic in `src/` used by scripts, web backend, tests  
✅ **Scalability**: Easy to add new agents, envs, baseline policies  
✅ **Testability**: Clear boundaries make unit testing straightforward  
✅ **Maintainability**: Logical organization, clear dependencies  
✅ **Deployment**: Separate backend/frontend; easy to dockerize  
✅ **Documentation**: Structured docs for onboarding & reference  
✅ **CI/CD Ready**: Tests, linting, type checking can be automated
