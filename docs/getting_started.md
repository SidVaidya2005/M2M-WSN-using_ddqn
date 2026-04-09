# Getting Started

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Redis for background jobs

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

## First Training Run

### Option A: Command Line (Recommended for First Run)

```bash
python scripts/train_model.py \
  --episodes 10 \
  --nodes 50 \
  --lr 1e-4 \
  --seed 42
```

**Parameters:**

- `--episodes`: Number of training episodes (start small: 10-50)
- `--nodes`: Number of sensor nodes (start small: 50-100)
- `--lr`: Learning rate (default: 1e-4)
- `--seed`: Random seed for reproducibility
- `--output-dir`: Where to save results (default: results)

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

### Option B: Web Server

```bash
python -m backend.app
```

Then visit `http://localhost:5001` in your browser.

**Form Fields:**

- **Episodes**: Training episodes (1-1000)
- **Nodes**: Number of nodes (10-10000)
- **Learning Rate**: 0.0001 to 0.1
- **Gamma**: 0.0 to 1.0

## Understanding Results

### Generated Files

After training, check `results/`:

```
results/
├── models/run_{timestamp}_model.pth              # Neural network weights
├── metrics/run_{timestamp}_metadata.json          # Per-run config + summary metrics
└── visualizations/run_{timestamp}_plot.png        # Training progress plot
```

Run IDs have the format `run_YYYYMMDD_HHMMSS` (e.g. `run_20260406_080528`).

### Metrics Explained

**Metrics JSON** (`run_{timestamp}_metadata.json`):

```json
{
  "run_id": "run_20260406_080528",
  "timestamp": "2026-04-06T08:06:10.290101",
  "config": {
    "model_type": "ddqn",
    "episodes": 10,
    "nodes": 50,
    "learning_rate": 0.0001
  },
  "metrics": {
    "mean_reward": 145.32,
    "max_reward": 180.5,
    "best_episode": 7,
    "avg_final_10": 172.4
  },
  "image_url": "/api/visualizations/run_20260406_080528_plot.png"
}
```

**Key Metrics:**

- `mean_reward`: Average reward per episode (higher is better)
- `mean_coverage`: Fraction of nodes kept awake (0-1)
- `final_soh`: Battery health at end (0-1, higher is better)
- `final_dead_nodes`: Number of failed nodes

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
pip install torch
```

### Issue: `CUDA out of memory`

**Solution:** Use CPU instead:

```python
# In your training code
agent = DDQNAgent(..., device='cpu')
```

### Issue: Training is very slow

**Solutions:**

- Reduce number of nodes: `--nodes 100`
- Reduce episodes: `--episodes 10`
- Reduce batch size in config: edit `config/config.yaml`

### Issue: `Config file not found`

**Solution:**

```bash
# Make sure you're in the project root
cd m2m_ddqn
python scripts/train_model.py ...
```

## Next Steps

1. **Understand the Code**: Read [Architecture Overview](architecture.md)
2. **Train a Real Model**: Using full settings:
   ```bash
   python scripts/train_model.py --episodes 500 --nodes 50
   ```
3. **Evaluate Baselines**: Compare against reference policies
   ```bash
   python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth --episodes 20
   ```
4. **Modify the Environment**: Edit `src/envs/wsn_env.py` to customize
5. **Deploy**: Run web server for remote training

## Common Modifications

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

Edit [src/agents/ddqn_agent.py](../src/agents/ddqn_agent.py):

```python
agent = DDQNAgent(
    ...,
    hidden_dims=[1024, 512, 256],  # Deeper network
    lr=1e-5,                         # Lower learning rate
)
```

## Getting Help

- **Documentation**: See [docs/](.) directory
- **Code Examples**: Check [scripts/](../scripts/) directory
- **Tests**: Review [tests/](../tests/) directory
- **Configuration**: Read [config/config.yaml](../config/config.yaml)
