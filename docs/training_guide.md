# Training Guide

## Overview

This guide covers all aspects of training DDQN agents for WSN scheduling optimization.

## Training Modes

### 1. Command Line Training (Recommended)

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

- `results/trained_model.pth` - Neural network weights
- `results/training_metrics.json` - Performance metrics
- `results/training_curve.png` - Loss visualization

### 2. Web Interface Training

Best for interactive exploration and non-technical users.

```bash
# Start server
python -m flask --app backend.app run

# Visit http://localhost:5000
```

**Advantages:**

- Visual form for parameters
- Real-time progress updates
- Download results directly
- No command line needed

**Disadvantages:**

- Slower for multiple runs
  -No easy parameter sweep
- Session-based (loses progress if connection drops)

### 3. Programmatic Training

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

## Training Workflow

### Step 1: Configuration

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

### Step 2: Start Training

```bash
python scripts/train_model.py --episodes 100 --nodes 50 --seed 42
```

**Monitor Progress:**

- Watch for increasing rewards over time
- 10-episode moving average should increase
- Took ~5-10 minutes per 100 episodes on CPU

### Step 3: Check Results

```bash
# View metrics
cat results/training_metrics.json

# Plot training curve
python -c "
import json
import matplotlib.pyplot as plt
with open('results/training_metrics.json') as f:
    data = json.load(f)
plt.plot(data['training']['rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('results/my_plot.png')
"
```

### Step 4: Evaluate Model

```bash
python scripts/evaluate_baselines.py --model results/trained_model.pth
```

**Outputs:**

- Mean reward vs baselines
- Coverage comparison
- Energy efficiency metrics

## Hyperparameter Tuning

### Grid Search

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

### Manual Tuning Guidelines

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

## Advanced Training

### Continue Training from Checkpoint

```python
agent = DDQNAgent(...)
agent.load_model('results/trained_model.pth')

trainer = Trainer(agent, env)
rewards, metrics = trainer.train(episodes=100)  # Train 100 more
trainer.save_checkpoint('results/trained_model_v2.pth')
```

### Custom Reward Function

Modify `src/envs/wsn_env.py` step() method:

```python
# Default
reward = 10.0 * r_coverage + 5.0 * r_energy + 1.0 * r_soh + 2.0 * r_balance

# Custom: prioritize coverage more
reward = 15.0 * r_coverage + 3.0 * r_energy + 1.0 * r_soh + 1.0 * r_balance

# Custom: prioritize energy efficiency
reward = 5.0 * r_coverage + 10.0 * r_energy + 1.0 * r_soh + 1.0 * r_balance
```

Then retrain and compare:

```bash
python scripts/train_model.py \
  --episodes 100 \
  --output-dir results/custom_reward
```

### Distributed Training (Advanced)

For training multiple seeds in parallel:

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

# Compute mean and variance
mean_reward = np.mean([r['training']['mean_reward'] for r in results])
std_reward = np.std([r['training']['mean_reward'] for r in results])
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

## Troubleshooting Training

### Problem: Rewards Not Increasing

**Likely causes:**

1. Learning rate too low
2. Not enough episodes
3. Hyperparameters bad

**Solutions:**

```bash
# Try higher learning rate
python scripts/train_model.py --lr 1e-3 --episodes 200

# Try different gamma
python scripts/train_model.py --gamma 0.95 --episodes 200
```

### Problem: Training Too Slow

**Likely causes:**

1. Too many nodes
2. Too large batch size
3. CPU instead of GPU

**Solutions:**

```bash
# Reduce complexity
python scripts/train_model.py --nodes 100 --batch-size 32

# Use GPU (if available)
# Edit script to use device='cuda'
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

## Reproducibility

To ensure reproducible results:

```bash
python scripts/train_model.py \
  --episodes 100 \
  --nodes 50 \
  --lr 1e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --seed 42  # IMPORTANT
```

The seed controls:

- Initial network weights
- Node positions
- Exploration randomness

Same seed → same results (always).

## Best Practices

1. **Use meaningful output directories:**

   ```bash
   --output-dir results/experiment_name
   ```

2. **Log your experiments:**

   ```bash
   tee results/experiment.log > >(python scripts/train_model.py ...)
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

## Next Steps

- [Deploy to Web Server](api.md)
- [Modify Environment Dynamics](architecture.md#environments)
- [Create Custom Baselines](architecture.md#baselines)
- [Integrate with MLflow](https://mlflow.org/docs/latest/tracking.html)
