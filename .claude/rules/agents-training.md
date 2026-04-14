# Agents & Training Rules

## BaseAgent Interface

All agents must implement:

```python
select_action(state, eval_mode=False) -> np.ndarray   # action vector length N
store_transition(state, action, reward, next_state, done)
learn_step() -> Optional[float]                        # returns loss or None
save_model(path: str)
load_model(path: str)
```

`eval_mode=True` disables exploration (epsilon=0). Always pass `eval_mode=True` during evaluation runs.

## DDQNAgent — Primary Agent

- **Two networks**: `q_net` (policy) and `target_net` — target is updated every `target_update_frequency` steps
- **Bellman target** (decoupled action selection/evaluation):
  ```
  a*  = argmax_a Q_online(s', a)          # online picks action
  y   = r + γ * Q_target(s', a*) * (1-done)   # target evaluates it
  ```
  This reduces maximization bias, giving more stable learning on multi-objective WSN reward
- **Replay buffer**: circular, size `replay_buffer_size`; `learn_step()` is a no-op until `len(buffer) >= min_replay_size`
- **Epsilon schedule**: decays from `epsilon_start` to `epsilon_end` over `epsilon_decay` steps
- **Gradient clipping**: applied in `learn_step()` with default clip norm `10.0`

## DQNAgent — Comparison Agent

| | DQNAgent | DDQNAgent |
|-|----------|-----------|
| Networks | Shares DDQN architecture | 2 (policy + target) |
| Q-target | `y = r + γ * max_a Q_target(s', a) * (1-done)` | Decoupled (see above) |
| Stability | lower | higher |
| Purpose | Ablation comparison only | Primary model for all API calls |

`DQNAgent` is a subclass of `DDQNAgent` that overrides only the target computation in `learn_step()`. It exists solely for DDQN-vs-DQN comparison graphs.

Default for all API calls is `"ddqn"`. Use `"dqn"` only for ablation comparisons.

## Trainer API

```python
trainer = Trainer(agent, env, seed=42)
rewards, metrics = trainer.train(episodes=100)      # returns (list[float], dict)
trainer.save_checkpoint(path)                        # saves agent weights

# Per-episode series (populated during train())
trainer.episode_series   # dict of lists: episode_reward, coverage, avg_soh,
                         #                 alive_fraction, mean_soc, step_counts

# Network lifetime — episode index where alive_fraction first drops below
# (1 - death_threshold); equals total episodes if never breached
trainer.network_lifetime  # int
```

**`train()` mutates agent state** (epsilon, replay buffer, network weights).

`state_dim` must always be derived from the env (`env.observation_space.shape[0]`).
With 6 features per node: `state_dim = N * 6` (e.g. 300 for N=50, 60 for tests with N=10).

## Hyperparameter Ranges

| Parameter | Safe Range | Notes |
|-----------|-----------|-------|
| learning_rate | 1e-5 – 1e-3 | Start with 1e-4; reduce if loss spikes |
| gamma | 0.95 – 0.99 | 0.99 = longer horizon planning |
| batch_size | 32 – 256 | Larger = more stable, slower per step |
| episodes | 50 – 1000 | 50-node default: fast on CPU |

**Seed controls**: initial weights, node positions, exploration noise — same seed = same results.

## Performance Notes

- PyTorch auto-selects CUDA if available; no manual device configuration needed
- For 50 nodes: training is fast on CPU
- Replay buffer fills before learning starts — first `min_replay_size` steps produce no loss
