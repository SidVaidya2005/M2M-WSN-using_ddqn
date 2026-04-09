# Agents & Training Rules

## BaseAgent Interface

All agents (including baselines) must implement:

```python
select_action(state, eval_mode=False) -> np.ndarray   # action vector length N
store_transition(state, action, reward, next_state, done)
learn_step() -> Optional[float]                        # returns loss or None
save_model(path: str)
load_model(path: str)
```

`eval_mode=True` disables exploration (epsilon=0 for DDQN, deterministic for baselines). Always pass `eval_mode=True` during benchmark evaluation.

## DDQNAgent Internals

- **Two networks**: `q_net` (policy) and `target_net` — target is updated periodically, not every step
- **Replay buffer**: experiences are stored in a circular buffer; `learn_step()` is a no-op until `len(buffer) >= batch_size`
- **Epsilon schedule**: decays from `epsilon_start` to `epsilon_end` over `epsilon_decay` steps — check `agent.epsilon` to monitor exploration
- **Gradient clipping**: applied in `learn_step()` to prevent exploding gradients; default clip norm is `10.0`

## DQNAgent vs DDQNAgent

| | DQNAgent | DDQNAgent |
|-|----------|-----------|
| Networks | 1 | 2 (policy + target) |
| Q-target | max Q from same network | action selected by policy, valued by target |
| Stability | lower | higher |

Default for all API calls is `"ddqn"`. Use `"dqn"` only for ablation comparisons.

## Trainer API

```python
trainer = Trainer(agent, env, seed=42)
rewards, metrics = trainer.train(episodes=100)      # returns (list[float], dict)
eval_rewards, _ = trainer.evaluate(episodes=10)     # no learning, no epsilon decay
trainer.save_checkpoint(path)                        # saves agent weights
```

**`train()` mutates agent state** (epsilon, replay buffer, network weights). `evaluate()` does not.

## Hyperparameter Ranges

| Parameter | Safe Range | Notes |
|-----------|-----------|-------|
| learning_rate | 1e-5 – 1e-3 | Start with 1e-4; reduce if loss spikes |
| gamma | 0.95 – 0.99 | 0.99 = longer horizon planning |
| batch_size | 32 – 256 | Larger = more stable, slower per step |
| episodes | 50 – 1000 | 550-node full-scale: ~45 min/100 ep on CPU |

**Seed controls**: initial weights, node positions, exploration noise — same seed = same results.

## Performance Notes

- PyTorch auto-selects CUDA if available; no manual device configuration needed
- For 550 nodes: expect ~45 min/100 episodes on CPU, ~8 min on GPU
- Replay buffer fills before learning starts — first `batch_size` steps produce no loss
