# Architecture Rules

## Design Patterns in Use

| Pattern | Where | Rule |
|---------|-------|------|
| Strategy | `BaseAgent` + agent subclasses | All agents must subclass `BaseAgent`; never call agent-specific methods directly from `Trainer` |
| Singleton | `get_config()` | Import config once via `get_config()`; never instantiate `Config` directly |
| Factory | `create_app()` in `backend/app.py` | Flask app is always created via `create_app()`; `app.run()` only in `if __name__ == "__main__"` |
| Composition | `Trainer` composes `agent + env` | `Trainer` owns the training loop; agents and envs do not call each other |

## Layer Responsibilities

```
frontend/          → display only; no business logic
backend/routes.py  → HTTP boundary: validate input, call tasks.py, return JSON
backend/tasks.py   → execution engine: construct env/agent, run Trainer, write artifacts
src/training/      → training loop only; no I/O, no Flask, no path construction
src/agents/        → Q-network math + replay buffer; no env knowledge
src/envs/          → simulation physics; no agent knowledge
config/            → read-only after startup; no side effects
```

**Never cross layers upward.** `src/` modules must not import from `backend/`.

## Two-Agent Architecture — DDQN + DQN Only

The system uses exactly **two agents**: DDQN (primary) and DQN (comparison).
All baseline policies have been removed. Benchmarking is exclusively DDQN-vs-DQN.

## Extension Points

- **New agent**: subclass `BaseAgent` in `src/agents/`, implement all abstract methods, add to `tasks.py` agent selection block (`agent_class = DDQNAgent if ... else DQNAgent`)
- **New route**: add handler to `backend/routes.py`, add schema to `backend/schemas.py`, add execution logic to `backend/tasks.py`
- **New metric**: add to `src/utils/metrics.py`, call from `Trainer._run_episode()`

## Data Flow (Training)

```
HTTP POST /api/train
  → schemas.py validates + applies defaults (nodes defaults to config.environment.num_nodes)
  → tasks.run_training() constructs env, agent, Trainer
  → Trainer.train() runs episodes
  → artifacts written: {run_id}_model.pth, {run_id}_plot.png, {run_id}_metadata.json
  → JSON response returned to frontend
```

## Data Flow (Async Training)

```
HTTP POST /api/train/async
  → schemas.py validates
  → tasks.submit_training_task() spawns daemon thread → returns task_id (UUID)
  → client polls GET /api/tasks/<task_id> for status
  → on completion, result is same as sync flow
```
