# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also:** root [`../CLAUDE.md`](../CLAUDE.md) for project-wide commands and architecture, [`../backend/CLAUDE.md`](../backend/CLAUDE.md) for the Flask layer, and [`../frontend/CLAUDE.md`](../frontend/CLAUDE.md) for the UI layer.

## Scope

`src/` is the **pure RL core**. It contains the environment simulation, the agents, the training loop, and shared utilities. It has **no knowledge of Flask, HTTP, file paths, or the frontend** — those live in `backend/`. Layer rule: `src/` modules must never import from `backend/`.

## Layout

```
src/
├── agents/
│   ├── base_agent.py    — BaseAgent ABC (the strategy interface)
│   ├── ddqn_agent.py    — DDQNAgent: policy + target net, decoupled Bellman target (primary)
│   └── dqn_agent.py     — DQNAgent: subclass of DDQNAgent, overrides only the target calc (ablation only)
├── envs/
│   ├── wsn_env.py       — Gymnasium WSNEnv: per-node SLEEP/AWAKE actions
│   └── battery_model.py — SoC/SoH dynamics, cycle + calendar degradation (SoH never recovers)
├── training/
│   └── trainer.py       — Episode loop: select_action → step → store_transition → learn_step
└── utils/
    ├── logger.py
    ├── metrics.py       — Add new metrics here; call from Trainer._run_episode()
    └── visualization.py
```

## Detailed Rules

The authoritative rules for code in this directory live in [`../.claude/rules/`](../.claude/rules):

| File | Covers |
|------|--------|
| [`rl-environment.md`](../.claude/rules/rl-environment.md) | `WSNEnv` API contract, observation/action space, reward weights, `BatteryModel` |
| [`agents-training.md`](../.claude/rules/agents-training.md) | `BaseAgent` interface, DDQN internals, `Trainer` API, hyperparameters |
| [`architecture.md`](../.claude/rules/architecture.md) | Layer responsibilities, extension points, two-agent (DDQN+DQN only) policy |
| [`config-paths.md`](../.claude/rules/config-paths.md) | `get_config()` singleton usage from inside `src/` |

Read these before modifying agent math, env dynamics, or the training loop.

## Critical contracts (do not break)

- **`BaseAgent` is the only path** between `Trainer` and any agent. Never call agent-specific methods from `Trainer`.
- **`Trainer` owns the loop.** Agents do not know about envs; envs do not know about agents.
- **`WSNEnv.reset()` and `step()` signatures** are documented in [`rl-environment.md`](../.claude/rules/rl-environment.md) — check before touching, the project has historically had bugs from mis-unpacking these.
- **Two agents only**: DDQN (primary) and DQN (ablation comparison). No baseline policies — they were removed in Phase 0.
- **`state_dim` is derived from the env**, never hardcoded: `env.observation_space.shape[0]`.

## Extending

- **New agent** → subclass `BaseAgent`, implement all abstract methods, then wire into `backend/tasks.py` agent selection. Do not import anything from `backend/` here.
- **New metric** → add to `utils/metrics.py`, call from `Trainer._run_episode()`.
- **New env** → subclass `gym.Env`, follow the existing `WSNEnv` info-dict shape so downstream metrics keep working.
