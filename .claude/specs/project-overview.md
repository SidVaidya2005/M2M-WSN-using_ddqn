# Project Overview Spec

## What This Project Is

A research-grade Deep Reinforcement Learning (RL) platform for optimizing Wireless Sensor Network (WSN) scheduling using Machine-to-Machine (M2M) cooperative behaviors. The system uses **DDQN (Double Deep Q-Network)** as its primary agent and includes a **DQN** agent for ablation comparison.

## Core Problem

Sensor nodes in a WSN have limited battery. The RL agent must learn to schedule nodes into SLEEP or AWAKE states each timestep to:
1. **Maximize network coverage** (enough nodes awake to cover the sensing area)
2. **Minimize energy consumption** (avoid draining batteries unnecessarily)
3. **Preserve battery health (SoH)** (deep discharge cycles degrade batteries)
4. **Balance load across nodes** (fair distribution of duty cycles)

## Two-Agent Design

| Agent | Role | Bellman Target |
|-------|------|---------------|
| **DDQN** | Primary — all API calls default to this | `y = r + γ * Q_target(s', argmax_a Q_online(s', a))` |
| **DQN** | Comparison only — exists for academic ablation | `y = r + γ * max_a Q_target(s', a)` |

All baseline policies (Random, Greedy, EnergyConservative, BalancedRotation) have been **removed**. Benchmarking is exclusively DDQN-vs-DQN.

## M2M-Specific Behaviors (Phase 2)

These features distinguish this from a generic WSN scheduler:
1. **Per-node charging** — nodes with SoC below threshold enter charging state, forced SLEEP
2. **SoH-aware decisions** — battery health is part of the observation, influencing agent decisions
3. **Cooperative wake-up** — when a node's SoC drops to ≤50%, its nearest SLEEP neighbor is automatically awakened (M2M cooperation, applied as a post-action environment rule)

## Technology Stack

- **Backend:** Python 3, Flask REST API
- **Frontend:** HTML/JS/CSS (served via Flask, no build step)
- **RL:** PyTorch, Gymnasium
- **Config:** YAML + Python dataclasses
- **Validation:** Marshmallow schemas
- **Tests:** pytest

## Directory Layout

```
WSN_M2M/
├── config/              # YAML config + Python dataclasses
│   ├── config.yaml      # Single source of truth for all parameters
│   └── settings.py      # Dataclass hierarchy + singleton accessor
├── src/                 # Core RL library (no I/O, no Flask)
│   ├── agents/          # BaseAgent ABC, DDQNAgent, DQNAgent
│   ├── envs/            # WSNEnv (Gymnasium), BatteryModel
│   ├── training/        # Trainer loop
│   └── utils/           # logging, metrics, visualization
├── backend/             # Flask REST API
│   ├── app.py           # App factory
│   ├── routes.py        # Route handlers
│   ├── schemas.py       # Marshmallow validation
│   └── tasks.py         # Training execution engine
├── frontend/            # Static web UI
│   ├── templates/       # index.html
│   └── static/          # JS, CSS
├── scripts/             # CLI entry points
│   └── train.py         # CLI training wrapper
├── tests/               # pytest suite
├── results/             # Output artifacts (models, metrics, plots)
└── plan.md              # Phased restructuring plan
```

## Restructure Plan Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Cleanup & directory restructure | ✅ Complete |
| 1 | Config & hardcode elimination | ✅ Complete |
| 2 | Environment: coverage, charging, SoH, cooperative wake-up | 🔲 Pending |
| 3 | Training output contract (JSON + 4-panel visualization) | 🔲 Pending |
| 4 | Frontend alignment | 🔲 Pending |
| 5 | Scripts & post-training comparison | 🔲 Pending |
| 6 | Tests, docs, final pass | 🔲 Pending |
