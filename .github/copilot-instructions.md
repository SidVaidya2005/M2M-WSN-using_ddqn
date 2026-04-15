# WSN DDQN Training Platform Guidelines

## Code Style & Typing

- **Type Safety on Gym Spaces**: Pyright may treat `observation_space` and its `shape` attribute as `Optional`. When dealing with spaces, assign them to local variables and explicitly guard against `None` before attempting to index `shape[0]` to prevent silent typing crashes.

## Architecture & RL Conventions

- **Modular Boundaries**: Core RL logic (`src/agents/`, `src/envs/`) must remain strictly decoupled from the web application layers (`backend/`).
- **Research Orientation**: Ensure any changes or metric outputs align with benchmark reporting in `results/`, as the platform is geared toward comparing heuristic baselines against the DDQN model.
- **Deterministic Evaluation**: In RL agents, keep evaluation deterministic. Always ensure epsilon is set to `0` in eval mode; nonzero eval epsilon can mask policy behavior and look like scheduling bugs.

## Build and Run

- **Setup**: `pip install -r requirements.txt` (and copy `.env.example` to `.env` if available)
- **Train CLI**: `python scripts/train_model.py --episodes 100 --nodes 50 --lr 1e-4`
- **Evaluate**: `python scripts/evaluate_baselines.py --model results/models/trained_model.pth --episodes 10`
- **Web UI**: `python -m backend.app` (runs Flask on `http://localhost:5000`)

## Documentation & References

Refer to the `docs/` directory for detailed context rather than duplicating patterns here:

- **Architecture**: See [docs/architecture.md](docs/architecture.md)
- **Training**: See [docs/training_guide.md](docs/training_guide.md)
- **API**: See [docs/api.md](docs/api.md)
- **Getting Started**: See [docs/getting_started.md](docs/getting_started.md)


## for more information
@CLAUDE.md
@.claude/