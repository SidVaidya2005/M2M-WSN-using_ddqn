"""Documentation main page."""

# WSN DDQN Training Platform - Documentation

Welcome to the WSN DDQN training platform documentation. This system provides a production-ready framework for training reinforcement learning agents to optimize wireless sensor network scheduling.

## Documentation Index

- **[Getting Started](getting_started.md)** - Installation and first run
- **[Architecture Overview](architecture.md)** - System design and components
- **[Training Guide](training_guide.md)** - How to train models
- **[API Reference](api.md)** - REST API endpoints and schemas
- **[Configuration](configuration.md)** - All configurable parameters
- **[Contributing](contributing.md)** - Development guidelines

## Quick Links

### For Users

1. **First Time?** Start with [Getting Started](getting_started.md)
2. **Want to train a model?** See [Training Guide](training_guide.md)
3. **Using the web interface?** Check [API Reference](api.md)

### For Developers

1. **Understanding the codebase?** Read [Architecture Overview](architecture.md)
2. **Contributing code?** Follow [Contributing Guidelines](contributing.md)
3. **Running tests?** See [configuration.md](configuration.md#testing)

## Key Concepts

### Double Deep Q-Network (DDQN)

Our agent uses DDQN to learn optimal WSN scheduling policies. DDQN:

- Uses two neural networks (policy and target) to reduce overestimation
- Maintains an experience replay buffer for sample efficiency
- Uses epsilon-greedy exploration with decaying epsilon schedule

### Environment

The WSN environment simulates:

- N sensor nodes with batteries (SoC and SoH)
- Sleep/Awake scheduling decisions
- Battery degradation from discharge cycles
- Coverage metrics
- Network lifetime until critical failure

### Baselines

For benchmarking, we compare against:

- **Random Policy**: Randomly choose awake/sleep
- **Greedy Policy**: Wake highest (SoC × SoH) nodes
- **Energy Conservative**: Wake only healthiest 20% of nodes
- **Balanced Rotation**: Rotate which nodes are awake periodically

## Project Statistics

- **Lines of Code**: ~3000+ (production code)
- **Test Coverage**: Recommended 70%+ (in development)
- **Documentation**: Comprehensive with examples
- **Dependencies**: ~20 core packages (see requirements.txt)

## Common Tasks

| Task               | Command                                        |
| ------------------ | ---------------------------------------------- |
| Install packages   | `pip install -r requirements.txt`              |
| Train model        | `python scripts/train_model.py --episodes 100` |
| Evaluate baselines | `python scripts/evaluate_baselines.py`         |
| Run web server     | `python -m backend.app`                        |
| Run tests          | `pytest tests/ -v`                             |
| Format code        | `black src/ backend/`                          |
| Check types        | `mypy src/ --ignore-missing-imports`           |

## Support

For issues or questions:

1. Check the relevant documentation file
2. Review examples in `/docs/` directory
3. Examine test cases in `/tests/` directory
4. Open a GitHub issue (if applicable)
