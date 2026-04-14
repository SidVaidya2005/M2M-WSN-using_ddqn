"""Shared pytest fixtures for the WSN DDQN test suite."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on sys.path when running tests from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Use a small node count to keep tests fast
N_NODES = 10
STATE_DIM = N_NODES * 6  # 6 features per node (Phase 2: added charging_flag)
ACTION_DIM = 2


@pytest.fixture
def node_count():
    return N_NODES


@pytest.fixture
def state_dim():
    return STATE_DIM


@pytest.fixture
def action_dim():
    return ACTION_DIM


@pytest.fixture
def sample_state():
    """A random observation vector matching WSNEnv output shape."""
    rng = np.random.default_rng(42)
    return rng.random(STATE_DIM).astype(np.float32)


@pytest.fixture
def wsn_env():
    """A small WSNEnv for integration tests."""
    from src.envs.wsn_env import WSNEnv
    env = WSNEnv(
        N=N_NODES,
        arena_size=(100, 100),
        sink=(50, 50),
        max_steps=50,
        death_threshold=0.3,
        seed=42,
    )
    yield env
    env.close()


@pytest.fixture
def ddqn_agent(state_dim, node_count, action_dim):
    from src.agents.ddqn_agent import DDQNAgent
    return DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_count=node_count,
        lr=1e-4,
        gamma=0.99,
        batch_size=8,
    )


@pytest.fixture
def flask_client():
    """Flask test client with test configuration."""
    import config.settings as settings_module
    # Reset singleton so each test session loads a fresh config
    settings_module._config = None

    from backend.app import create_app
    app = create_app(config_path=str(_project_root / "config" / "config.yaml"))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

    # Reset singleton after tests so other fixtures aren't affected
    settings_module._config = None
