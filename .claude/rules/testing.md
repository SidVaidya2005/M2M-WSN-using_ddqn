# Testing Rules

## Test Files

| File | Covers |
|------|--------|
| `tests/test_agent.py` | `DDQNAgent`, `DQNAgent` — unit tests, no env needed |
| `tests/test_env.py` | `WSNEnv`, `BatteryModel` — environment dynamics |
| `tests/test_baselines.py` | All baseline policies via `BaseAgent` interface |
| `tests/test_backend.py` | Flask routes — uses `app.test_client()` |

## Config Singleton Reset (Critical)

Any test file that imports from `config.settings` must reset the singleton before and after the session:

```python
import config.settings as settings_module

@pytest.fixture(autouse=True, scope="session")
def reset_config():
    settings_module._config = None
    yield
    settings_module._config = None
```

Without this, a test that sets config first will silently contaminate all later tests in the same session.

## Small Node Fixture

Tests use `N_NODES=10` to stay fast. The env fixture in `conftest.py`:

```python
@pytest.fixture
def small_env():
    return WSNEnv(N=10, max_steps=50, death_threshold=0.3)
```

Never use the production default of 550 nodes in tests — it makes the suite unusable.

## Agent Test Pattern

```python
agent = DDQNAgent(state_dim=50, action_dim=2, node_count=10, batch_size=8)
# state_dim = N * 5 features
state = np.zeros(50)
action = agent.select_action(state)
assert action.shape == (10,)
assert set(action).issubset({0, 1})
```

## Backend Test Pattern

```python
@pytest.fixture
def client(app):
    return app.test_client()

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["status"] == "healthy"
```

The Flask app fixture must set `app.config["CONFIG"]` — the `conftest.py` creates a minimal config for this.

## Running Tests

```bash
pytest tests/                                          # all tests
pytest tests/test_agent.py                             # single file
pytest tests/test_agent.py::TestDDQNAgent::test_init  # single test
pytest tests/ -x                                       # stop on first failure
pytest tests/ -v                                       # verbose output
```
