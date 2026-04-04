"""Unit tests for baseline policies."""

import numpy as np
import pytest

from src.baselines.baseline_policies import (
    RandomPolicy,
    GreedyPolicy,
    EnergyConservativePolicy,
    BalancedRotationPolicy,
)


@pytest.fixture
def policies(state_dim, node_count, action_dim):
    """All baseline policies initialized with the same parameters."""
    return [
        RandomPolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count),
        GreedyPolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count),
        EnergyConservativePolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count),
        BalancedRotationPolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count),
    ]


class TestBaselinePolicies:
    def test_action_shape(self, policies, sample_state, node_count):
        for policy in policies:
            action = policy.select_action(sample_state)
            assert action.shape == (node_count,), f"{type(policy).__name__} wrong shape"

    def test_action_values_valid(self, policies, sample_state, action_dim):
        for policy in policies:
            action = policy.select_action(sample_state)
            assert np.all(action >= 0) and np.all(action < action_dim), \
                f"{type(policy).__name__} has out-of-range actions"

    def test_store_transition_noop(self, policies, sample_state, node_count):
        action = np.zeros(node_count, dtype=np.int64)
        for policy in policies:
            policy.store_transition(sample_state, action, 0.0, sample_state, False)

    def test_learn_step_returns_none(self, policies):
        for policy in policies:
            result = policy.learn_step()
            assert result is None, f"{type(policy).__name__}.learn_step() should return None"

    def test_save_load_noop(self, policies, tmp_path):
        for policy in policies:
            path = str(tmp_path / f"{type(policy).__name__}.pth")
            policy.save_model(path)
            policy.load_model(path)

    def test_random_policy_randomness(self, state_dim, node_count, action_dim, sample_state):
        policy = RandomPolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count)
        actions = [policy.select_action(sample_state) for _ in range(20)]
        # With 20 samples, at least 2 distinct actions expected
        unique = set(tuple(a) for a in actions)
        assert len(unique) > 1

    def test_greedy_policy_deterministic(self, state_dim, node_count, action_dim, sample_state):
        policy = GreedyPolicy(state_dim=state_dim, action_dim=action_dim, node_count=node_count)
        a1 = policy.select_action(sample_state)
        a2 = policy.select_action(sample_state)
        np.testing.assert_array_equal(a1, a2)
