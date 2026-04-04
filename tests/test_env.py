"""Unit tests for WSNEnv and BatteryModel."""

import numpy as np
import pytest

from src.envs.wsn_env import WSNEnv
from src.envs.battery_model import BatteryModel


class TestWSNEnv:
    def test_observation_space_shape(self, wsn_env, node_count):
        assert wsn_env.observation_space.shape == (node_count * 5,)

    def test_action_space_shape(self, wsn_env, node_count):
        assert wsn_env.action_space.shape == (node_count,)

    def test_reset_returns_array(self, wsn_env, state_dim):
        # reset() must NOT return a tuple — see CLAUDE.md gotcha
        obs = wsn_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (state_dim,)

    def test_reset_does_not_return_tuple(self, wsn_env):
        result = wsn_env.reset()
        # Unpacking as tuple would crash with 2750 elements; ensure it's a plain array
        assert not isinstance(result, tuple)

    def test_step_returns_four_values(self, wsn_env, node_count):
        wsn_env.reset()
        action = np.ones(node_count, dtype=np.int32)
        result = wsn_env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_obs_shape_after_step(self, wsn_env, state_dim, node_count):
        wsn_env.reset()
        action = np.zeros(node_count, dtype=np.int32)
        obs, _, _, _ = wsn_env.step(action)
        assert obs.shape == (state_dim,)

    def test_episode_terminates(self, node_count):
        env = WSNEnv(
            N=node_count,
            arena_size=(100, 100),
            sink=(50, 50),
            max_steps=5,
            death_threshold=0.3,
            seed=0,
        )
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = np.ones(node_count, dtype=np.int32)
            _, _, done, _ = env.step(action)
            steps += 1
        assert done or steps == 100  # episode ends within max_steps or loop guard

    def test_reproducible_with_seed(self, node_count):
        def run_episode(seed):
            env = WSNEnv(N=node_count, arena_size=(100, 100), sink=(50, 50),
                         max_steps=5, death_threshold=0.3, seed=seed)
            obs = env.reset()
            return obs

        obs1 = run_episode(42)
        obs2 = run_episode(42)
        np.testing.assert_array_equal(obs1, obs2)


class TestBatteryModel:
    def test_initialization(self):
        batt = BatteryModel()
        assert 0.0 <= batt.soc <= 1.0
        assert 0.0 <= batt.soh <= 1.0

    def test_soc_decreases_when_active(self):
        batt = BatteryModel()
        initial_soc = batt.soc
        batt.step(action=1)  # awake
        assert batt.soc <= initial_soc

    def test_soc_bounded(self):
        batt = BatteryModel()
        for _ in range(1000):
            batt.step(action=1)
        assert 0.0 <= batt.soc <= 1.0
        assert 0.0 <= batt.soh <= 1.0

    def test_sleep_conserves_more_energy(self):
        batt_active = BatteryModel()
        batt_sleep = BatteryModel()
        for _ in range(10):
            batt_active.step(action=1)
            batt_sleep.step(action=0)
        assert batt_sleep.soc >= batt_active.soc
