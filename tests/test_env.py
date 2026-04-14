"""Unit tests for WSNEnv and BatteryModel."""

import numpy as np
import pytest

from src.envs.wsn_env import WSNEnv
from src.envs.battery_model import BatteryModel


# ---------------------------------------------------------------------------
# WSNEnv tests
# ---------------------------------------------------------------------------

class TestWSNEnv:
    def test_observation_space_shape(self, wsn_env, node_count):
        # Phase 2: 6 features per node
        assert wsn_env.observation_space.shape == (node_count * 6,)

    def test_action_space_shape(self, wsn_env, node_count):
        assert wsn_env.action_space.shape == (node_count,)

    def test_reset_returns_obs_info_tuple(self, wsn_env, node_count):
        # reset() is Gymnasium-compliant and returns (obs, info)
        result = wsn_env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (node_count * 6,)
        assert isinstance(info, dict)

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

    def test_obs_shape_after_step(self, wsn_env, node_count):
        wsn_env.reset()
        action = np.zeros(node_count, dtype=np.int32)
        obs, _, _, _ = wsn_env.step(action)
        assert obs.shape == (node_count * 6,)

    def test_obs_values_in_range(self, wsn_env, node_count):
        wsn_env.reset()
        action = np.ones(node_count, dtype=np.int32)
        obs, _, _, _ = wsn_env.step(action)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_info_dict_has_required_keys(self, wsn_env, node_count):
        wsn_env.reset()
        action = np.ones(node_count, dtype=np.int32)
        _, _, _, info = wsn_env.step(action)
        for key in ("coverage", "avg_soh", "alive_fraction", "dead_count",
                    "mean_soc", "cooperative_wakes", "charging_count", "step_count"):
            assert key in info, f"Missing info key: {key}"

    def test_episode_terminates_at_max_steps(self, node_count):
        env = WSNEnv(N=node_count, arena_size=(100, 100), sink=(50, 50),
                     max_steps=5, death_threshold=0.3, seed=0)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = env.step(np.ones(node_count, dtype=np.int32))
            steps += 1
        assert done

    def test_reproducible_with_seed(self, node_count):
        def first_obs(seed):
            env = WSNEnv(N=node_count, arena_size=(100, 100), sink=(50, 50),
                         max_steps=5, death_threshold=0.3, seed=seed)
            obs, _ = env.reset()
            return obs

        obs1 = first_obs(42)
        obs2 = first_obs(42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_coverage_in_info(self, node_count):
        env = WSNEnv(N=node_count, arena_size=(100, 100), sink=(50, 50),
                     max_steps=50, death_threshold=0.3, seed=1,
                     sensing_radius=200.0)  # large radius → full coverage
        env.reset()
        # All nodes awake → coverage should approach 1.0
        _, _, _, info = env.step(np.ones(node_count, dtype=np.int32))
        assert 0.0 <= info["coverage"] <= 1.0

    def test_all_sleep_reduces_coverage(self, node_count):
        env = WSNEnv(N=node_count, arena_size=(100, 100), sink=(50, 50),
                     max_steps=50, death_threshold=0.3, seed=2,
                     sensing_radius=10.0)  # small radius
        env.reset()
        _, _, _, info_awake = env.step(np.ones(node_count, dtype=np.int32))
        _, _, _, info_sleep = env.step(np.zeros(node_count, dtype=np.int32))
        # All-sleep means 0 coverage
        assert info_sleep["coverage"] == 0.0
        # All-awake should have higher or equal coverage than all-sleep
        assert info_awake["coverage"] >= info_sleep["coverage"]


# ---------------------------------------------------------------------------
# Phase 2: Charging behaviour tests
# ---------------------------------------------------------------------------

class TestCharging:
    def test_node_enters_charging_when_low_soc(self):
        """A node with SoC below charging_threshold should be forced SLEEP and marked charging."""
        env = WSNEnv(
            N=3,
            arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            charging_enabled=True, charging_rate=0.05, charging_threshold=0.2,
        )
        env.reset()
        # Drain node 0's battery to below 20% (< 20 SoC out of 100)
        env.batteries[0].soc = 15.0  # 15% < threshold 20%
        action = np.ones(3, dtype=np.int32)  # all nodes want to be AWAKE
        _, _, _, info = env.step(action)
        assert info["charging_count"] >= 1, "Expected at least one charging node"
        assert env.batteries[0].charging is True

    def test_charging_node_forced_sleep_in_obs(self):
        """Charging flag in the observation should be 1 for a charging node."""
        env = WSNEnv(
            N=2, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            charging_enabled=True, charging_rate=0.05, charging_threshold=0.2,
        )
        env.reset()
        env.batteries[0].soc = 10.0  # force into charging
        obs, _, _, _ = env.step(np.ones(2, dtype=np.int32))
        # charging_flag is index 5 (0-based) in each 6-feature block
        charging_flag_node0 = obs[5]
        assert charging_flag_node0 == 1.0, "Charging flag should be 1 for a charging node"

    def test_node_exits_charging_at_high_soc(self):
        """A charging node whose SoC reaches 95% should exit charging state."""
        env = WSNEnv(
            N=2, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            charging_enabled=True, charging_rate=0.05, charging_threshold=0.2,
        )
        env.reset()
        env.batteries[0].soc = 10.0
        env.batteries[0].charging = True
        # Pump SoC high enough to exit
        env.batteries[0].soc = 96.0  # > 95 %
        env.step(np.zeros(2, dtype=np.int32))
        assert env.batteries[0].charging is False

    def test_soc_recovers_while_charging(self):
        """A charging battery's SoC should increase each step."""
        env = WSNEnv(
            N=2, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            charging_enabled=True, charging_rate=0.05, charging_threshold=0.2,
        )
        env.reset()
        env.batteries[0].soc = 15.0
        soc_before = env.batteries[0].soc
        env.step(np.ones(2, dtype=np.int32))
        soc_after = env.batteries[0].soc
        assert soc_after > soc_before, "SoC should increase during charging"


# ---------------------------------------------------------------------------
# Phase 2: Cooperative wake-up tests
# ---------------------------------------------------------------------------

class TestCooperativeWakeUp:
    def test_low_soc_awake_node_wakes_neighbour(self):
        """An AWAKE node at ≤ 50% SoC should cause its nearest SLEEP neighbour to wake."""
        env = WSNEnv(
            N=3, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            wake_cooperation_soc=0.5, charging_enabled=False,
        )
        env.reset()
        # Node 0: low SoC awake; nodes 1,2 sleeping
        env.batteries[0].soc = 40.0   # 40% < 50% threshold
        action = np.array([1, 0, 0], dtype=np.int32)  # only node 0 awake
        _, _, _, info = env.step(action)
        assert len(info["cooperative_wakes"]) >= 1, "Expected at least one cooperative wake"

    def test_high_soc_awake_node_does_not_trigger_wake(self):
        """An AWAKE node above the SoC threshold should NOT trigger cooperative wake."""
        env = WSNEnv(
            N=3, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=0,
            wake_cooperation_soc=0.5, charging_enabled=False,
        )
        env.reset()
        env.batteries[0].soc = 80.0   # 80% > 50% threshold
        action = np.array([1, 0, 0], dtype=np.int32)
        _, _, _, info = env.step(action)
        assert info["cooperative_wakes"] == [], "Should not wake any neighbour when SoC is high"

    def test_cooperative_wakes_recorded_in_info(self):
        env = WSNEnv(
            N=4, arena_size=(100, 100), sink=(50, 50),
            max_steps=50, death_threshold=0.3, seed=5,
            wake_cooperation_soc=0.5, charging_enabled=False,
        )
        env.reset()
        env.batteries[0].soc = 30.0
        env.batteries[1].soc = 30.0
        action = np.array([1, 1, 0, 0], dtype=np.int32)
        _, _, _, info = env.step(action)
        # cooperative_wakes must be a list
        assert isinstance(info["cooperative_wakes"], list)


# ---------------------------------------------------------------------------
# BatteryModel tests
# ---------------------------------------------------------------------------

class TestBatteryModel:
    def test_initialization(self):
        batt = BatteryModel(E_max=100.0)
        assert batt.soc == 100.0
        assert 0.0 <= batt.soh <= 1.0
        assert batt.charging is False

    def test_discharge_reduces_soc(self):
        batt = BatteryModel(E_max=100.0)
        initial_soc = batt.soc
        batt.discharge(10.0)
        assert batt.soc < initial_soc

    def test_soc_cannot_go_negative(self):
        batt = BatteryModel(E_max=100.0)
        batt.discharge(1000.0)
        assert batt.soc == 0.0

    def test_soh_decreases_on_discharge(self):
        batt = BatteryModel(E_max=100.0)
        initial_soh = batt.soh
        for _ in range(100):
            batt.discharge(5.0)
        assert batt.soh < initial_soh

    def test_charge_increases_soc(self):
        batt = BatteryModel(E_max=100.0)
        batt.soc = 20.0
        batt.charge(rate=0.1)
        assert batt.soc > 20.0

    def test_charge_does_not_exceed_e_max(self):
        batt = BatteryModel(E_max=100.0)
        batt.soc = 99.0
        batt.charge(rate=0.5)
        assert batt.soc <= 100.0

    def test_needs_charge_below_threshold(self):
        batt = BatteryModel(E_max=100.0)
        batt.soc = 15.0
        assert batt.needs_charge(0.2) is True

    def test_needs_charge_above_threshold(self):
        batt = BatteryModel(E_max=100.0)
        batt.soc = 80.0
        assert batt.needs_charge(0.2) is False

    def test_is_charging_flag(self):
        batt = BatteryModel()
        assert batt.is_charging is False
        batt.charging = True
        assert batt.is_charging is True

    def test_reset_to_health_clears_charging(self):
        batt = BatteryModel()
        batt.charging = True
        batt.soc = 10.0
        batt.reset_to_health(soh=0.9)
        assert batt.charging is False
        assert batt.soc == batt.E_max
        assert batt.soh == pytest.approx(0.9)

    def test_is_dead_fully_depleted(self):
        batt = BatteryModel(E_max=100.0)
        batt.discharge(100.0)
        assert batt.is_dead()

    def test_is_dead_healthy_battery(self):
        batt = BatteryModel(E_max=100.0)
        assert not batt.is_dead()
