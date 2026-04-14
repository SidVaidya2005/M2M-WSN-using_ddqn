"""
Configuration management for WSN DDQN Training Platform.

Loads YAML config, provides validation, and centralizes all settings.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    episodes: int
    batch_size: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: int
    target_update_frequency: int
    replay_buffer_size: int
    min_replay_size: int


@dataclass
class RewardWeightsConfig:
    """Reward function weights."""
    coverage: float
    energy: float
    soh: float
    balance: float


@dataclass
class ChargingConfig:
    """Charging model parameters."""
    enabled: bool
    rate: float       # SoC fraction recovered per step while charging
    threshold: float  # SoC below which a node enters charging state


@dataclass
class WakeCooperationConfig:
    """Cooperative wake-up rule parameters."""
    low_battery_soc: float  # SoC at which env forces a neighbour awake


@dataclass
class EnvironmentConfig:
    """Environment parameters."""
    num_nodes: int
    arena_size: List[int]
    sink_position: List[int]
    max_steps: int
    timestep_energy_awake: float
    energy_sleep: float
    death_threshold: float
    seed: int
    sensing_radius: float
    reward_weights: RewardWeightsConfig
    charging: ChargingConfig
    wake_cooperation: WakeCooperationConfig


@dataclass
class PathConfig:
    """Directory paths for outputs."""
    models: str
    metrics: str
    visualizations: str
    logs: str

    def create_all(self):
        """Create all directories if they don't exist."""
        for path in [self.models, self.metrics, self.visualizations, self.logs]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    save_plots: bool
    plot_dpi: int
    animation_interval: int


@dataclass
class Config:
    """Master configuration class."""
    training: TrainingConfig
    environment: EnvironmentConfig
    paths: PathConfig
    visualization: VisualizationConfig

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Config object with all settings

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config validation fails
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        env_data = data["environment"]
        environment = EnvironmentConfig(
            num_nodes=env_data["num_nodes"],
            arena_size=env_data["arena_size"],
            sink_position=env_data["sink_position"],
            max_steps=env_data["max_steps"],
            timestep_energy_awake=env_data["timestep_energy_awake"],
            energy_sleep=env_data["energy_sleep"],
            death_threshold=env_data["death_threshold"],
            seed=env_data["seed"],
            sensing_radius=env_data["sensing_radius"],
            reward_weights=RewardWeightsConfig(**env_data["reward_weights"]),
            charging=ChargingConfig(**env_data["charging"]),
            wake_cooperation=WakeCooperationConfig(**env_data["wake_cooperation"]),
        )

        return cls(
            training=TrainingConfig(**data["training"]),
            environment=environment,
            paths=PathConfig(**data["paths"]),
            visualization=VisualizationConfig(**data["visualization"]),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Create Config from dictionary (useful for API requests)."""
        env_data = data.get("environment", {})
        environment = EnvironmentConfig(
            num_nodes=env_data.get("num_nodes", 50),
            arena_size=env_data.get("arena_size", [500, 500]),
            sink_position=env_data.get("sink_position", [250, 250]),
            max_steps=env_data.get("max_steps", 1000),
            timestep_energy_awake=env_data.get("timestep_energy_awake", 1.0),
            energy_sleep=env_data.get("energy_sleep", 0.01),
            death_threshold=env_data.get("death_threshold", 0.3),
            seed=env_data.get("seed", 42),
            sensing_radius=env_data.get("sensing_radius", 100.0),
            reward_weights=RewardWeightsConfig(
                **env_data.get("reward_weights", {"coverage": 10.0, "energy": 5.0, "soh": 1.0, "balance": 2.0})
            ),
            charging=ChargingConfig(
                **env_data.get("charging", {"enabled": True, "rate": 0.05, "threshold": 0.2})
            ),
            wake_cooperation=WakeCooperationConfig(
                **env_data.get("wake_cooperation", {"low_battery_soc": 0.5})
            ),
        )
        return cls(
            training=TrainingConfig(**data.get("training", {})),
            environment=environment,
            paths=PathConfig(**data.get("paths", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )

    def to_dict(self) -> Dict:
        """Convert Config to dictionary."""
        env = self.environment
        return {
            "training": self.training.__dict__,
            "environment": {
                "num_nodes": env.num_nodes,
                "arena_size": env.arena_size,
                "sink_position": env.sink_position,
                "max_steps": env.max_steps,
                "timestep_energy_awake": env.timestep_energy_awake,
                "energy_sleep": env.energy_sleep,
                "death_threshold": env.death_threshold,
                "seed": env.seed,
                "sensing_radius": env.sensing_radius,
                "reward_weights": env.reward_weights.__dict__,
                "charging": env.charging.__dict__,
                "wake_cooperation": env.wake_cooperation.__dict__,
            },
            "paths": self.paths.__dict__,
            "visualization": self.visualization.__dict__,
        }

    def validate(self) -> bool:
        """Validate configuration values.

        Returns:
            True if valid, raises ValueError if invalid
        """
        if self.training.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.training.batch_size < 8:
            raise ValueError("batch_size must be >= 8")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.training.gamma < 0 or self.training.gamma > 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.environment.num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        return True


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """Get the global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = Config.load(config_path)
        _config.validate()
        _config.paths.create_all()
    return _config


def reload_config(config_path: str = "config/config.yaml") -> Config:
    """Reload configuration from file (useful for testing)."""
    global _config
    _config = Config.load(config_path)
    _config.validate()
    return _config
