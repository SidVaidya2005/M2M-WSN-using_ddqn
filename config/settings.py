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
        
        return cls(
            training=TrainingConfig(**data["training"]),
            environment=EnvironmentConfig(**data["environment"]),
            paths=PathConfig(**data["paths"]),
            visualization=VisualizationConfig(**data["visualization"]),
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Create Config from dictionary (useful for API requests)."""
        return cls(
            training=TrainingConfig(**data.get("training", {})),
            environment=EnvironmentConfig(**data.get("environment", {})),
            paths=PathConfig(**data.get("paths", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )
    
    def to_dict(self) -> Dict:
        """Convert Config to dictionary."""
        return {
            "training": self.training.__dict__,
            "environment": self.environment.__dict__,
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
