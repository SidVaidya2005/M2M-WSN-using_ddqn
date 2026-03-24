"""Visualization utilities for plots and animations."""

from typing import List, Dict, Optional
import numpy as np
import json
from pathlib import Path


def save_metrics_json(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(output_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics_json(input_path: str) -> Dict:
    """Load metrics from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Metrics dictionary
    """
    with open(input_path, "r") as f:
        return json.load(f)


def plot_training_curve(
    episode_rewards: List[float],
    output_path: Optional[str] = None,
    window_size: int = 10,
) -> None:
    """Plot training curve (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.plot(episode_rewards, color="#fe964a", alpha=0.9, label="Episode Reward", linewidth=1.5)

    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        axes.plot(
            range(window_size - 1, len(episode_rewards)),
            moving_avg,
            color="#0077b6",
            label=f"{window_size}-Episode MA",
            linewidth=2.5,
        )

    axes.set_xlabel("Episode")
    axes.set_ylabel("Reward")
    axes.set_title("Training Progress")
    axes.legend()
    # Change background colour slightly
    axes.set_facecolor('#fdfdfd')
    axes.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_comparison(
    baselines: Dict[str, List[float]],
    agent_name: str = "DDQN",
    output_path: Optional[str] = None,
) -> None:
    """Plot comparison of different policies."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for policy_name, rewards in baselines.items():
        mean_reward = np.mean(rewards)
        ax.axhline(y=mean_reward, label=policy_name, linewidth=2)

    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"Policy Comparison: {agent_name} vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()
