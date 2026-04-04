"""Metrics computation and logging utilities."""

from typing import Dict, List
import numpy as np


def compute_episode_metrics(
    episode_rewards: List[float],
    episode_infos: List[Dict],
) -> Dict[str, float]:
    """Compute metrics for an episode.
    
    Args:
        episode_rewards: List of step rewards
        episode_infos: List of step info dicts
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "total_reward": sum(episode_rewards),
        "mean_reward": np.mean(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "episode_length": len(episode_rewards),
    }
    
    if episode_infos:
        coverage_ratios = [info.get("coverage_ratio", 0) for info in episode_infos]
        soh_values = [info.get("avg_soh", 0) for info in episode_infos]
        dead_nodes = [info.get("dead_nodes", 0) for info in episode_infos]
        
        metrics.update({
            "mean_coverage": np.mean(coverage_ratios),
            "final_coverage": coverage_ratios[-1] if coverage_ratios else 0,
            "mean_soh": np.mean(soh_values),
            "final_soh": soh_values[-1] if soh_values else 0,
            "final_dead_nodes": dead_nodes[-1] if dead_nodes else 0,
        })
    
    return metrics
