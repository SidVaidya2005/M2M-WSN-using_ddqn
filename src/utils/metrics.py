"""Metrics computation and logging utilities."""

from typing import Dict, List, Tuple
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


def aggregate_metrics(
    episodes_metrics: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across multiple episodes.
    
    Args:
        episodes_metrics: List of episode metric dicts
        
    Returns:
        Dictionary of (mean, std) for each metric
    """
    if not episodes_metrics:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for metrics in episodes_metrics:
        metric_names.update(metrics.keys())
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m.get(metric_name, 0) for m in episodes_metrics]
        aggregated[metric_name] = (np.mean(values), np.std(values))
    
    return aggregated


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary to readable string.
    
    Args:
        metrics: Metrics dictionary
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.{precision}f}")
        else:
            lines.append(f"{key}: {value}")
    return ", ".join(lines)


def compute_lifetime_metrics(
    dead_nodes_per_step: List[int],
    total_nodes: int,
) -> Dict[str, float]:
    """Compute network lifetime metrics.
    
    Args:
        dead_nodes_per_step: Number of dead nodes at each step
        total_nodes: Total number of nodes
        
    Returns:
        Lifetime metrics
    """
    dead_threshold = 0.3  # Network dead if > 30% nodes dead
    dead_nodes_threshold = int(total_nodes * dead_threshold)
    
    network_dead_step = None
    for step, dead_count in enumerate(dead_nodes_per_step):
        if dead_count > dead_nodes_threshold:
            network_dead_step = step
            break
    
    return {
        "network_lifetime": network_dead_step if network_dead_step else len(dead_nodes_per_step),
        "nodes_dead_at_end": dead_nodes_per_step[-1] if dead_nodes_per_step else 0,
        "total_nodes": total_nodes,
    }
