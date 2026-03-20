#!/usr/bin/env python3
"""
Legacy evaluation script for realistic WSN metrics comparison.

⚠️  DEPRECATED: Use scripts/evaluate_baselines.py instead
    python scripts/evaluate_baselines.py --episodes 5

This module is maintained to support existing code that imports evaluate_realistic_metric().
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from typing import Dict, Optional, Type

from src.envs.wsn_env import WSNEnv
from src.agents.ddqn_agent import DDQNAgent
from src.baselines.baseline_policies import (
    RandomPolicy,
    GreedyPolicy,
    EnergyConservativePolicy,
    BalancedRotationPolicy,
)
from src.utils.visualization import save_metrics_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_realistic_metric(
    policy_class: Optional[Type] = None,
    model_path: Optional[str] = None,
    env: Optional[WSNEnv] = None,
    episodes: int = 5,
    name: str = "Method",
) -> Dict:
    """
    Evaluate using REALISTIC metrics for WSN.
    
    ⚠️  DEPRECATED: Use Trainer.evaluate() instead
    
    Metrics:
    - Service Time: Steps with >30% coverage maintained
    - Energy Efficiency: Service steps per joule
    - Reliability: Standard deviation of coverage
    
    Args:
        policy_class: Baseline policy class (or None for model_path)
        model_path: Path to trained DDQN model checkpoint
        env: WSNEnv instance
        episodes: Number of evaluation episodes
        name: Method name for reporting
        
    Returns:
        Dictionary with metrics
    """
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    
    if model_path:
        # Use trained DDQN model
        agent = DDQNAgent(state_dim, action_dim, env.N, lr=1e-4)
        agent.q_net.load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        policies = [agent] * episodes
    else:
        # Use baseline policy
        policy = policy_class(state_dim, action_dim, env.N)
        policies = [policy] * episodes
    
    service_times = []
    energies = []
    coverages = []
    
    for ep in range(episodes):
        state = env.reset()
        policy = policies[ep]
        
        episode_coverage = []
        episode_energy = 0
        
        done = False
        while not done:
            # Select action with eval_mode to disable epsilon-greedy
            action = (
                policy.select_action(state, eval_mode=True)
                if hasattr(policy, "select_action")
                else policy.select_action(state)
            )
            state, reward, done, info = env.step(action)
            
            cov = info.get("coverage_ratio", 0) * 100
            energy = info.get("total_energy_used", 0)
            
            episode_coverage.append(cov)
            episode_energy += energy
        
        # Service time = steps with >30% coverage
        service_time = sum(1 for c in episode_coverage if c > 30)
        service_times.append(service_time)
        energies.append(episode_energy)
        coverages.append(np.mean(episode_coverage))
    
    return {
        "name": name,
        "service_time": np.mean(service_times),
        "service_time_std": np.std(service_times),
        "total_energy": np.mean(energies),
        "avg_coverage": np.mean(coverages),
        "energy_efficiency": np.mean(service_times) / (np.mean(energies) + 0.1),
    }


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("REALISTIC WSN METRICS COMPARISON")
    print("=" * 80 + "\n")

    env = WSNEnv(N=550)

    best_model = "results/final_ddqn_latest.pth"
    results = []

    if os.path.exists(best_model):
        results.append(evaluate_realistic_metric(None, best_model, env, 5, "DDQN (Trained)"))

    results.extend([
        evaluate_realistic_metric(RandomPolicy, None, env, 5, "Random"),
        evaluate_realistic_metric(GreedyPolicy, None, env, 5, "Greedy"),
        evaluate_realistic_metric(EnergyConservativePolicy, None, env, 5, "Energy Conservative"),
        evaluate_realistic_metric(BalancedRotationPolicy, None, env, 5, "Balanced Rotation"),
    ])

    results_sorted = sorted(results, key=lambda x: x["energy_efficiency"], reverse=True)
    for idx, result in enumerate(results_sorted, start=1):
        print(f"{idx}. {result['name']:30s} | service={result['service_time']:6.1f} | coverage={result['avg_coverage']:6.1f}% | efficiency={result['energy_efficiency']:6.2f}")

    payload = {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in results}
    save_metrics_json(payload, "results/realistic_comparison.json")
    print("\nSaved: results/realistic_comparison.json")
