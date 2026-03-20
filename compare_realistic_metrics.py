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
from typing import Dict, Type

from config.settings import get_config
from src.envs.wsn_env import WSNEnv
from src.agents.ddqn_agent import DDQNAgent
from src.baselines.baseline_policies import (
    RandomPolicy,
    GreedyPolicy,
    EnergyConservativePolicy,
    BalancedRotationPolicy,
)
from src.training.trainer import Trainer
from src.utils.visualization import save_metrics_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_realistic_metric(
    policy_class: Type = None,
    model_path: str = None,
    env: WSNEnv = None,
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
    
    # Save
    with open('results/realistic_comparison.json', 'w') as f:
        json.dump({r['name']: {k: v for k,v in r.items() if k != 'name'} for r in results}, f, indent=2)
    
    print(f"✓ Saved: results/realistic_comparison.json\n")
