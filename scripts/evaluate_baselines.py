"""
Baseline evaluation script.

Compares DDQN against baseline policies to benchmark performance.

Usage:
    python scripts/evaluate_baselines.py --model results/trained_model.pth
"""

import argparse
from pathlib import Path
from typing import Dict

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.baselines import RandomPolicy, GreedyPolicy, EnergyConservativePolicy
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.metrics import compute_episode_metrics
from src.utils.visualization import save_metrics_json

logger = get_logger(__name__)


def evaluate_policy(policy, env, episodes=10, eval_mode=True):
    """Evaluate a policy."""
    rewards = []
    metrics = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_info = None
        done = False
        
        while not done:
            action = policy.select_action(state, eval_mode=eval_mode)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_info = info
            
            if hasattr(policy, 'store_transition'):
                policy.store_transition(state, action, reward, state, done)
        
        rewards.append(episode_reward)
        m = compute_episode_metrics([episode_reward], [episode_info])
        metrics.append(m)
    
    return rewards, metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate baselines")
    
    parser.add_argument(
        "--model",
        type=str,
        default="results/trained_model.pth",
        help="Path to trained DDQN model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per policy",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=550,
        help="Number of network nodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation script."""
    args = parse_args()
    
    config = get_config()
    config.paths.create_all()
    
    logger.info(f"Evaluating baselines with {args.episodes} episodes...")
    
    # Create environment
    env = WSNEnv(
        N=args.nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    
    # Initialize policies
    policies = {
        "Random": RandomPolicy(state_dim, action_dim, args.nodes),
        "Greedy": GreedyPolicy(state_dim, action_dim, args.nodes, awake_ratio=0.5),
        "EnergyConservative": EnergyConservativePolicy(
            state_dim, action_dim, args.nodes, awake_ratio=0.2
        ),
    }
    
    results = {}
    
    # Evaluate baselines
    for policy_name, policy in policies.items():
        logger.info(f"Evaluating {policy_name}...")
        rewards, metrics = evaluate_policy(policy, env, episodes=args.episodes)
        results[policy_name] = {
            "rewards": rewards,
            "mean_reward": sum(rewards) / len(rewards),
            "metrics": metrics,
        }
        logger.info(f"{policy_name}: mean reward = {results[policy_name]['mean_reward']:.2f}")
    
    # Load and evaluate DDQN
    logger.info("Evaluating DDQN...")
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_count=args.nodes,
    )
    
    model_path = Path(args.model)
    if model_path.exists():
        agent.load_model(str(model_path))
        trainer = Trainer(agent, env, logger_obj=logger)
        rewards, metrics = trainer.evaluate(episodes=args.episodes)
        results["DDQN"] = {
            "rewards": rewards,
            "mean_reward": sum(rewards) / len(rewards),
            "metrics": metrics,
        }
        logger.info(f"DDQN: mean reward = {results['DDQN']['mean_reward']:.2f}")
    else:
        logger.warning(f"Model not found at {model_path}, skipping DDQN evaluation")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_path = output_dir / "baseline_comparison.json"
    save_metrics_json(results, str(comparison_path))
    logger.info(f"Results saved to {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    for policy_name, result in results.items():
        print(f"{policy_name:20s}: {result['mean_reward']:8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
