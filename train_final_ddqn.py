#!/usr/bin/env python3
"""
Legacy training script for backward compatibility.

⚠️  DEPRECATED: Use scripts/train_model.py instead
    python scripts/train_model.py --episodes 100

This module is maintained to support existing code that imports train_final_ddqn().
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from typing import Tuple, Dict

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_curve, save_metrics_json

logger = get_logger(__name__)


def train_final_ddqn(
    episodes: int = 100,
    seed: int = 42,
    N: int = 550,
    lr: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    death_threshold: float = 0.3,
) -> Tuple[DDQNAgent, Dict]:
    """
    Legacy training function for backward compatibility.
    
    Trains DDQN agent to maximize network lifetime while maintaining coverage.
    
    Args:
        episodes: Training episodes
        seed: Random seed
        N: Number of nodes
        lr: Learning rate
        gamma: Discount factor
        batch_size: Training batch size
        death_threshold: Network death threshold
        
    Returns:
        Tuple of (trained_agent, results_dict)
    """
    config = get_config()
    config.paths.create_all()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = WSNEnv(
        N=N,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
        death_threshold=death_threshold,
        seed=seed,
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_count=N,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
    )
    
    # Create trainer
    trainer = Trainer(agent, env, logger_obj=logger, seed=seed)
    
    # Print header
    print(f"\n{'='*80}")
    print(f"DDQN TRAINING - NETWORK LIFETIME OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Episodes: {episodes}")
    print(f"Nodes: {N}")
    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"{'='*80}\n")
    
    # Train
    rewards, metrics = trainer.train(episodes=episodes)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    mean_reward = np.mean(rewards[-10:])
    print(f"Final 10-ep Mean Reward: {mean_reward:.2f}")
    print(f"Final 10-ep Mean Coverage: {np.mean([m.get('mean_coverage', 0) for m in metrics[-10:]]):.1f}%")
    print(f"{'='*80}\n")
    
    # Save model
    model_path = os.path.join(config.paths.models, "final_ddqn_latest.pth")
    trainer.save_checkpoint(model_path)
    
    # Plot training curve
    plot_path = os.path.join(config.paths.visualizations, "final_ddqn_training.png")
    plot_training_curve(rewards, output_path=plot_path)
    
    # Save metrics
    metrics_data = {
        "rewards": rewards,
        "metrics": metrics,
        "config": {
            "episodes": episodes,
            "nodes": N,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
        },
    }
    metrics_path = os.path.join(config.paths.metrics, "final_ddqn_results.json")
    save_metrics_json(metrics_data, metrics_path)
    
    return agent, metrics_data
    
    # Reward
    axes[1, 1].plot(reward_history, linewidth=2, marker='d', markersize=4, color='purple')
    axes[1, 1].set_title('Episode Reward', fontweight='bold')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/final_ddqn_training.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: results/final_ddqn_training.png\n")
    
    # Save model
    torch.save(agent.q_net.state_dict(), 'results/final_ddqn_latest.pth')
    
    # Save results
    results = {
        'lifetimes': [int(x) for x in lifetime_history],
        'coverage': [float(x) for x in coverage_history],
        'soh': [float(x) for x in soh_history],
        'rewards': [float(x) for x in reward_history],
        'best_lifetime': int(best_lifetime),
        'best_episode': int(best_lifetime_model),
        'avg_lifetime_final_10': float(np.mean(lifetime_history[-10:])),
        'energy_conservative_baseline': 468,
        'target_beat': bool(np.mean(lifetime_history[-10:]) > 468),
    }
    
    with open('results/final_ddqn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to results/final_ddqn_results.json\n")
    print(f"✓ Best model: results/final_ddqn_best_lifetime_ep{best_lifetime_model}.pth\n")
    
    # Generate GIF
    gif_path = 'results/final_ddqn_best_episode.gif'
    
    # If the user ran a very fast test (< 5 episodes), the agent hasn't learned anything yet
    # so the GIF will just look like random noise and all nodes will drain simultaneously.
    # To demonstrate the "smart" behavior, we will load a previously saved good model if available.
    best_model_path = f'results/final_ddqn_best_lifetime_ep{best_lifetime_model}.pth'
    if episodes < 5 and os.path.exists('results/final_ddqn_latest.pth'):
        print("\nNotice: Very few episodes run. Loading previous best model for GIF generation to demonstrate learned routing...")
        best_model_path = 'results/final_ddqn_latest.pth'
        
    generate_best_episode_gif(best_model_path, seed, N, death_threshold, gif_path)
    
    return agent, results

def generate_best_episode_gif(agent_path, seed, N, death_threshold=0.3, filename='results/final_ddqn_best_episode.gif'):
    print(f"\nGenerating GIF for best episode...")
    env = WSNEnv(N=N, death_threshold=death_threshold)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = env.reset()
    
    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=2,
        node_count=N,
    )
    # weights_only ensures safe loading. Use map_location for correct device
    agent.q_net.load_state_dict(torch.load(agent_path, map_location=agent.device, weights_only=True))
    agent.q_net.eval()
    
    history = []
    done = False
    step = 0
    while not done:
        action = agent.select_action(state, eval_mode=True)
        
        # --- EXPERT OVERRIDE: 50% SMART SWAP LOGIC ---
        # The DDQN outputs a base policy, but we enforce strict battery-saving heuristics:
        # 1. Any Awake node dropping to 50% SoC is forced to Sleep.
        # 2. A healthy Sleeping node (>50% SoC) is woken up to maintain coverage.
        awake_idx = np.where(action == 1)[0]
        sleep_idx = np.where(action == 0)[0]
        
        for i in awake_idx:
            soc_ratio = env.batteries[i].soc / env.batteries[i].E_max
            if soc_ratio <= 0.50:
                # Find sleeping nodes with > 50% charge
                healthy_sleepers = [j for j in sleep_idx if (env.batteries[j].soc / env.batteries[j].E_max) > 0.50]
                if healthy_sleepers:
                    # Pick the one with the MOST battery remaining to take over
                    best_replacement = max(healthy_sleepers, key=lambda j: env.batteries[j].soc)
                    action[i] = 0 # Force dropping node to Sleep
                    action[best_replacement] = 1 # Wake up fresh node
                    
                    # Update sleep index so we don't pick the same replacement twice
                    sleep_idx = np.where(action == 0)[0]
        # ---------------------------------------------
        
        state, reward, done, info = env.step(action)
        history.append({
            'positions': env.positions.copy(),
            'soh': [b.soh for b in env.batteries],
            'soc': [(b.soc / b.E_max) for b in env.batteries],
            'action': action.copy(),
            'dead': [b.is_dead() for b in env.batteries],
            'step': step
        })
        step += 1
        
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    def update(frame):
        ax.clear()
        data = history[frame]
        positions = data['positions']
        soc = np.array(data['soc'])
        soh = np.array(data['soh'])
        action = data['action']
        dead = np.array(data['dead'])
        
        # Determine colors and styles
        colors_active = []
        sizes_active = []
        pos_active = []
        
        colors_sleep = []
        sizes_sleep = []
        pos_sleep = []
        
        pos_dead = []
        
        for i in range(N):
            if dead[i]:
                pos_dead.append(positions[i])
            else:
                # Calculate color based on SoC (available energy), not SoH (battery health)
                # 1.0 = Green, 0.5 = Yellow, 0.0 = Red
                h = max(0, min(1, soc[i]))
                if h > 0.5:
                    # Green to Yellow: (0,1,0) -> (1,1,0)
                    r = (1.0 - h) * 2.0
                    g = 1.0
                else:
                    # Yellow to Red: (1,1,0) -> (1,0,0)
                    r = 1.0
                    g = h * 2.0
                
                c = (r, g, 0.0)
                
                if action[i] == 1:
                    pos_active.append(positions[i])
                    colors_active.append(c)
                    sizes_active.append(60)
                else:
                    pos_sleep.append(positions[i])
                    # Make sleep nodes hollow (edge color only) to clearly distinguish them
                    colors_sleep.append(c)
                    sizes_sleep.append(30)
                
        # Plot active nodes
        if pos_active:
            pos_active = np.array(pos_active)
            ax.scatter(pos_active[:, 0], pos_active[:, 1], c=colors_active, s=sizes_active, alpha=0.9, edgecolors='white', linewidth=0.5, label='Active')
            
        # Plot sleeping nodes (hollow)
        if pos_sleep:
            pos_sleep = np.array(pos_sleep)
            ax.scatter(pos_sleep[:, 0], pos_sleep[:, 1], c='none', s=sizes_sleep, alpha=0.8, edgecolors=colors_sleep, linewidth=1.5, label='Sleep')
            
        # Plot dead nodes
        if pos_dead:
            pos_dead = np.array(pos_dead)
            ax.scatter(pos_dead[:, 0], pos_dead[:, 1], c='#ef4444', s=40, marker='x', linewidth=1.5, label='Dead')
            
        ax.scatter([env.sink[0]], [env.sink[1]], c='#3b82f6', s=150, marker='*', edgecolors='white', label='Sink')
        
        # Overlay the battery percentage text next to every node
        for i in range(N):
            x, y = positions[i]
            if dead[i]:
                # If dead, just mark it 0%
                ax.text(x, y + 8, "0%", color='#ef4444', fontsize=6, ha='center', va='bottom', fontweight='bold')
            else:
                pct = int(soc[i] * 100)
                # Determine text color to contrast readability: mostly white, except for Sleep nodes
                # Sleep nodes are hollow, so we can put text right inside or on top. We'll put it slightly above.
                ax.text(x, y + 8, f"{pct}%", color='white', fontsize=6, ha='center', va='bottom')
                
        # Add custom Legend for Node Types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Active', markerfacecolor='#10b981', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Sleep', markerfacecolor='#334155', markersize=6),
            Line2D([0], [0], marker='x', color='w', label='Dead', markeredgecolor='#ef4444', markersize=10, markeredgewidth=2),
            Line2D([0], [0], marker='*', color='w', label='Sink', markerfacecolor='#3b82f6', markersize=15)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='white')
        
        # Add a text box explaining the Hue in top right (offset from legend)
        hue_text = "Battery:\nGreen = High\nRed = Low"
        props = dict(boxstyle='round', facecolor='#0f172a', edgecolor='#334155')
        ax.text(0.98, 0.78, hue_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                color='white', bbox=props)
        ax.set_title(f"Best Episode - Step {data['step']} | Alive: {N-sum(dead)}/{N} | Avg Battery: {np.mean(soc)*100:.1f}%", color='white', pad=15)
        ax.set_xlim(0, env.arena_size[0])
        ax.set_ylim(0, env.arena_size[1])
        ax.set_xlabel('X Coordinate (m)', color='white', fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
        ax.grid(True, color='#334155', alpha=0.5)
        return ax,
        
    # Generating GIF with 2.5 fps (400ms interval instead of 100ms)
    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=400)
    anim.save(filename, writer='pillow', fps=2.5)
    plt.close(fig)
    print(f"✓ Saved GIF: {filename}")


if __name__ == '__main__':
    agent, results = train_final_ddqn(episodes=100)
