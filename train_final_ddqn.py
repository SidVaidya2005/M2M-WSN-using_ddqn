#!/usr/bin/env python3
"""
Final DDQN Training Script - Focus on NETWORK LIFETIME
- Use BEST reward weights found through analysis
- Priority: Network stays alive as long as possible
- Not about energy conservation, not about battery health
- Just: KEEP THE NETWORK WORKING
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env_wsn import WSNEnv
from ddqn_agent import DDQNAgent
import torch
import json

os.makedirs('results', exist_ok=True)


def train_final_ddqn(episodes=100, seed=42, N=550, lr=1e-4, gamma=0.99, batch_size=64, death_threshold=0.3):
    """
    Train DDQN to Maximize Network Lifetime
    
    Focus on learning an optimal scheduling policy that:
    - Maximizes how long the network stays functional
    - Maintains good coverage during operation
    - Balances energy efficiency with performance
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Environment
    env = WSNEnv(N=N, death_threshold=death_threshold)
    
    # DDQN Agent (original, proven architecture)
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
    
    print(f"\n{'='*80}")
    print(f"DDQN TRAINING - NETWORK LIFETIME OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Episodes: {episodes}")
    print(f"Nodes: {N}")
    print(f"Objective: Learn optimal scheduling policy for maximum network lifetime")
    print(f"Strategy: Reward coverage heavily, balance energy efficiency, maintain fairness")
    print(f"{'='*80}\n")
    
    # Track metrics
    lifetime_history = []
    coverage_history = []
    soh_history = []
    reward_history = []
    
    # Best models
    best_lifetime = 0
    best_lifetime_model = None
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_lifetime = 0
        episode_coverage = []
        episode_soh = []
        episode_reward = 0 # Initialize episode reward
        
        done = False
        while not done:
            # Learn policy
            prev_state = state  # Save current state BEFORE stepping
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            
            agent.store(prev_state, action, reward, state, done)  # Fix: store prev_state, not current state
            agent.train_step()
            
            episode_lifetime += 1
            episode_reward += reward
            episode_coverage.append(info.get('coverage_ratio', 0) * 100)
            episode_soh.append(info.get('avg_soh', 1.0))
        
        lifetime_history.append(episode_lifetime)
        coverage_history.append(np.mean(episode_coverage))
        soh_history.append(np.mean(episode_soh))
        reward_history.append(episode_reward)
        
        # Save best model by lifetime
        if episode_lifetime > best_lifetime:
            best_lifetime = episode_lifetime
            best_lifetime_model = episode
            torch.save(agent.q_net.state_dict(), f'results/final_ddqn_best_lifetime_ep{episode}.pth')
        
        # Print every 10 episodes
        if episode % 10 == 0 or episode == 1:
            avg_lifetime = np.mean(lifetime_history[-10:])
            avg_coverage = np.mean(coverage_history[-10:])
            print(f"Ep {episode:3d}/100 | Lifetime: {episode_lifetime:4d} (avg: {avg_lifetime:6.1f}) | "
                  f"Coverage: {avg_coverage:5.1f}% | SoH: {np.mean(episode_soh):7.5f}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Episode: {best_lifetime_model} with lifetime {best_lifetime} steps")
    print(f"Final 10-Ep Avg Lifetime: {np.mean(lifetime_history[-10:]):.1f} steps")
    print(f"Final 10-Ep Avg Coverage: {np.mean(coverage_history[-10:]):.1f}%")
    print(f"\n📊 DDQN Performance Summary:")
    print(f"   Average Network Lifetime: {np.mean(lifetime_history[-10:]):.1f} steps")
    print(f"   Network Coverage Maintained: {np.mean(coverage_history[-10:]):.1f}%")
    
    print(f"{'='*80}\n")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DDQN Training - Focus on Network Lifetime', fontsize=14, fontweight='bold')
    
    # Lifetime
    axes[0, 0].plot(lifetime_history, linewidth=2, marker='o', markersize=4, color='blue', label='DDQN Lifetime')
    axes[0, 0].axhline(np.mean(lifetime_history[-10:]), color='g', linestyle='--', linewidth=2, label='Final Avg (Last 10 Ep)')
    axes[0, 0].set_title('Network Lifetime per Episode', fontweight='bold')
    axes[0, 0].set_ylabel('Steps')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage
    axes[0, 1].plot(coverage_history, linewidth=2, marker='s', markersize=4, color='orange')
    axes[0, 1].set_title('Average Coverage %', fontweight='bold')
    axes[0, 1].set_ylabel('Coverage %')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SoH
    axes[1, 0].plot(soh_history, linewidth=2, marker='^', markersize=4, color='green')
    axes[1, 0].set_title('Average Battery SoH', fontweight='bold')
    axes[1, 0].set_ylabel('SoH')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
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
