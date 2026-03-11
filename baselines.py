# baselines.py
"""
Baseline policies for WSN scheduling to compare against DDQN
"""
import numpy as np
import random


class RandomPolicy:
    """
    Baseline: Random sleep/awake decisions for each node
    """
    def __init__(self, state_dim, action_dim, node_count):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_count = node_count
    
    def select_action(self, state, eval_mode=False):
        """
        Returns random action for each node (0: sleep, 1: awake)
        """
        return np.random.randint(0, self.action_dim, size=self.node_count)
    
    def store(self, state, action, reward, next_state, done):
        """No-op for random policy"""
        pass
    
    def train_step(self):
        """No-op for random policy"""
        return None


class GreedyPolicy:
    """
    Baseline: Greedy heuristic - wake nodes with high SoC/SoH
    Keep ~50% of nodes awake, prioritize healthy, high-energy nodes
    """
    def __init__(self, state_dim, action_dim, node_count, awake_ratio=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_count = node_count
        self.awake_ratio = awake_ratio
    
    def select_action(self, state, eval_mode=False):
        """
        Greedy: wake nodes with highest (SoC * SoH) score
        """
        # State format: [soc_norm, soh, last_action, dist_norm, recent_activity] * N
        obs_dim_per_node = 5
        
        scores = []
        for i in range(self.node_count):
            idx = i * obs_dim_per_node
            soc_norm = state[idx]
            soh = state[idx + 1]
            dist_norm = state[idx + 3]
            
            # Score: prioritize high SoC/SoH, penalize distance
            score = soc_norm * soh * (1.0 - 0.3 * dist_norm)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Wake top nodes by score
        num_awake = max(1, int(self.node_count * self.awake_ratio))
        top_indices = np.argsort(scores)[-num_awake:]
        
        action = np.zeros(self.node_count, dtype=int)
        action[top_indices] = 1
        
        return action
    
    def store(self, state, action, reward, next_state, done):
        """No-op for greedy policy"""
        pass
    
    def train_step(self):
        """No-op for greedy policy"""
        return None


class EnergyConservativePolicy:
    """
    Baseline: Conservative energy policy - minimize awake nodes
    Only wake when necessary for coverage, keep most nodes asleep
    """
    def __init__(self, state_dim, action_dim, node_count, awake_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_count = node_count
        self.awake_ratio = awake_ratio
    
    def select_action(self, state, eval_mode=False):
        """
        Conservative: only wake healthiest nodes, minimize energy usage
        """
        obs_dim_per_node = 5
        
        health_scores = []
        for i in range(self.node_count):
            idx = i * obs_dim_per_node
            soc_norm = state[idx]
            soh = state[idx + 1]
            
            # Only score by health (SoC * SoH), ignore distance
            health_score = soc_norm * soh
            health_scores.append(health_score)
        
        health_scores = np.array(health_scores)
        
        # Wake only the healthiest nodes
        num_awake = max(1, int(self.node_count * self.awake_ratio))
        top_indices = np.argsort(health_scores)[-num_awake:]
        
        action = np.zeros(self.node_count, dtype=int)
        action[top_indices] = 1
        
        return action
    
    def store(self, state, action, reward, next_state, done):
        """No-op"""
        pass
    
    def train_step(self):
        """No-op"""
        return None


class BalancedPolicy:
    """
    Baseline: Balanced approach - rotate which nodes are awake
    Distribute energy burden fairly across all nodes
    """
    def __init__(self, state_dim, action_dim, node_count, awake_ratio=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_count = node_count
        self.awake_ratio = awake_ratio
        self.step_count = 0
    
    def select_action(self, state, eval_mode=False):
        """
        Balanced: rotate which nodes are awake to distribute load
        """
        num_awake = max(1, int(self.node_count * self.awake_ratio))
        
        # Circular rotation based on step count
        offset = self.step_count // 10  # Change every 10 steps
        start_idx = (offset % self.node_count)
        
        action = np.zeros(self.node_count, dtype=int)
        indices = np.arange(start_idx, start_idx + num_awake) % self.node_count
        action[indices] = 1
        
        self.step_count += 1
        return action
    
    def store(self, state, action, reward, next_state, done):
        """No-op"""
        pass
    
    def train_step(self):
        """No-op"""
        return None
