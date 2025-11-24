"""
Environment wrapper and utilities for SimpleTag competition.

This module provides a clean interface to the PettingZoo SimpleTag environment
with competition-specific configurations.
"""

import numpy as np
from pettingzoo.mpe import simple_tag_v3
from typing import Dict, List, Any, Optional
import json


class SimpleTagEnv:
    """
    Wrapper for PettingZoo's SimpleTag environment with competition settings.
    
    Features:
    - Fixed seeds for reproducible evaluation
    - Episode statistics tracking
    - Simplified interface for agents
    """
    
    def __init__(
        self,
        num_good: int = 1,
        num_adversaries: int = 3,
        num_obstacles: int = 2,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize SimpleTag environment.
        
        Args:
            num_good: Number of good agents (prey) - default 1
            num_adversaries: Number of adversary agents (predators) - default 3
            num_obstacles: Number of obstacles in the environment
            max_cycles: Maximum number of cycles per episode
            continuous_actions: Whether to use continuous actions (False for competition)
            render_mode: 'human' for visualization, None for headless
        """
        self.env = simple_tag_v3.parallel_env(
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.max_cycles = max_cycles
        
        # Agent names
        self.good_agents = [f"agent_{i}" for i in range(num_good)]
        self.adversary_agents = [f"adversary_{i}" for i in range(num_adversaries)]
        self.all_agents = self.adversary_agents + self.good_agents
        
        # Episode statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset episode statistics."""
        self.episode_rewards = {agent: 0.0 for agent in self.all_agents}
        self.episode_length = 0
        self.catches = 0
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observations: Dictionary mapping agent names to observations
        """
        if seed is not None:
            observations, infos = self.env.reset(seed=seed)
        else:
            observations, infos = self.env.reset()
        
        self.reset_stats()
        return observations
    
    def step(self, actions: Dict[str, int]) -> tuple:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary mapping agent names to actions
            
        Returns:
            observations: Dictionary of observations
            rewards: Dictionary of rewards
            terminations: Dictionary of termination flags
            truncations: Dictionary of truncation flags
            infos: Dictionary of info
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Update statistics
        self.episode_length += 1
        for agent, reward in rewards.items():
            self.episode_rewards[agent] += reward
        
        # Check for catches (good agent got negative reward)
        for agent in self.good_agents:
            if agent in rewards and rewards[agent] < -5:  # Catch penalty
                self.catches += 1
        
        return observations, rewards, terminations, truncations, infos
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dictionary containing episode statistics
        """
        return {
            'episode_length': self.episode_length,
            'total_rewards': self.episode_rewards.copy(),
            'avg_reward': {
                'good': np.mean([self.episode_rewards[a] for a in self.good_agents]),
                'adversary': np.mean([self.episode_rewards[a] for a in self.adversary_agents]),
            },
            'catches': self.catches,
        }
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def observation_space(self, agent: str):
        """Get observation space for a specific agent."""
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        """Get action space for a specific agent."""
        return self.env.action_space(agent)


def evaluate_agents(
    env: SimpleTagEnv,
    good_agents: List[Any],
    adversary_agents: List[Any],
    num_episodes: int = 100,
    seeds: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate agents over multiple episodes.
    
    Args:
        env: SimpleTagEnv instance
        good_agents: List of good agent instances
        adversary_agents: List of adversary agent instances
        num_episodes: Number of episodes to evaluate
        seeds: List of seeds for reproducible evaluation
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation results
    """
    if seeds is None:
        seeds = list(range(num_episodes))
    
    assert len(seeds) >= num_episodes, "Not enough seeds provided"
    
    all_episode_stats = []
    
    for episode_idx in range(num_episodes):
        seed = seeds[episode_idx]
        
        # Set numpy random seed for reproducibility (affects agent behavior)
        np.random.seed(seed)
        
        observations = env.reset(seed=seed)
        
        done = False
        step_count = 0
        
        while not done:
            actions = {}
            
            # Get actions from adversary agents
            for i, agent_name in enumerate(env.adversary_agents):
                if agent_name in observations:
                    actions[agent_name] = adversary_agents[i].act(
                        observations[agent_name], deterministic=True
                    )
            
            # Get actions from good agents
            for i, agent_name in enumerate(env.good_agents):
                if agent_name in observations:
                    actions[agent_name] = good_agents[i].act(
                        observations[agent_name], deterministic=True
                    )
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            step_count += 1
            
            if step_count >= env.max_cycles:
                done = True
        
        # Collect episode statistics
        stats = env.get_episode_stats()
        all_episode_stats.append(stats)
        
        if verbose and (episode_idx + 1) % 10 == 0:
            print(f"Episode {episode_idx + 1}/{num_episodes} completed")
    
    # Aggregate statistics
    results = {
        'num_episodes': num_episodes,
        'avg_episode_length': np.mean([s['episode_length'] for s in all_episode_stats]),
        'avg_good_reward': np.mean([s['avg_reward']['good'] for s in all_episode_stats]),
        'avg_adversary_reward': np.mean([s['avg_reward']['adversary'] for s in all_episode_stats]),
        'total_catches': sum([s['catches'] for s in all_episode_stats]),
        'avg_catches_per_episode': np.mean([s['catches'] for s in all_episode_stats]),
        'all_episode_stats': all_episode_stats,
    }
    
    return results


def visualize_episode(
    env: SimpleTagEnv,
    good_agents: List[Any],
    adversary_agents: List[Any],
    seed: Optional[int] = None,
):
    """
    Visualize a single episode with rendering.
    
    Args:
        env: SimpleTagEnv instance with render_mode='human'
        good_agents: List of good agent instances
        adversary_agents: List of adversary agent instances
        seed: Random seed
    """
    observations = env.reset(seed=seed)
    
    done = False
    step_count = 0
    
    print(f"Starting episode visualization (seed={seed})...")
    
    while not done:
        actions = {}
        
        # Get actions from adversary agents
        for i, agent_name in enumerate(env.adversary_agents):
            if agent_name in observations:
                actions[agent_name] = adversary_agents[i].act(observations[agent_name])
        
        # Get actions from good agents
        for i, agent_name in enumerate(env.good_agents):
            if agent_name in observations:
                actions[agent_name] = good_agents[i].act(observations[agent_name])
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check if episode is done
        done = all(terminations.values()) or all(truncations.values())
        step_count += 1
        
        if step_count >= env.max_cycles:
            done = True
    
    stats = env.get_episode_stats()
    print("\nEpisode finished:")
    print(f"  Length: {stats['episode_length']} steps")
    print(f"  Good agent avg reward: {stats['avg_reward']['good']:.2f}")
    print(f"  Adversary avg reward: {stats['avg_reward']['adversary']:.2f}")
    print(f"  Total catches: {stats['catches']}")
    

def load_evaluation_seeds(seed_file: str = "seeds.json") -> List[int]:
    """
    Load evaluation seeds from file.
    """
    import json
    with open(seed_file, 'r') as f:
        seeds = json.load(f)
    return seeds
