"""
Agent Template for SimpleTag Competition

Students must implement this PPOAgent class with their own PPO implementation.
This file serves as the interface between your trained agent and the tournament system.
"""

import numpy as np
from typing import Any, Dict, Optional


class PPOAgent:
    """
    PPO Agent template for SimpleTag competition.
    
    Students must implement this class with their PPO algorithm.
    The tournament system will instantiate this class and call the act() method.
    """
    
    def __init__(self, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
        """
        Initialize your PPO agent.
        
        Args:
            observation_space: The observation space from the environment
                              (typically Box with continuous values)
            action_space: The action space from the environment
                         (Discrete(5) for SimpleTag: [no-op, up, down, left, right])
            agent_id: String identifier for the agent (e.g., 'adversary_0', 'agent_0')
        
        Example:
            self.observation_dim = observation_space.shape[0]
            self.action_dim = action_space.n
            self.policy_network = self.build_policy_network()
            self.load_weights('model_weights.pth')
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = agent_id
        
        # TODO: Initialize your PPO networks (actor, critic)
        # TODO: Load your trained model weights
        
        raise NotImplementedError("Students must implement __init__ method")
    
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select an action given an observation.
        
        This method will be called by the tournament system during evaluation.
        
        Args:
            observation: numpy array of shape (observation_dim,) representing the current state
            deterministic: If True, select the most probable action (argmax).
                          If False, sample from the policy distribution.
                          Default is True for evaluation.
        
        Returns:
            action: Integer from 0 to 4 representing the discrete action
                   0 = no-op
                   1 = move up
                   2 = move down  
                   3 = move left
                   4 = move right
        
        Example Implementation:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            # Get action probabilities from policy network
            with torch.no_grad():
                action_logits = self.policy_network(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
            
            # Select action
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()
            
            return action
        """
        # TODO: Implement action selection using your trained policy
        
        raise NotImplementedError("Students must implement act() method")
    
    def load_weights(self, weights_path: str):
        """
        Load trained model weights.
        
        Args:
            weights_path: Path to the model weights file (e.g., 'model_weights.pth')
        
        Example:
            checkpoint = torch.load(weights_path, map_location='cpu')
            self.policy_network.load_state_dict(checkpoint['policy'])
            self.value_network.load_state_dict(checkpoint['value'])
        """
        # TODO: Implement weight loading
        
        raise NotImplementedError("Students must implement load_weights() method")


class PPOTrainer:
    """
    Optional: PPO Training class template.
    
    Students can implement their training loop here for local development.
    This class is NOT used during tournament evaluation.
    """
    
    def __init__(self, env, agent: PPOAgent, config: Dict[str, Any]):
        """
        Initialize PPO trainer.
        
        Args:
            env: The PettingZoo environment
            agent: PPOAgent instance to train
            config: Dictionary containing hyperparameters
                   e.g., {'lr': 3e-4, 'gamma': 0.99, 'epsilon': 0.2, ...}
        """
        self.env = env
        self.agent = agent
        self.config = config
        
        # TODO: Initialize optimizer, buffers, etc.
    
    def train(self, num_episodes: int):
        """
        Main training loop for PPO.
        
        Args:
            num_episodes: Number of episodes to train
        """
        # TODO: Implement PPO training loop
        # - Collect trajectories
        # - Compute advantages using GAE
        # - Update policy using clipped surrogate objective
        # - Update value function
        # - Log metrics
        
        raise NotImplementedError("Students must implement training loop")
    
    def collect_trajectories(self, num_steps: int):
        """
        Collect trajectories by running the policy in the environment.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary containing observations, actions, rewards, values, log_probs, etc.
        """
        # TODO: Implement trajectory collection
        
        raise NotImplementedError("Students must implement trajectory collection")
    
    def compute_advantages(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            next_values: Array of next state value estimates
            dones: Array of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Target values for critic update
        """
        # TODO: Implement GAE computation
        
        raise NotImplementedError("Students must implement GAE")
    
    def update_policy(self, trajectories):
        """
        Update policy using PPO clipped objective.
        
        Args:
            trajectories: Dictionary containing trajectory data
        """
        # TODO: Implement PPO policy update
        # - Compute probability ratios
        # - Compute clipped surrogate loss
        # - Add entropy bonus
        # - Backpropagate and update
        
        raise NotImplementedError("Students must implement policy update")


# Example configuration for reference
DEFAULT_CONFIG = {
    # PPO Hyperparameters
    'learning_rate': 3e-4,
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE lambda
    'epsilon': 0.2,             # PPO clipping parameter
    'entropy_coef': 0.01,       # Entropy bonus coefficient
    'value_coef': 0.5,          # Value loss coefficient
    'max_grad_norm': 0.5,       # Gradient clipping
    
    # Training parameters
    'num_steps': 2048,          # Steps per update
    'batch_size': 64,           # Minibatch size
    'num_epochs': 10,           # Epochs per update
    'num_envs': 1,              # Number of parallel environments
    
    # Network architecture
    'hidden_sizes': [64, 64],   # Hidden layer sizes
    'activation': 'tanh',       # Activation function
    
    # Logging
    'log_interval': 10,         # Episodes between logs
    'save_interval': 100,       # Episodes between saves
}
