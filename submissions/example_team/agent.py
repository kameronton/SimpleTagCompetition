"""
Example submission for SimpleTag competition.

NOTE: The tournament system determines whether your agent plays as prey or predator
based on the '--track' argument. Your agent will be instantiated with the correct
observation and action spaces for the role.

Replace the implementation with your trained PPO agent.
"""

import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    """Your neural network architecture."""
    
    def __init__(self, observation_dim, action_dim):
        super(PPONetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value


class PPOAgent:
    """
    PPO Agent for SimpleTag competition.

    The agent_id is set by the tournament system to help distinguish agent roles:
    - For prey: agent_id will contain 'agent_0'
    - For predator: agent_id will contain 'adversary'
    Use this to load the correct weights for each role.
    """
    
    def __init__(self, observation_space, action_space, agent_id=None):
        """
        Initialize your agent.
        
        Args:
            observation_space: The observation space
            action_space: The action space
            agent_id: Agent identifier
        """
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.n
        self.agent_id = agent_id
        
        self.network = PPONetwork(self.observation_dim, self.action_dim)

        # Determine weights file based on agent type
        import os
        base_dir = os.path.dirname(__file__)
        if agent_id is not None and "adversary" in agent_id:
            weights_path = os.path.join(base_dir, "model_weights_predator.pth")
        else:
            weights_path = os.path.join(base_dir, "model_weights_prey.pth")
        try:
            self.load_weights(weights_path)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    def act(self, observation, deterministic=True):
        """
        Select an action given an observation.
        
        Args:
            observation: numpy array of the current state
            deterministic: If True, select argmax action
            
        Returns:
            action: Integer from 0 to 4
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get action from network
        with torch.no_grad():
            action_logits, _ = self.network(obs_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
        
        return action
    
    def load_weights(self, weights_path):
        """
        Load model weights from file.
        
        Args:
            weights_path: Path to the weights file
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
        else:
            self.network.load_state_dict(checkpoint)
        
        self.network.eval()  # Set to evaluation mode
