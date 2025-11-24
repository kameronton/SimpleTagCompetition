"""
Baseline agents for the SimpleTag competition.

These agents serve as benchmarks for student submissions.
"""

import numpy as np
from typing import Any, Optional


class RandomAgent:
    """
    Random baseline agent that selects actions uniformly at random.
    
    This is the weakest baseline and should be easily beaten.
    """
    
    def __init__(self, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
        """Initialize random agent."""
        self.action_space = action_space
        self.agent_id = agent_id
        self.num_actions = action_space.n
    
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select a random action."""
        return np.random.randint(0, self.num_actions)


class HeuristicPreyAgent:
    """
    Heuristic prey agent that uses simple rules to evade predators.
    
    Strategy:
    - Calculate distance to all adversaries
    - Move away from the nearest adversary
    - Stay within boundaries
    """
    
    def __init__(self, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
        """Initialize heuristic prey agent."""
        self.action_space = action_space
        self.agent_id = agent_id
        
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action based on heuristic rules.
        
        Observation structure for good agent (prey):
        - Own velocity (2 values)
        - Own position (2 values)
        - Landmark positions (2 values each)
        - Adversary positions relative to self (2 values each)
        """
        # Action mapping: 0=no-op, 1=up, 2=down, 3=left, 4=right
        
        # Extract information from observation
        # Note: The exact structure depends on the environment configuration
        # This is a simplified heuristic
        
        try:
            # Assuming observation contains relative positions to adversaries
            # Find the direction of the nearest threat
            obs_len = len(observation)
            
            # Skip velocity and position, look at relative positions
            # The observation typically starts with self info, then others
            if obs_len >= 6:  # At least one adversary position
                # Get relative positions (simplified)
                adversary_positions = []
                start_idx = 4  # After velocity and position
                
                for i in range(start_idx, obs_len, 2):
                    if i + 1 < obs_len:
                        adversary_positions.append([observation[i], observation[i+1]])
                
                if adversary_positions:
                    # Find nearest adversary
                    distances = [np.sqrt(pos[0]**2 + pos[1]**2) for pos in adversary_positions]
                    nearest_idx = np.argmin(distances)
                    nearest_pos = adversary_positions[nearest_idx]
                    
                    # Move away from nearest adversary
                    dx, dy = nearest_pos[0], nearest_pos[1]
                    
                    # Choose action that moves away
                    if abs(dx) > abs(dy):
                        # Move horizontally
                        if dx > 0:
                            return 3  # Move left (away from adversary on right)
                        else:
                            return 4  # Move right (away from adversary on left)
                    else:
                        # Move vertically
                        if dy > 0:
                            return 2  # Move down (away from adversary above)
                        else:
                            return 1  # Move up (away from adversary below)
            
            # If no clear threat, move randomly
            return np.random.randint(1, 5)  # Avoid no-op
            
        except Exception:
            # Fallback to random action
            return np.random.randint(1, 5)


class HeuristicPredatorAgent:
    """
    Heuristic predator agent that uses simple rules to chase prey.
    
    Strategy:
    - Calculate distance to prey
    - Move towards prey
    - Simple coordination (avoid clustering)
    """
    
    def __init__(self, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
        """Initialize heuristic predator agent."""
        self.action_space = action_space
        self.agent_id = agent_id
        
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action based on pursuit heuristic.
        
        Observation structure for adversary (predator):
        - Own velocity (2 values)
        - Own position (2 values)
        - Landmark positions (2 values each)
        - Good agent position relative to self (2 values)
        - Other adversary positions (2 values each)
        """
        # Action mapping: 0=no-op, 1=up, 2=down, 3=left, 4=right
        
        try:
            # Extract prey position (simplified)
            obs_len = len(observation)
            
            if obs_len >= 6:
                # Prey position is typically after self info and landmarks
                # This is a simplified version
                start_idx = 4  # After velocity and position
                
                # Assuming the first relative position is the prey
                if start_idx + 1 < obs_len:
                    prey_dx = observation[start_idx]
                    prey_dy = observation[start_idx + 1]
                    
                    # Move towards prey
                    if abs(prey_dx) > abs(prey_dy):
                        # Move horizontally towards prey
                        if prey_dx > 0:
                            return 4  # Move right
                        else:
                            return 3  # Move left
                    else:
                        # Move vertically towards prey
                        if prey_dy > 0:
                            return 1  # Move up
                        else:
                            return 2  # Move down
            
            # Fallback: move randomly
            return np.random.randint(1, 5)
            
        except Exception:
            # Fallback to random action
            return np.random.randint(1, 5)


class GreedyPursuitAgent:
    """
    More sophisticated pursuit agent using greedy pursuit with simple prediction.
    
    This provides a stronger baseline for students to beat.
    """
    
    def __init__(self, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
        """Initialize greedy pursuit agent."""
        self.action_space = action_space
        self.agent_id = agent_id
        self.previous_prey_pos = None
        
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action using greedy pursuit with velocity consideration.
        """
        try:
            obs_len = len(observation)
            
            if obs_len >= 6:
                # Own velocity may be present but is unused in this heuristic
                # _own_vx, _own_vy = observation[0], observation[1]
                
                # Get prey relative position
                start_idx = 4
                if start_idx + 1 < obs_len:
                    prey_dx = observation[start_idx]
                    prey_dy = observation[start_idx + 1]
                    
                    # Simple pursuit with momentum consideration
                    # Predict where prey will be
                    if self.previous_prey_pos is not None:
                        prey_vx = prey_dx - self.previous_prey_pos[0]
                        prey_vy = prey_dy - self.previous_prey_pos[1]
                        # Predict next position
                        predicted_dx = prey_dx + prey_vx * 0.5
                        predicted_dy = prey_dy + prey_vy * 0.5
                    else:
                        predicted_dx = prey_dx
                        predicted_dy = prey_dy
                    
                    self.previous_prey_pos = [prey_dx, prey_dy]
                    
                    # Choose action to intercept
                    if abs(predicted_dx) > abs(predicted_dy):
                        if predicted_dx > 0:
                            return 4  # Move right
                        else:
                            return 3  # Move left
                    else:
                        if predicted_dy > 0:
                            return 1  # Move up
                        else:
                            return 2  # Move down
            
            return np.random.randint(1, 5)
            
        except Exception:
            return np.random.randint(1, 5)


# Dictionary for easy access to baseline agents
BASELINE_AGENTS = {
    'random': RandomAgent,
    'heuristic_prey': HeuristicPreyAgent,
    'heuristic_predator': HeuristicPredatorAgent,
    'greedy_pursuit': GreedyPursuitAgent,
}


def get_baseline_agent(agent_type: str, observation_space: Any, action_space: Any, agent_id: Optional[str] = None):
    """
    Factory function to create baseline agents.
    
    Args:
        agent_type: Type of baseline agent ('random', 'heuristic_prey', 'heuristic_predator', 'greedy_pursuit')
        observation_space: Environment observation space
        action_space: Environment action space
        agent_id: Agent identifier
        
    Returns:
        Instance of the requested baseline agent
    """
    if agent_type not in BASELINE_AGENTS:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(BASELINE_AGENTS.keys())}")
    
    agent_class = BASELINE_AGENTS[agent_type]
    return agent_class(observation_space, action_space, agent_id)
