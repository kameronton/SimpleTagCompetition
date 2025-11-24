"""
Example PPO training script for SimpleTag competition.

This script provides a skeleton for implementing PPO from scratch.
Students should use this as a starting point and implement the missing parts.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from environment import SimpleTagEnv

class PPONetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    This network outputs both policy logits and value estimates.
    """
    
    def __init__(self, observation_dim, action_dim, hidden_sizes=[64, 64]):
        """
        Initialize the network.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
        """
        super(PPONetwork, self).__init__()
        
        # Build shared feature extractor
        layers = []
        prev_size = observation_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(prev_size, action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Observation tensor
            
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.shared(x)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action(self, observation, deterministic=False):
        """
        Select an action given an observation.
        
        Args:
            observation: numpy array
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: Value estimate
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.forward(obs_tensor)
        
        # Create categorical distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()


class PPOBuffer:
    """
    Buffer for storing trajectories and computing advantages.
    """
    
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        """
        Initialize buffer.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, observation, action, reward, value, log_prob, done):
        """Store a transition."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, last_value=0):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            last_value: Value estimate for the final state
            
        Returns:
            advantages: Computed advantages
            returns: Target values for critic
        """
        # TODO: Implement GAE computation
        
        # Hint: GAE formula is:
        # A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        # where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute advantages backwards through time
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # Returns are advantages + values
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def get(self):
        """
        Get all stored data and clear buffer.
        
        Returns:
            Dictionary containing all trajectory data
        """
        data = {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones),
        }
        
        # Clear buffer
        self.__init__(self.gamma, self.gae_lambda)
        
        return data
    
    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO trainer for SimpleTag environment.
    """
    
    def __init__(
        self,
        env,
        network,
        track='prey',
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=10,
        batch_size=64,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env: SimpleTagEnv instance
            network: PPONetwork instance
            track: 'prey' or 'predator'
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs per update
            batch_size: Minibatch size
        """
        self.env = env
        self.network = network
        self.track = track
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.buffer = PPOBuffer(gamma, gae_lambda)
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def update(self, advantages, returns, old_log_probs, observations, actions):
        """
        Update policy using PPO objective.
        
        Args:
            advantages: Computed advantages
            returns: Target returns for value function
            old_log_probs: Log probabilities from old policy
            observations: State observations
            actions: Actions taken
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Multiple epochs of updates
        for _ in range(self.num_epochs):
            # Create mini-batches
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # TODO: Implement PPO loss computation
                # Students must implement this critical part
                
                # Forward pass
                action_logits, values = self.network(batch_obs)
                
                # Compute current log probabilities
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Compute probability ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                
                # PPO policy loss (we want to maximize, so minimize negative)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Entropy bonus (encourage exploration)
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def train(self, num_episodes=1000, steps_per_update=2048):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train
            steps_per_update: Number of steps before PPO update
        """
        print(f"Starting PPO training for {self.track} track...")
        print(f"Episodes: {num_episodes}, Steps per update: {steps_per_update}")
        
        episode_count = 0
        total_steps = 0
        
        while episode_count < num_episodes:
            # Reset environment
            observations = self.env.reset(seed=None)
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and len(self.buffer) < steps_per_update:
                # TODO: Students should understand this rollout process
                
                # Get actions for all agents
                # For simplicity, we'll control the relevant agent(s)
                actions = {}
                
                if self.track == 'prey':
                    # Control good agent
                    agent_name = 'agent_0'
                    obs = observations[agent_name]
                    action, log_prob, value = self.network.get_action(obs)
                    actions[agent_name] = action
                    
                    # Store in buffer
                    self.buffer.store(obs, action, 0, value, log_prob, False)
                    
                    # Random actions for adversaries (or use baselines)
                    for adv_name in self.env.adversary_agents:
                        actions[adv_name] = np.random.randint(0, 5)
                
                else:  # predator
                    # Control adversaries
                    # For simplicity, we'll use shared policy
                    for adv_name in self.env.adversary_agents:
                        obs = observations[adv_name]
                        action, log_prob, value = self.network.get_action(obs)
                        actions[adv_name] = action
                        self.buffer.store(obs, action, 0, value, log_prob, False)
                    
                    # Random action for good agent
                    actions['agent_0'] = np.random.randint(0, 5)
                
                # Step environment
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Update rewards in buffer
                if self.track == 'prey':
                    reward = rewards.get('agent_0', 0)
                    episode_reward += reward
                    self.buffer.rewards[-1] = reward
                else:
                    reward = np.mean([rewards.get(a, 0) for a in self.env.adversary_agents])
                    episode_reward += reward
                    # Update last N rewards (one per adversary)
                    for i in range(len(self.env.adversary_agents)):
                        if len(self.buffer.rewards) > i:
                            self.buffer.rewards[-(i+1)] = rewards.get(f'adversary_{i}', 0)
                
                episode_length += 1
                total_steps += 1
                
                # Check if done
                done = all(terminations.values()) or all(truncations.values())
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    episode_count += 1
                    
                    # Reset for next episode if we haven't collected enough steps
                    if len(self.buffer) < steps_per_update:
                        observations = self.env.reset(seed=None)
                        done = False
                        episode_reward = 0
                        episode_length = 0
            
            # Perform PPO update
            if len(self.buffer) >= steps_per_update or episode_count >= num_episodes:
                # Compute advantages
                last_value = 0  # Could use bootstrap value here
                advantages, returns = self.buffer.compute_advantages(last_value)
                
                # Get data
                data = self.buffer.get()
                
                # Update policy
                self.update(
                    advantages,
                    returns,
                    data['log_probs'],
                    data['observations'],
                    data['actions']
                )
                
                # Log progress
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    print(f"Episode {episode_count}/{num_episodes} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Length: {avg_length:.1f} | "
                          f"Total Steps: {total_steps}")
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for SimpleTag")
    
    parser.add_argument('--track', type=str, choices=['prey', 'predator'], default='prey',
                        help='Competition track')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--steps-per-update', type=int, default=2048,
                        help='Steps before PPO update')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--save-path', type=str,
                        help='Path to save model (default: based on track)')
    
    args = parser.parse_args()

    # Set default save path based on track if not provided
    if args.save_path is None:
        args.save_path = f"examples/model_weights_{args.track}.pth"

    # Create environment
    env = SimpleTagEnv()
    
    # Determine observation and action dimensions
    if args.track == 'prey':
        obs_space = env.observation_space('agent_0')
        action_space = env.action_space('agent_0')
    else:
        obs_space = env.observation_space('adversary_0')
        action_space = env.action_space('adversary_0')
    
    obs_dim = obs_space.shape[0]
    action_dim = action_space.n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create network
    network = PPONetwork(obs_dim, action_dim, hidden_sizes=[64, 64])
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        network=network,
        track=args.track,
        learning_rate=args.lr,
    )
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        steps_per_update=args.steps_per_update,
    )
    
    # Save model
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer.save(args.save_path)
    
    env.close()


if __name__ == "__main__":
    main()
