"""
Test script to verify the SimpleTag environment is working correctly.
"""

from environment import SimpleTagEnv, visualize_episode
from baselines import get_baseline_agent

# A dirty trick to suppress pygame warnings during tests
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

def test_environment():
    """Test basic environment functionality."""
    print("Testing SimpleTag Environment...")
    print("-" * 60)
    
    # Create environment
    env = SimpleTagEnv(render_mode=None)
    
    print("Environment created successfully")
    print(f"  Good agents: {env.good_agents}")
    print(f"  Adversary agents: {env.adversary_agents}")
    print(f"  Max cycles: {env.max_cycles}")
    
    # Test observation and action spaces
    print("\nObservation and Action Spaces:")
    for agent_name in env.all_agents:
        obs_space = env.observation_space(agent_name)
        action_space = env.action_space(agent_name)
        print(f"  {agent_name}:")
        print(f"    Observation shape: {obs_space.shape}")
        print(f"    Action space: {action_space}")
    
    # Test reset
    print("\nTesting reset...")
    observations = env.reset(seed=42)
    print(f"Reset successful, got {len(observations)} observations")
    
    # Test step
    print("\nTesting episode...")
    actions = {agent: 0 for agent in observations.keys()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("Step successful")
    print(f"  Rewards: {rewards}")
    
    # Run a full episode
    print("\nRunning full episode with random actions...")
    observations = env.reset(seed=42)
    done = False
    step_count = 0
    total_rewards = {agent: 0 for agent in env.all_agents}
    
    while not done and step_count < env.max_cycles:
        actions = {agent: env.action_space(agent).sample() for agent in observations.keys()}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        
        step_count += 1
        done = all(terminations.values()) or all(truncations.values())
    
    print(f"Episode completed in {step_count} steps")
    print(f"  Total rewards: {total_rewards}")
    
    stats = env.get_episode_stats()
    print(f"  Episode stats: {stats}")
    
    env.close()
    print("\nAll tests passed!")


def test_baseline_agents():
    """Test baseline agents."""
    print("\n" + "=" * 60)
    print("Testing Baseline Agents...")
    print("=" * 60)
    
    env = SimpleTagEnv(render_mode=None)
    
    # Test with heuristic agents
    print("\nTesting heuristic prey vs greedy pursuit...")
    
    good_agents = [
        get_baseline_agent('heuristic_prey', env.observation_space('agent_0'), 
                  env.action_space('agent_0'), 'agent_0')
    ]
    
    adversary_agents = [
        get_baseline_agent('greedy_pursuit', env.observation_space(f'adversary_{i}'),
                  env.action_space(f'adversary_{i}'), f'adversary_{i}')
        for i in range(3)
    ]
    
    observations = env.reset(seed=42)
    done = False
    step_count = 0
    
    while not done and step_count < env.max_cycles:
        actions = {}
        
        # Get actions from agents
        for i, agent_name in enumerate(env.good_agents):
            if agent_name in observations:
                actions[agent_name] = good_agents[i].act(observations[agent_name])
        
        for i, agent_name in enumerate(env.adversary_agents):
            if agent_name in observations:
                actions[agent_name] = adversary_agents[i].act(observations[agent_name])
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        done = all(terminations.values()) or all(truncations.values())
    
    stats = env.get_episode_stats()
    print(f"Episode completed in {step_count} steps")
    print(f"  Good agent reward: {stats['avg_reward']['good']:.2f}")
    print(f"  Adversary reward: {stats['avg_reward']['adversary']:.2f}")
    print(f"  Catches: {stats['catches']}")
    
    env.close()
    print("\nBaseline agent tests passed!")


def test_visualization():
    """Test visualization (requires rendering)."""
    print("\n" + "=" * 60)
    print("Testing Visualization...")
    print("=" * 60)
    
    try:
        import importlib.util
        if importlib.util.find_spec('pygame') is None:
            print("Pygame not installed, skipping visualization test")
            print("  Install with: pip install pygame")
            return

        import pygame

        print("\nCreating visualization environment...")
        env = SimpleTagEnv(render_mode='human')
        
        good_agents = [
            get_baseline_agent('heuristic_prey', env.observation_space('agent_0'),
                              env.action_space('agent_0'), 'agent_0')
        ]
        
        adversary_agents = [
            get_baseline_agent('random', env.observation_space(f'adversary_{i}'),
                              env.action_space(f'adversary_{i}'), f'adversary_{i}')
            for i in range(3)
        ]
        
        input("\nPress Enter to start visualization...")

        print("Running visual episode (close window to continue)...")
        visualize_episode(env, good_agents, adversary_agents, seed=42)
        
        # Keep window open until user closes it
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        env.close()
        print("Visualization test passed!")
        
    except ImportError:
        print("Pygame not installed, skipping visualization test")
        print("  Install with: pip install pygame")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLETAG ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    try:
        test_environment()
        test_baseline_agents()
        test_visualization()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to start training your PPO agent!")
        print("Try running: python -m examples.train_ppo --track prey --episodes 100")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
