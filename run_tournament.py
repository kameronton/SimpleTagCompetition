"""
Tournament system for SimpleTag competition.

This script supports two modes:
- local-test: Evaluate a single team against baseline heuristics
- round-robin: All teams compete against each other
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from environment import SimpleTagEnv, evaluate_agents, load_evaluation_seeds
from baselines import get_baseline_agent

class TournamentRunner:
    """
    Main tournament runner that orchestrates agent evaluation.
    """
    
    def __init__(
        self,
        submissions_dir: str = "submissions",
        results_dir: str = "results",
        num_episodes: int = 50,
    ):
        """
        Initialize tournament runner.
        
        Args:
            submissions_dir: Directory containing team submissions
            results_dir: Directory to save results
            num_episodes: Number of episodes to run per evaluation
        """
        self.submissions_dir = Path(submissions_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation seeds
        self.seeds = load_evaluation_seeds()[:num_episodes]
        
        # Initialize environment
        self.env = SimpleTagEnv(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,
            continuous_actions=False,
            render_mode=None,
        )
        print("Tournament initialized:")
        print(f"  Episodes: {len(self.seeds)}")
        print(f"  Submissions directory: {self.submissions_dir}")
    
    def load_agent_from_path(self, agent_path: Path, agent_id: str):
        """
        Dynamically load an agent from a Python file.
        
        Args:
            agent_path: Path to the agent.py file
            agent_id: Identifier for the agent
            
        Returns:
            Loaded agent instance
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(f"agent_{agent_id}", agent_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {agent_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"agent_{agent_id}"] = module
            spec.loader.exec_module(module)
            
            # Get the PPOAgent class
            if not hasattr(module, 'PPOAgent'):
                raise AttributeError(f"Module {agent_path} does not have PPOAgent class")
            
            PPOAgent = module.PPOAgent
            
            # Determine which agent type to create based on agent_id
            if 'adversary' in agent_id:
                # Predator agent
                obs_space = self.env.observation_space('adversary_0')
                action_space = self.env.action_space('adversary_0')
            else:
                # Prey agent
                obs_space = self.env.observation_space('agent_0')
                action_space = self.env.action_space('agent_0')
            
            # Instantiate agent
            agent = PPOAgent(obs_space, action_space, agent_id)
            
            return agent
            
        except Exception as e:
            print(f"Error loading agent from {agent_path}: {e}")
            raise
    
    def load_baseline_agents(self):
        """
        Load baseline agents for the current track.
        
        Returns:
            Tuple of (good_agents, adversary_agents)
        """
        if self.track == "prey":
            # Student controls good agent, baseline controls adversaries
            good_agents = []  # Will be filled with student agent
            adversary_agents = [
                get_baseline_agent(
                    'greedy_pursuit',
                    self.env.observation_space(f'adversary_{i}'),
                    self.env.action_space(f'adversary_{i}'),
                    f'adversary_{i}'
                )
                for i in range(3)
            ]
        else:  # predator
            # Student controls adversaries, baseline controls good agent
            good_agents = [
                get_baseline_agent(
                    'heuristic_prey',
                    self.env.observation_space('agent_0'),
                    self.env.action_space('agent_0'),
                    'agent_0'
                )
            ]
            adversary_agents = []  # Will be filled with student agents
        
        return good_agents, adversary_agents
    
    def evaluate_submission(self, team_name: str, agent_path: Path) -> Dict[str, Any]:
        """
        Evaluate a single submission.
        
        Args:
            team_name: Name of the team
            agent_path: Path to the agent.py file
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\nEvaluating team: {team_name}")
        print(f"  Agent path: {agent_path}")
        
        try:
            # Load baseline agents
            good_agents, adversary_agents = self.load_baseline_agents()
            
            # Load student agent(s)
            if self.track == "prey":
                # Load one good agent
                student_agent = self.load_agent_from_path(agent_path, f"{team_name}_agent_0")
                good_agents = [student_agent]
            else:  # predator
                # Load three adversary agents (or one if they use same policy)
                # For simplicity, we'll use the same agent instance for all 3
                student_agent = self.load_agent_from_path(agent_path, f"{team_name}_adversary")
                adversary_agents = [student_agent] * 3
            
            # Evaluate
            results = evaluate_agents(
                env=self.env,
                good_agents=good_agents,
                adversary_agents=adversary_agents,
                num_episodes=len(self.seeds),
                seeds=self.seeds,
                verbose=True,
            )
            
            # Add team info
            results['team_name'] = team_name
            results['track'] = self.track
            results['timestamp'] = datetime.now().isoformat()
            
            # Calculate score based on track
            if self.track == "prey":
                results['score'] = results['avg_good_reward']
            else:  # predator
                results['score'] = results['avg_adversary_reward']
            
            print("  Evaluation complete.")
            print(f"    Score: {results['score']:.2f}")
            print(f"    Avg episode length: {results['avg_episode_length']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            return {
                'team_name': team_name,
                'track': self.track,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'score': float('-inf'),
            }
    
    def find_submissions(self) -> List[tuple]:
        """
        Find all valid submissions in the submissions directory.
        Only keeps the latest submission for each github_id.
        
        Returns:
            List of (team_name, agent_path) tuples
        """
        all_submissions = []
        
        if not self.submissions_dir.exists():
            print(f"Warning: Submissions directory {self.submissions_dir} does not exist")
            return []
        
        # Look for team directories
        for team_dir in self.submissions_dir.iterdir():
            if team_dir.is_dir():
                agent_path = team_dir / "agent.py"
                metadata_path = team_dir / "metadata.json"
                
                if agent_path.exists():
                    # Read metadata if it exists
                    github_id = None
                    timestamp = None
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                github_id = metadata.get('github_id')
                                timestamp = metadata.get('timestamp')
                        except Exception as e:
                            print(f"Warning: Could not read metadata for {team_dir.name}: {e}")
                    
                    all_submissions.append({
                        'team_name': team_dir.name,
                        'agent_path': agent_path,
                        'github_id': github_id,
                        'timestamp': timestamp
                    })
        
        # Filter to keep only the latest submission per github_id
        github_id_map = {}
        for submission in all_submissions:
            github_id = submission['github_id']
            
            # If no github_id, treat each submission as unique
            if github_id is None:
                github_id = submission['team_name']  # Use team name as unique ID
            
            # Keep the latest submission for each github_id
            if github_id not in github_id_map:
                github_id_map[github_id] = submission
            else:
                # Compare timestamps (None is treated as oldest)
                existing_ts = github_id_map[github_id]['timestamp']
                new_ts = submission['timestamp']
                
                if new_ts is not None:
                    if existing_ts is None or new_ts > existing_ts:
                        github_id_map[github_id] = submission
        
        # Convert back to list of tuples
        submissions = [(s['team_name'], s['agent_path']) for s in github_id_map.values()]
        
        return submissions
    
    def get_all_contenders(self) -> List[tuple]:
        """
        Get all contenders including submissions and baseline agents.
        
        Returns:
            List of (team_name, agent_path_or_type) tuples
            For baselines, agent_path is actually the baseline type string
        """
        contenders = self.find_submissions()
        
        # Add baseline agents
        baseline_types = ['random', 'heuristic_prey', 'heuristic_predator', 'greedy_pursuit']
        for baseline_type in baseline_types:
            contenders.append((f"baseline_{baseline_type}", f"BASELINE:{baseline_type}"))
        
        return contenders
    
    def load_contender_agent(self, contender_path, agent_id: str):
        """
        Load an agent - either from a file or create a baseline.
        
        Args:
            contender_path: Either a Path to agent.py or a string like 'BASELINE:type'
            agent_id: Identifier for the agent
        
        Returns:
            Loaded agent instance
        """
        # Check if this is a baseline agent
        if isinstance(contender_path, str) and contender_path.startswith('BASELINE:'):
            baseline_type = contender_path.split(':', 1)[1]
            
            # Determine observation/action space based on agent_id
            if 'adversary' in agent_id:
                obs_space = self.env.observation_space('adversary_0')
                action_space = self.env.action_space('adversary_0')
            else:
                obs_space = self.env.observation_space('agent_0')
                action_space = self.env.action_space('agent_0')
            
            return get_baseline_agent(baseline_type, obs_space, action_space, agent_id)
        else:
            # Load from file
            return self.load_agent_from_path(contender_path, agent_id)
    
    def run_local_test(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Run local test: evaluate a team against baseline heuristics.
        
        Args:
            team_name: Name of the team to evaluate
            
        Returns:
            List of evaluation results (prey and predator tracks)
        """
        print("\n" + "="*60)
        print(f"LOCAL TEST: {team_name}")
        print("="*60)
        
        submissions = self.find_submissions()
        agent_path = None
        for name, path in submissions:
            if name == team_name:
                agent_path = path
                break
        
        if agent_path is None:
            print(f"Team '{team_name}' not found!")
            return []
        
        all_results = []
        
        # Test as prey against baseline predators
        print("\n--- Testing as PREY ---")
        try:
            student_agent = self.load_agent_from_path(agent_path, f"{team_name}_agent_0")
            baseline_predators = [
                get_baseline_agent(
                    'greedy_pursuit',
                    self.env.observation_space(f'adversary_{i}'),
                    self.env.action_space(f'adversary_{i}'),
                    f'adversary_{i}'
                )
                for i in range(3)
            ]
            
            results = evaluate_agents(
                env=self.env,
                good_agents=[student_agent],
                adversary_agents=baseline_predators,
                num_episodes=len(self.seeds),
                seeds=self.seeds,
                verbose=True,
            )
            results['team_name'] = team_name
            results['role'] = 'prey'
            results['opponent'] = 'baseline_greedy_pursuit'
            results['score'] = results['avg_good_reward']
            results['timestamp'] = datetime.now().isoformat()
            all_results.append(results)
            
            print(f"  Score as prey: {results['score']:.2f}")
            print(f"  Avg episode length: {results['avg_episode_length']:.1f}")
        except Exception as e:
            print(f"  Failed as prey: {e}")
            all_results.append({
                'team_name': team_name,
                'role': 'prey',
                'opponent': 'baseline_greedy_pursuit',
                'error': str(e),
                'score': float('-inf'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Test as predator against baseline prey
        print("\n--- Testing as PREDATOR ---")
        try:
            student_agent = self.load_agent_from_path(agent_path, f"{team_name}_adversary")
            baseline_prey = get_baseline_agent(
                'heuristic_prey',
                self.env.observation_space('agent_0'),
                self.env.action_space('agent_0'),
                'agent_0'
            )
            
            results = evaluate_agents(
                env=self.env,
                good_agents=[baseline_prey],
                adversary_agents=[student_agent] * 3,
                num_episodes=len(self.seeds),
                seeds=self.seeds,
                verbose=True,
            )
            results['team_name'] = team_name
            results['role'] = 'predator'
            results['opponent'] = 'baseline_heuristic_prey'
            results['score'] = results['avg_adversary_reward']
            results['timestamp'] = datetime.now().isoformat()
            all_results.append(results)
            
            print(f"  Score as predator: {results['score']:.2f}")
            print(f"  Avg episode length: {results['avg_episode_length']:.1f}")
        except Exception as e:
            print(f"  Failed as predator: {e}")
            all_results.append({
                'team_name': team_name,
                'role': 'predator',
                'opponent': 'baseline_heuristic_prey',
                'error': str(e),
                'score': float('-inf'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"local_test_{team_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
        
        # Print summary
        self.print_local_test_summary(all_results)
        
        return all_results
    
    def print_local_test_summary(self, results: List[Dict[str, Any]]):
        """
        Print a summary of local test results.
        """
        print("\n" + "="*60)
        print("LOCAL TEST SUMMARY")
        print("="*60)
        
        if not results:
            print("No results to display")
            return
        
        team_name = results[0].get('team_name', 'Unknown')
        print(f"Team: {team_name}")
        print("-"*60)
        
        for result in results:
            role = result.get('role', 'unknown')
            opponent = result.get('opponent', 'unknown')
            
            print(f"\n{role.upper()} vs {opponent}")
            
            if 'error' in result:
                print(f"  Status: ❌ FAILED")
                print(f"  Error: {result['error']}")
            else:
                score = result.get('score', 0)
                avg_len = result.get('avg_episode_length', 0)
                print(f"  Status: ✓ SUCCESS")
                print(f"  Score: {score:.2f}")
                print(f"  Avg Episode Length: {avg_len:.1f}")
        
        print("\n" + "="*60)
    
    def print_round_robin_leaderboard(self, matchup_results: List[Dict[str, Any]]):
        """
        Print a summary leaderboard for round-robin results.
        Only includes submissions, not baselines.
        """
        print("\n" + "="*60)
        print("ROUND-ROBIN LEADERBOARD (Submissions Only)")
        print("="*60)
        
        # Collect scores per team
        team_scores = {}
        
        for result in matchup_results:
            if 'error' in result:
                continue
            
            team_a = result['team_a']
            team_b = result['team_b']
            role_a = result['team_a_role']
            role_b = result['team_b_role']
            
            # Skip if both are baselines (shouldn't happen, but just in case)
            if team_a.startswith('baseline_') and team_b.startswith('baseline_'):
                continue
            
            # Only track scores for non-baseline teams
            if not team_a.startswith('baseline_'):
                if team_a not in team_scores:
                    team_scores[team_a] = {'as_prey': [], 'as_predator': []}
            if not team_b.startswith('baseline_'):
                if team_b not in team_scores:
                    team_scores[team_b] = {'as_prey': [], 'as_predator': []}
            
            # Add scores based on roles (only for submissions)
            if role_a == 'prey' and not team_a.startswith('baseline_'):
                team_scores[team_a]['as_prey'].append(
                    result['avg_good_reward']
                )
            if role_b == 'predator' and not team_b.startswith('baseline_'):
                team_scores[team_b]['as_predator'].append(
                    result['avg_adversary_reward']
                )
            if role_a == 'predator' and not team_a.startswith('baseline_'):
                team_scores[team_a]['as_predator'].append(
                    result['avg_adversary_reward']
                )
            if role_b == 'prey' and not team_b.startswith('baseline_'):
                team_scores[team_b]['as_prey'].append(
                    result['avg_good_reward']
                )
        
        # Calculate averages and total scores
        leaderboard = []
        for team, scores in team_scores.items():
            prey_scores = scores['as_prey']
            pred_scores = scores['as_predator']
            prey_avg = sum(prey_scores) / len(prey_scores) if prey_scores else 0
            pred_avg = sum(pred_scores) / len(pred_scores) if pred_scores else 0
            total = prey_avg + pred_avg
            leaderboard.append((team, prey_avg, pred_avg, total))
        
        # Sort by total score
        leaderboard.sort(key=lambda x: x[3], reverse=True)
        
        print(f"{'Rank':<6} {'Team':<20} {'Prey':<12} "
              f"{'Predator':<12} {'Total':<10}")
        print("-"*60)
        
        for rank, (team, prey_avg, pred_avg, total) in enumerate(
            leaderboard, 1
        ):
            print(f"{rank:<6} {team:<20} {prey_avg:<12.2f} "
                  f"{pred_avg:<12.2f} {total:<10.2f}")
        
        print("="*60)

    def run_round_robin(self) -> List[Dict[str, Any]]:
        """
        Run a round-robin tournament: each submission plays every other
        submission and every baseline in both prey and predator roles.
        Returns:
            List of evaluation results for all matchups
        """
        print("\n" + "="*60)
        print("SIMPLETAG ROUND-ROBIN TOURNAMENT")
        print("="*60)

        submissions = self.find_submissions()
        if len(submissions) < 1:
            print("Need at least 1 submission for round-robin!")
            return []

        # Get baseline contenders
        baseline_types = ['random', 'heuristic_prey', 'heuristic_predator', 'greedy_pursuit']
        baselines = [(f"baseline_{bt}", f"BASELINE:{bt}") for bt in baseline_types]

        print(f"\nFound {len(submissions)} submission(s):")
        for team_name, _ in submissions:
            print(f"  - {team_name}")
        print(f"\nIncluding {len(baselines)} baseline(s) as opponents:")
        for baseline_name, _ in baselines:
            print(f"  - {baseline_name}")

        matchup_results = []

        # Submissions vs Submissions
        for i, (team_a, contender_a) in enumerate(submissions):
            for j, (team_b, contender_b) in enumerate(submissions):
                if i >= j:
                    continue  # Skip self-play and duplicates

                # Match 1: Team A is prey, Team B is predator
                print(f"\nMatch: {team_a} (prey) vs {team_b} (predator)")
                try:
                    prey_agent = self.load_contender_agent(
                        contender_a, f"{team_a}_agent_0"
                    )
                    pred_agent = self.load_contender_agent(
                        contender_b, f"{team_b}_adversary"
                    )
                    
                    results = evaluate_agents(
                        env=self.env,
                        good_agents=[prey_agent],
                        adversary_agents=[pred_agent] * 3,
                        num_episodes=len(self.seeds),
                        seeds=self.seeds,
                        verbose=True,
                    )
                    results['team_a'] = team_a
                    results['team_b'] = team_b
                    results['team_a_role'] = 'prey'
                    results['team_b_role'] = 'predator'
                    results['timestamp'] = datetime.now().isoformat()
                    matchup_results.append(results)
                except Exception as e:
                    print(f"  Match failed: {e}")
                    matchup_results.append({
                        'team_a': team_a,
                        'team_b': team_b,
                        'team_a_role': 'prey',
                        'team_b_role': 'predator',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

                # Match 2: Team B is prey, Team A is predator
                print(f"\nMatch: {team_b} (prey) vs {team_a} (predator)")
                try:
                    prey_agent = self.load_contender_agent(
                        contender_b, f"{team_b}_agent_0"
                    )
                    pred_agent = self.load_contender_agent(
                        contender_a, f"{team_a}_adversary"
                    )
                    
                    results = evaluate_agents(
                        env=self.env,
                        good_agents=[prey_agent],
                        adversary_agents=[pred_agent] * 3,
                        num_episodes=len(self.seeds),
                        seeds=self.seeds,
                        verbose=True,
                    )
                    results['team_a'] = team_b
                    results['team_b'] = team_a
                    results['team_a_role'] = 'prey'
                    results['team_b_role'] = 'predator'
                    results['timestamp'] = datetime.now().isoformat()
                    matchup_results.append(results)
                except Exception as e:
                    print(f"  Match failed: {e}")
                    matchup_results.append({
                        'team_a': team_b,
                        'team_b': team_a,
                        'team_a_role': 'prey',
                        'team_b_role': 'predator',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        # Submissions vs Baselines
        for team_name, team_path in submissions:
            for baseline_name, baseline_path in baselines:
                # Match 1: Submission is prey, Baseline is predator
                print(f"\nMatch: {team_name} (prey) vs {baseline_name} (predator)")
                try:
                    prey_agent = self.load_contender_agent(
                        team_path, f"{team_name}_agent_0"
                    )
                    pred_agent = self.load_contender_agent(
                        baseline_path, f"{baseline_name}_adversary"
                    )
                    
                    results = evaluate_agents(
                        env=self.env,
                        good_agents=[prey_agent],
                        adversary_agents=[pred_agent] * 3,
                        num_episodes=len(self.seeds),
                        seeds=self.seeds,
                        verbose=True,
                    )
                    results['team_a'] = team_name
                    results['team_b'] = baseline_name
                    results['team_a_role'] = 'prey'
                    results['team_b_role'] = 'predator'
                    results['timestamp'] = datetime.now().isoformat()
                    matchup_results.append(results)
                except Exception as e:
                    print(f"  Match failed: {e}")
                    matchup_results.append({
                        'team_a': team_name,
                        'team_b': baseline_name,
                        'team_a_role': 'prey',
                        'team_b_role': 'predator',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

                # Match 2: Baseline is prey, Submission is predator
                print(f"\nMatch: {baseline_name} (prey) vs {team_name} (predator)")
                try:
                    prey_agent = self.load_contender_agent(
                        baseline_path, f"{baseline_name}_agent_0"
                    )
                    pred_agent = self.load_contender_agent(
                        team_path, f"{team_name}_adversary"
                    )
                    
                    results = evaluate_agents(
                        env=self.env,
                        good_agents=[prey_agent],
                        adversary_agents=[pred_agent] * 3,
                        num_episodes=len(self.seeds),
                        seeds=self.seeds,
                        verbose=True,
                    )
                    results['team_a'] = baseline_name
                    results['team_b'] = team_name
                    results['team_a_role'] = 'prey'
                    results['team_b_role'] = 'predator'
                    results['timestamp'] = datetime.now().isoformat()
                    matchup_results.append(results)
                except Exception as e:
                    print(f"  Match failed: {e}")
                    matchup_results.append({
                        'team_a': baseline_name,
                        'team_b': team_name,
                        'team_a_role': 'prey',
                        'team_b_role': 'predator',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_robin_{timestamp}.json"
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(matchup_results, f, indent=2)
        print(f"\nRound-robin results saved to: {filepath}")

        # Print leaderboard
        self.print_round_robin_leaderboard(matchup_results)

        return matchup_results


def main():
    """Main entry point for tournament runner."""
    parser = argparse.ArgumentParser(
        description="Run SimpleTag competition tournament"
    )
    
    parser.add_argument(
        '--submissions-dir',
        type=str,
        default='submissions',
        help='Directory containing team submissions'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--team',
        type=str,
        default=None,
        help='Team name for local test mode'
    )
    
    parser.add_argument(
        '--local-test',
        action='store_true',
        help='Run local test against baseline heuristics (requires --team)'
    )
    
    parser.add_argument(
        '--round-robin',
        action='store_true',
        help='Run round-robin tournament (all teams vs all teams)'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=50,
        help='Number of episodes to run (default: 50, use 10 for quick tests)'
    )
    
    args = parser.parse_args()
    
    # Initialize tournament
    runner = TournamentRunner(
        submissions_dir=args.submissions_dir,
        results_dir=args.results_dir,
        num_episodes=args.num_episodes,
    )
    
    # Run appropriate mode
    if args.local_test:
        if not args.team:
            print("Error: --local-test requires --team argument")
            return
        runner.run_local_test(args.team)
    elif args.round_robin:
        runner.run_round_robin()
    else:
        print("Error: Must specify either --local-test or --round-robin")
        print("Examples:")
        print("  python run_tournament.py --local-test --team example_team")
        print("  python run_tournament.py --round-robin")


if __name__ == "__main__":
    main()
