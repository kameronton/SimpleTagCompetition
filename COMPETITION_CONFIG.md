# Competition Configuration

## Fixed Seeds

The competition uses fixed random seeds for reproducible evaluation:

- **Public Leaderboard**: 50 episodes with seeds generated from base seed 42
- **Private Leaderboard**: 200 episodes with different seeds generated from base seed 42

Seeds are generated using `environment.generate_evaluation_seeds()`.

## Environment Settings

- **Number of Good Agents**: 1 (prey)
- **Number of Adversary Agents**: 3 (predators)
- **Number of Obstacles**: 2
- **Max Cycles per Episode**: 25
- **Action Space**: Discrete(5) - [no-op, up, down, left, right]
- **Observation Space**: Continuous (varies by agent, typically 14-16 dimensions)

## Tracks

### Prey Track
- Students control the good agent (prey)
- Compete against baseline adversary agents
- Goal: Maximize survival time and reward

### Predator Track
- Students control adversary agents (predators)
- Compete against baseline good agent
- Goal: Catch prey efficiently and maximize reward

## Scoring

Score is calculated as the average episode reward over all evaluation episodes:

- **Prey Track**: `score = avg_good_reward`
- **Predator Track**: `score = avg_adversary_reward`

## Baselines

Available baseline agents:
- `random`: Random action selection
- `heuristic_prey`: Rule-based evasion strategy
- `heuristic_predator`: Rule-based pursuit strategy
- `greedy_pursuit`: More sophisticated pursuit with prediction

Students must outperform these baselines to be competitive.

## Submission Format

Teams must submit:
1. `agent.py` - Implementation of PPOAgent class
2. `model_weights.pth` - Trained model weights
3. (Optional) `config.json` - Configuration file

## Evaluation Process

1. Load student's PPOAgent from `submissions/team_name/agent.py`
2. Load baseline agents for the opposite role
3. Run episodes with fixed seeds
4. Compute average rewards and statistics
5. Update leaderboard

## Hardware Constraints

- Evaluation runs on CPU
- Model size limit: 100 MB
- Timeout: 5 seconds per episode (on standard hardware)

## Rules

1. Must implement PPO algorithm (not just use pre-trained models)
2. No hardcoding based on test seeds
3. Fair play - no adversarial attacks on evaluation system
4. Code must be original work (can reference implementations but no copying)
