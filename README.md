# SimpleTag RL Competition

Welcome to the SimpleTag Reinforcement Learning Competition! This competition is designed as a final project for the RL course, where you'll implement and train PPO (Proximal Policy Optimization) agents to compete in the SimpleTag environment from PettingZoo's Multi-Agent Particle Environment (MPE) suite.

## Environment Overview

**SimpleTag** is a competitive multi-agent environment with:
- **Continuous State Space**: Agents observe relative positions and velocities (requires neural network-based PPO)
- **Discrete Action Space**: 5 actions - No-op, Move Up, Move Down, Move Left, Move Right
- **Two Roles**:
  - **Adversaries (Red/Predators)**: 3 slower agents that try to catch the good agent
  - **Good Agent (Green/Prey)**: 1 faster agent that tries to avoid being caught

## Competition Tracks

You will participate in both tracks:

### Track 1: Prey Master
- **Your Goal**: Train a single Good Agent (prey) that survives as long as possible
- **Opponent**: Your agent competes against a fixed baseline of 3 Adversary agents
- **Reward**: Negative reward when caught, penalty for leaving boundaries
- **Strategy**: Exploit speed advantage and map boundaries

### Track 2: Predator Master
- **Your Goal**: Train 3 Adversary agents (predators) that catch the prey efficiently
- **Opponent**: Your agents compete against a fixed baseline Good Agent
- **Reward**: Positive reward for catching the prey
- **Strategy**: Learn coordination and pursuit tactics

## Requirements

- Python 3.8+
- PyTorch
- PettingZoo[mpe]
- See `requirements.txt` for full dependencies

## Quick Start

### 1. Installation

We recommend using `uv` for faster dependency management, but `pip` with `venv` is also supported.

<details>
<summary>Option 1: Using uv (Recommended)</summary>

```bash
# Clone the repository
git clone <your-repo-url>
cd SimpleTagCompetition

# Install uv if you don't have it
pip install uv

# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync requirements.txt
```

</details>

<details>
<summary>Option 2: Using pip and venv</summary>

```bash
# Clone the repository
git clone <your-repo-url>
cd SimpleTagCompetition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
</details>

### 2. Test the Environment

```bash
python test_environment.py
```

### 3. Train Your Agent

Use the provided training template:

```bash
python -m examples.train_ppo --track prey  # or --track predator
```

### 4. Test Your Agent Locally

```bash
# Test against baselines
python run_tournament.py --local-test --team your_team_name --num-episodes 50

# Run full round-robin tournament
python run_tournament.py --round-robin --num-episodes 50
```

## Submission Guidelines

### File Structure

Your submission should be organized as:

```
submissions/
└── your_team_name/
    ├── agent.py              # Your PPOAgent implementation
    ├── model_weights.pth     # Your trained model weights
    └── config.json           # (Optional) Any configuration
```

### Agent Implementation

Your `agent.py` must implement the `PPOAgent` class as shown in `agent_template.py`:

```python
class PPOAgent:
    def __init__(self, observation_space, action_space):
        """
        Initialize your PPO agent
        Args:
            observation_space: The observation space of the environment
            action_space: The action space (Discrete(5) for SimpleTag)
        """
        # Load your model, networks, etc.
        pass
    
    def act(self, observation, agent_id):
        """
        Select an action given an observation
        Args:
            observation: numpy array of the current observation
            agent_id: string identifier of the agent
        Returns:
            action: integer from 0 to 4
        """
        # Return discrete action
        pass
```

### Submission Process

Since you are not a collaborator on this repository, you will need to fork it, create a branch for your team, and then open a pull request to submit your work.

1. **Fork the Repository**: Click the 'Fork' button at the top-right of the GitHub page to create your own copy of this repository.

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SimpleTagCompetition.git
   cd SimpleTagCompetition
   ```

3. **Create a branch** for your team in your forked repository:
   ```bash
   git checkout -b team/your_team_name
   ```

4. **Add your files** to `submissions/your_team_name/`

3. **Create a Pull Request**:
   - Push your code to a new branch
   - Create a PR to the main branch
   - Automatic evaluation will run and comment results on your PR
   - **Size Limit**: Maximum 1 MB per submission (automatically enforced)

## Evaluation & Scoring

### Tournament Structure

The competition uses a **round-robin tournament** where all submitted agents compete against each other:

1. **Submission Management**: 
   - Submit via Pull Request to the main branch
   - Only your **latest submission** is kept (older submissions are automatically removed)
   - Each submission tagged with GitHub username and timestamp
   - Automatic validation runs local tests and full tournament

2. **Match Format**:
   - Each pair of teams plays in both roles (prey and predator)
   - Team A as prey vs Team B as predator
   - Team B as prey vs Team A as predator
   - Multiple episodes per matchup using fixed seeds

3. **Scoring**:
   - Individual scores: Average reward for prey role and predator role
   - Final ranking: Combined performance (average of both roles)
   - Baseline agents compete but don't appear in final rankings

4. **Fixed Seeds**: All matches use 50 predetermined random seeds from `seeds.json` for reproducibility

5. **Metrics**:
   - Average episode reward
   - Episode length (survival/capture time)
   - Combined performance across both roles

### Automated Evaluation

When you submit a PR, GitHub Actions automatically:
- Validates PR size (max 1 MB)
- Removes any previous submissions from the same GitHub user
- Creates `metadata.json` with your GitHub ID and timestamp
- Runs local tests against baseline opponents
- Executes full round-robin tournament with all current submissions
- Posts detailed results as a PR comment
- Commits accepted submission to main branch

### Leaderboard

The leaderboard shows:

- Overall ranking based on average score (prey + predator combined)
- Separate scores for prey and predator roles
- Number of matches played per team
- Your team is highlighted in tournament results
- Baseline agents compete but are excluded from final rankings

## Resources

### Environment Documentation
- [PettingZoo Simple Tag](https://pettingzoo.farama.org/environments/mpe/simple_tag/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)

### PPO Resources
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Rules & Restrictions

1. **Original Implementation**: You must implement PPO yourself. You can reference existing implementations but copying is not allowed.
2. **No Hardcoding**: Agents should not contain hardcoded policies specific to test seeds
3. **Submission Size**: Maximum 1 MB per submission (automatically enforced)
4. **One Submission Per User**: Only your latest submission is kept; older ones are automatically removed
5. **Fair Play**: No adversarial attacks on the evaluation system

## Grading Criteria

Your final grade will be based on:

1. **Performance (50%)**: Final ranking on the round-robin tournament leaderboard
2. **Implementation (50%)**: Code quality, PPO implementation correctness

## Timeline

- **Competition Start**: [26/11/2025]
- **Submission Deadline**: [04/01/2026]

## Reproducibility

All tournament matches use fixed seeds from `seeds.json` to ensure:

- Deterministic evaluation across all submissions
- Fair comparison between agents
- Reproducible results for debugging and validation
