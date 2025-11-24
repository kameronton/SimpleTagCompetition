# Example Team Submission

This directory contains an example submission structure.

## Files

- `agent.py` - Your PPOAgent implementation
- `model_weights_prey.pth` - Your trained model weights for prey role
- `model_weights_predator.pth` - Your trained model weights for predator role
- `metadata.json` - Submission metadata (required)
- `README.md` - This file

## Instructions

1. Train your PPO agent using `examples/train_ppo.py` or your own implementation
2. Copy your trained model weights here
3. Modify `agent.py` to load your specific model architecture
4. **Create `metadata.json`** (only needed for local testing - auto-generated in PRs):
   ```json
   {
     "github_id": "your-github-username",
     "timestamp": "2025-11-24T14:30:00",
     "description": "Optional description of this submission"
   }
   ```
5. Test locally: `python run_tournament.py --local-test --team example_team`
6. Submit via pull request (metadata.json will be auto-generated from PR info)

## Metadata Format

When submitting via PR, `metadata.json` is **automatically created** from:
- `github_id`: Your GitHub username (from PR author)
- `timestamp`: PR submission time (UTC)
- `pr_number`: Pull request number
- `commit_sha`: Commit hash

For local testing only, manually create `metadata.json` with:
- `github_id` (required): Your GitHub username
- `timestamp` (required): ISO format timestamp (YYYY-MM-DDTHH:MM:SS)
- `description` (optional): Brief description of your submission

**Important**: Only the latest submission per `github_id` will be included in tournaments.

## Notes

- Make sure your agent can load without external dependencies beyond what's in `requirements.txt`
- The `act()` method will be called for each action selection
- Keep model size under 100 MB
