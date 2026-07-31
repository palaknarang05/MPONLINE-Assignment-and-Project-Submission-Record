# CartPole RL Agent Training

Reinforcement learning agent trained with **Proximal Policy Optimization (PPO)** to solve the `CartPole-v1` environment (Gymnasium), balancing a pole on a moving cart.

## Developer Info
**Name:** Palak Narang
**Registration Number:** 23BCE11819

## Tech Stack
Python, Gymnasium, Stable-Baselines3 (PPO), PyTorch, TensorBoard

## Project Structure
```
cartpole/
├── config.py       # Environment + PPO hyperparameters
├── utils.py        # Helper functions
├── train.py        # Trains PPO agent, saves checkpoints + best model
├── evaluate.py      # Runs evaluation episodes, generates report
├── test.py          # Quick sanity test of trained policy
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Train:
```bash
python train.py
```
Evaluate:
```bash
python evaluate.py
```
Test:
```bash
python test.py
```

## Approach
- Environment: `CartPole-v1`
- Algorithm: PPO (`stable-baselines3`)
- Reward, learning curve, and checkpoint logging via TensorBoard and CSV history
- Best model selected via `EvalCallback` during training

## Output
- Trained model saved to `models/`
- Learning curve plot saved to `plots/`
- Evaluation report at `evaluation_report.txt`
