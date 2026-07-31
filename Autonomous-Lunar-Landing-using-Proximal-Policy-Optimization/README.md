# Autonomous Lunar Landing using Proximal Policy Optimization (PPO)

## Developer Information

**Name:** Palak Narang  
**Registration Number:** 23BCE11819  
**Application Number:** IN26011657  
**Batch Number:** 1A  
**Email ID:** palaknarang05@gmail.com  

---

## Project Overview

This project implements a **Proximal Policy Optimization (PPO)** reinforcement learning agent to solve the **LunarLander-v3** environment from Gymnasium. The objective is to train an autonomous spacecraft to perform safe and controlled landings on a designated landing pad using deep reinforcement learning.

Instead of relying on manually designed control rules, the agent learns an optimal landing strategy through interaction with the environment by maximizing cumulative rewards.

---

## Environment Details

- **Environment:** LunarLander-v3
- **Framework:** Gymnasium
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Policy Network:** Multi-Layer Perceptron (MLP)

### Observation Space

The agent receives an 8-dimensional state vector consisting of:

- Horizontal Position
- Vertical Position
- Horizontal Velocity
- Vertical Velocity
- Lander Angle
- Angular Velocity
- Left Leg Contact
- Right Leg Contact

### Action Space

The agent chooses one of four discrete actions:

- Do Nothing
- Fire Left Orientation Engine
- Fire Main Engine
- Fire Right Orientation Engine

---

# Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- Pandas
- Matplotlib

---

# Project Structure

```
lunar-lander/
│
├── train.py
├── evaluate.py
├── test.py
├── record_video.py
├── plot_training.py
├── config.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── ppo_lunarlander.zip
│
├── logs/
│
├── graphs/
│   └── learning_curve.png
│
└── videos/
    └── lunar_lander.mp4
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/palaknarang05/MPONLINE-Assignment-and-Project-Submission-Record.git
```

Navigate to the project directory:

```bash
cd MPONLINE-Assignment-and-Project-Submission-Record/lunar-lander
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Learning Rate | 3e-4 |
| Discount Factor | 0.99 |
| Batch Size | 64 |
| Rollout Steps | 2048 |
| Epochs | 10 |
| Clip Range | 0.2 |
| Policy | MlpPolicy |

---

# Running the Project

## Train the Agent

```bash
python train.py
```

This trains the PPO agent and saves the trained model.

---

## Evaluate the Agent

```bash
python evaluate.py
```

Generates evaluation metrics such as average reward over multiple episodes.

---

## Test the Agent

```bash
python test.py
```

Runs the trained agent with live rendering.

---

## Record Gameplay

```bash
python record_video.py
```

Creates an MP4 recording of the trained agent.

---

## Plot Learning Curve

```bash
python plot_training.py
```

Generates the training reward graph.

---

# Model Performance

Typical performance after training:

- Mean Evaluation Reward: **170+**
- Stable Landing Success
- Smooth convergence during training
- Successful autonomous landings in most evaluation episodes

---

# Results

The PPO agent successfully learns to control the spacecraft by balancing exploration and exploitation during training.

The trained policy demonstrates:

- Stable landings
- Reduced crash frequency
- Higher cumulative rewards
- Efficient control of orientation and thrust

---

# Future Improvements

- Hyperparameter tuning
- Curriculum learning
- Parallel environment training
- Continuous action environments
- Custom reward shaping

---

# Requirements

Main dependencies:

- stable-baselines3
- gymnasium
- torch
- numpy
- pandas
- matplotlib

Install using:

```bash
pip install -r requirements.txt
```

---

# References

- Gymnasium Documentation: https://gymnasium.farama.org/
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- PPO Paper: https://arxiv.org/abs/1707.06347

---

## Author

**Palak Narang**  
B.Tech Computer Science and Engineering  
VIT Bhopal University
