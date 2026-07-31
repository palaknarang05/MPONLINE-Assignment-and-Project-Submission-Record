"""
evaluate.py

Evaluate a trained PPO agent on CartPole-v1.

Author: Palak Narang
"""

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO

from config import *
from utils import *


# ==========================================================
# Create Environment
# ==========================================================

def create_environment():

    env = gym.make(ENV_NAME)

    return env


# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    print("\nLoading trained model...")

    model = PPO.load(str(MODEL_PATH))

    print("Model loaded successfully.\n")

    return model


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate(model, env):

    rewards = []

    print(f"Running {EVAL_EPISODES} evaluation episodes...\n")

    for episode in range(EVAL_EPISODES):

        observation, _ = env.reset(seed=RANDOM_SEED + episode)

        done = False

        truncated = False

        episode_reward = 0

        while not (done or truncated):

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, done, truncated, info = env.step(action)

            episode_reward += reward

        rewards.append(episode_reward)

        print(
            f"Episode {episode + 1:03d} | Reward : {episode_reward:.2f}"
        )

    return rewards


# ==========================================================
# Generate Report
# ==========================================================

def generate_report(rewards):

    stats = calculate_statistics(rewards)

    print_statistics(stats)

    save_report(

        stats,

        EVALUATION_REPORT

    )

    print("\nEvaluation report saved.")

    return stats


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)

    print("PPO CARTPOLE EVALUATION")

    print("=" * 60)

    env = create_environment()

    model = load_model()

    rewards = evaluate(model, env)

    generate_report(rewards)

    env.close()

    print("\nEvaluation completed successfully.")

    print("=" * 60)


if __name__ == "__main__":

    main()