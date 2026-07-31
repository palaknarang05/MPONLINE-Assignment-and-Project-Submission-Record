"""
test.py

Visualize the trained PPO agent playing CartPole-v1.

Author: Palak Narang
"""

import time
import gymnasium as gym

from stable_baselines3 import PPO

from config import *


# ==========================================================
# Create Environment
# ==========================================================

def create_environment():

    env = gym.make(

        ENV_NAME,

        render_mode="human"

    )

    return env


# ==========================================================
# Load Model
# ==========================================================

def load_model():

    print("\nLoading trained model...\n")

    model = PPO.load(str(MODEL_PATH))

    print("Model loaded successfully.\n")

    return model


# ==========================================================
# Run Demonstration
# ==========================================================

def run_demo(model, env, episodes=5):

    for episode in range(episodes):

        observation, _ = env.reset(seed=RANDOM_SEED + episode)

        done = False

        truncated = False

        total_reward = 0

        print("=" * 50)

        print(f"Episode {episode + 1}")

        print("=" * 50)

        while not (done or truncated):

            action, _ = model.predict(

                observation,

                deterministic=True

            )

            observation, reward, done, truncated, info = env.step(action)

            total_reward += reward

            time.sleep(0.01)

        print(f"Reward : {total_reward}")

    env.close()


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)

    print("PPO CARTPOLE VISUAL TEST")

    print("=" * 60)

    env = create_environment()

    model = load_model()

    run_demo(

        model,

        env,

        episodes=5

    )

    print("\nTesting completed successfully.")


if __name__ == "__main__":

    main()