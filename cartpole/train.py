"""
train.py

Train a PPO agent on the CartPole-v1 environment.

Author: Palak Narang
"""

import time
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from config import *
from utils import *

warnings.filterwarnings("ignore")


# ==========================================================
# Set Random Seeds
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


# ==========================================================
# Reward Callback
# ==========================================================

class RewardTrackerCallback(BaseCallback):

    def __init__(self, verbose=0):

        super().__init__(verbose)

        self.episode_rewards = []

        self.current_reward = 0


    def _on_step(self):

        reward = self.locals["rewards"][0]

        done = self.locals["dones"][0]

        self.current_reward += reward

        if done:

            self.episode_rewards.append(self.current_reward)

            self.current_reward = 0

        return True


# ==========================================================
# Create Environment
# ==========================================================

def create_environment():

    env = gym.make(ENV_NAME)

    env = Monitor(env)

    return env


# ==========================================================
# Create PPO Model
# ==========================================================

def build_model(environment):

    model = PPO(

        policy="MlpPolicy",

        env=environment,

        learning_rate=LEARNING_RATE,

        gamma=GAMMA,

        n_steps=N_STEPS,

        batch_size=BATCH_SIZE,

        n_epochs=N_EPOCHS,

        gae_lambda=GAE_LAMBDA,

        clip_range=CLIP_RANGE,

        ent_coef=ENT_COEF,

        vf_coef=VF_COEF,

        max_grad_norm=MAX_GRAD_NORM,

        tensorboard_log=str(TENSORBOARD_LOG),

        verbose=1,

        seed=RANDOM_SEED

    )

    return model

# ==========================================================
# Training Function
# ==========================================================

def train():

    print("=" * 60)
    print("PPO CARTPOLE TRAINING")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    create_directory(MODELS_DIR)
    create_directory(LOGS_DIR)
    create_directory(PLOTS_DIR)

    env = create_environment()

    eval_env = create_environment()

    model = build_model(env)

    reward_callback = RewardTrackerCallback()

    checkpoint_callback = CheckpointCallback(

        save_freq=5000,

        save_path=str(MODELS_DIR),

        name_prefix="ppo_checkpoint"

    )

    eval_callback = EvalCallback(

        eval_env,

        best_model_save_path=str(BEST_MODEL_PATH),

        log_path=str(LOGS_DIR),

        eval_freq=5000,

        deterministic=True,

        render=False

    )

    callbacks = [

        reward_callback,

        checkpoint_callback,

        eval_callback

    ]

    print("\nTraining Started...\n")

    start_time = time.time()

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        callback=callbacks,

        progress_bar=True

    )

    end_time = time.time()

    elapsed = end_time - start_time

    print("\nTraining Finished.")

    print(f"Training Time : {seconds_to_hms(elapsed)}")

    model.save(str(MODEL_PATH))

    print("\nModel Saved Successfully.")

    rewards = reward_callback.episode_rewards

    if len(rewards) == 0:

        print("No rewards collected.")

        return

    save_reward_history(

        rewards,

        REWARD_HISTORY

    )

    plot_rewards(

        rewards,

        LEARNING_CURVE

    )

    plot_smoothed_rewards(

        rewards,

        PLOTS_DIR / "smoothed_learning_curve.png"

    )

    stats = calculate_statistics(rewards)

    print_statistics(stats)

    save_report(

        stats,

        LOGS_DIR / "training_statistics.txt"

    )

    print("\nLearning Curve Saved.")

    print("Reward History Saved.")

    env.close()

    eval_env.close()

    return model

# ==========================================================
# MAIN FUNCTION
# ==========================================================

def print_configuration():

    print("\nConfiguration")

    print("-" * 40)

    print(f"Environment        : {ENV_NAME}")
    print(f"Algorithm          : PPO")
    print(f"Total Timesteps    : {TOTAL_TIMESTEPS}")
    print(f"Learning Rate      : {LEARNING_RATE}")
    print(f"Gamma              : {GAMMA}")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Epochs             : {N_EPOCHS}")
    print(f"Random Seed        : {RANDOM_SEED}")

    print("-" * 40)


def print_project_structure():

    print("\nOutput Files")

    print("-" * 40)

    print(f"Model Directory    : {MODELS_DIR}")
    print(f"Logs Directory     : {LOGS_DIR}")
    print(f"Plots Directory    : {PLOTS_DIR}")

    print("-" * 40)


def training_summary():

    print("\nTraining Completed Successfully")

    print("=" * 60)

    print("Generated Files")

    print("-------------------------------")

    print(f"Model               : {MODEL_PATH}")

    print(f"Reward CSV          : {REWARD_HISTORY}")

    print(f"Learning Curve      : {LEARNING_CURVE}")

    print(f"Training Statistics : {LOGS_DIR / 'training_statistics.txt'}")

    print("\nNext Step")

    print("Run evaluate.py to evaluate the trained agent.")

    print("=" * 60)


def main():

    print("\n")

    print("=" * 60)

    print("PPO CARTPOLE REINFORCEMENT LEARNING")

    print("=" * 60)

    print_configuration()

    print_project_structure()

    train()

    training_summary()


# ==========================================================
# PROGRAM ENTRY
# ==========================================================

if __name__ == "__main__":

    main()