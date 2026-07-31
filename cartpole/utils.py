"""
Utility Functions for PPO CartPole Project

Author: Palak Narang
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ===========================
# DIRECTORY UTILITIES
# ===========================

def create_directory(directory):
    """
    Creates a directory if it doesn't already exist.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


# ===========================
# REWARD STATISTICS
# ===========================

def calculate_statistics(rewards):
    """
    Returns useful reward statistics.
    """

    rewards = np.array(rewards)

    return {
        "episodes": len(rewards),
        "average_reward": float(np.mean(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "minimum_reward": float(np.min(rewards)),
        "median_reward": float(np.median(rewards)),
        "standard_deviation": float(np.std(rewards))
    }


# ===========================
# SAVE REWARD HISTORY
# ===========================

def save_reward_history(rewards, filename):
    """
    Save reward history into CSV.
    """

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["Episode", "Reward"])

        for index, reward in enumerate(rewards, start=1):

            writer.writerow([index, reward])


# ===========================
# LOAD REWARD HISTORY
# ===========================

def load_reward_history(filename):

    rewards = []

    with open(filename, "r") as file:

        reader = csv.reader(file)

        next(reader)

        for row in reader:

            rewards.append(float(row[1]))

    return rewards


# ===========================
# SAVE REPORT
# ===========================

def save_report(statistics, filename):

    with open(filename, "w") as file:

        file.write("=================================\n")
        file.write(" PPO CARTPOLE EVALUATION REPORT\n")
        file.write("=================================\n\n")

        for key, value in statistics.items():

            pretty_key = key.replace("_", " ").title()

            file.write(f"{pretty_key}: {value}\n")


# ===========================
# PLOT REWARD CURVE
# ===========================

def plot_rewards(rewards, save_path):

    plt.figure(figsize=(10, 6))

    plt.plot(rewards, linewidth=2)

    plt.title("Training Reward Curve")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ===========================
# MOVING AVERAGE
# ===========================

def moving_average(values, window=20):

    if len(values) < window:

        return values

    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, mode="valid")


# ===========================
# PLOT MOVING AVERAGE
# ===========================

def plot_smoothed_rewards(rewards, save_path):

    smooth = moving_average(rewards)

    plt.figure(figsize=(10, 6))

    plt.plot(rewards, alpha=0.3, label="Raw Rewards")

    plt.plot(
        range(len(smooth)),
        smooth,
        linewidth=3,
        label="Moving Average"
    )

    plt.legend()

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Smoothed Reward Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ===========================
# PRINT STATISTICS
# ===========================

def print_statistics(statistics):

    print("\nEvaluation Statistics\n")

    print("-" * 40)

    for key, value in statistics.items():

        print(f"{key:<25} : {value}")

    print("-" * 40)


# ===========================
# FORMAT TIME
# ===========================

def seconds_to_hms(seconds):

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"