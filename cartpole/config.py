"""
Configuration file for CartPole PPO Project
Author: Palak Narang
"""

from pathlib import Path

# ============================
# Project Directories
# ============================

PROJECT_ROOT = Path(__file__).parent

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ============================
# Environment Configuration
# ============================

ENV_NAME = "CartPole-v1"

# ============================
# PPO Hyperparameters
# ============================

TOTAL_TIMESTEPS = 50000

LEARNING_RATE = 3e-4

GAMMA = 0.99

N_STEPS = 2048

BATCH_SIZE = 64

N_EPOCHS = 10

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

ENT_COEF = 0.0

VF_COEF = 0.5

MAX_GRAD_NORM = 0.5

# ============================
# Evaluation
# ============================

EVAL_EPISODES = 100

RANDOM_SEED = 42

# ============================
# Output Files
# ============================

MODEL_PATH = MODELS_DIR / "ppo_cartpole"

BEST_MODEL_PATH = MODELS_DIR / "best_model"

LEARNING_CURVE = PLOTS_DIR / "learning_curve.png"

EVALUATION_REPORT = PROJECT_ROOT / "evaluation_report.txt"

REWARD_HISTORY = LOGS_DIR / "reward_history.csv"

# ============================
# TensorBoard
# ============================

TENSORBOARD_LOG = LOGS_DIR / "tensorboard"

# ============================
# Plot Settings
# ============================

FIGURE_SIZE = (10, 6)

LINE_WIDTH = 2

GRID = True