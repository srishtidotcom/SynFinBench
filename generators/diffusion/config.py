from pathlib import Path

import torch


WINDOW_SIZE = 30
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 500
GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

DATA_PATH = Path("../data/processed/train_windows.npy")
