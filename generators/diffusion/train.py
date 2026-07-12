"""Train a DDPM noise-prediction model on financial time-series windows.

Usage:
    # From repository root:
    python -m generators.diffusion.train

    # From generators/diffusion/:
    python train.py

Output (written to generators/diffusion/outputs/):
    best_model.pt    — checkpoint with lowest validation loss
    final_model.pt   — checkpoint after the final epoch
    loss.csv         — epoch-wise training loss history
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure the project root is on sys.path so package imports work
# regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from generators.diffusion.model import DiffusionModel
from generators.diffusion.scheduler import DDPMScheduler

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 500
NUM_TIMESTEPS: int = 1000
BETA_START: float = 1e-4
BETA_END: float = 0.02
WINDOW_SIZE: int = 30
DATA_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_windows.npy"
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_windows = np.load(DATA_PATH)
print(f"Loaded data shape: {train_windows.shape}")

data_mean = train_windows.mean()
data_std = train_windows.std()
print(f"Data mean: {data_mean:.6f}, std: {data_std:.6f}")

tensor_data = torch.tensor(train_windows, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Number of samples: {len(dataset)}")
print(f"Batches per epoch: {len(dataloader)}")

# ---------------------------------------------------------------------------
# Model, scheduler, optimizer
# ---------------------------------------------------------------------------
model = DiffusionModel().to(device)
scheduler = DDPMScheduler(NUM_TIMESTEPS, BETA_START, BETA_END)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_loss = float("inf")
loss_history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d}/{NUM_EPOCHS}", leave=False)

    for batch in pbar:
        # (a) Move batch to device
        x0 = batch[0].to(device)  # (B, 30)

        # (b) Sample random timestep for each sample in the batch
        B = x0.size(0)
        t = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)  # (B,)

        # (c) Generate Gaussian noise
        noise = torch.randn_like(x0)  # (B, 30)

        # (d) Create noisy samples via the forward diffusion process
        x_t = scheduler.add_noise(x0, noise, t)  # (B, 30)

        # (e) Predict the noise added at timestep t
        noise_pred = model(x_t, t)  # (B, 30)

        # (f) DDPM objective: MSE between predicted and true noise
        loss = nn.functional.mse_loss(noise_pred, noise)

        # (g) Zero gradients
        optimizer.zero_grad()

        # (h) Backpropagate
        loss.backward()

        # (i) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # (j) Optimizer step
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        print(f"  Epoch {epoch+1:3d} — Loss: {avg_loss:.6f}  (best, saved)")
    else:
        print(f"  Epoch {epoch+1:3d} — Loss: {avg_loss:.6f}")

# ---------------------------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
print(f"\nTraining complete. Best loss: {best_loss:.6f}")

# ---------------------------------------------------------------------------
# Save loss history
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss"])
    for i, loss in enumerate(loss_history, 1):
        writer.writerow([i, loss])

print(f"Loss history saved to {OUTPUT_DIR / 'loss.csv'}")
