"""Generate synthetic financial windows using a trained DDPM model.

Usage:
    # From repository root:
    python -m generators.diffusion.sample

    # From generators/diffusion/:
    python sample.py

Requires a trained checkpoint at generators/diffusion/outputs/best_model.pt
(from a completed run of train.py).

Output (written to generators/diffusion/outputs/):
    synthetic_windows.npy   — generated windows (num_samples, 30) in normalized space
    synthetic_returns.csv   — denormalized log-return windows
"""

import sys
from pathlib import Path

import numpy as np
import torch
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
NUM_TIMESTEPS: int = 1000
BETA_START: float = 1e-4
BETA_END: float = 0.02
WINDOW_SIZE: int = 30
BATCH_SIZE: int = 256
DATA_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_windows.npy"
SCRIPT_DIR: Path = Path(__file__).resolve().parent
CHECKPOINT_PATH: Path = SCRIPT_DIR / "outputs" / "best_model.pt"
OUTPUT_DIR: Path = SCRIPT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Reproducibility (same seed as training)
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load training data for metadata (num_samples, mean, std)
# ---------------------------------------------------------------------------
train_windows = np.load(DATA_PATH)
num_samples = len(train_windows)
data_mean = train_windows.mean()
data_std = train_windows.std()
print(f"Data: {train_windows.shape}, mean={data_mean:.6f}, std={data_std:.6f}")

# ---------------------------------------------------------------------------
# Load model and scheduler
# ---------------------------------------------------------------------------
model = DiffusionModel().to(device)
state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

scheduler = DDPMScheduler(NUM_TIMESTEPS, BETA_START, BETA_END)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------
synthetic_windows = []

with torch.no_grad():
    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        B = end - start

        # Start from pure Gaussian noise
        x_t = torch.randn(B, WINDOW_SIZE, device=device)

        # Reverse diffusion: t = T-1 down to 0
        pbar = tqdm(
            reversed(range(NUM_TIMESTEPS)),
            desc=f"Sampling batch {start}:{end}",
            total=NUM_TIMESTEPS,
            leave=False,
        )
        for t in pbar:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict the noise at timestep t
            noise_pred = model(x_t, t_tensor)

            # Perform one reverse step: x_{t-1} from x_t
            noise = torch.randn_like(x_t) if t > 0 else None
            x_t = scheduler.sample_timestep(x_t, noise_pred, t_tensor, noise)

        synthetic_windows.append(x_t.cpu().numpy())

synthetic_windows = np.concatenate(synthetic_windows, axis=0)
print(f"Generated synthetic windows: {synthetic_windows.shape}")

# ---------------------------------------------------------------------------
# Denormalize: revert the z-score scaling applied during preprocessing
# ---------------------------------------------------------------------------
synthetic_returns = synthetic_windows * data_std + data_mean
print(
    f"Denormalized returns: mean={synthetic_returns.mean():.6f}, "
    f"std={synthetic_returns.std():.6f}"
)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
np.save(OUTPUT_DIR / "synthetic_windows.npy", synthetic_windows)
print(f"Saved {OUTPUT_DIR / 'synthetic_windows.npy'}")

np.savetxt(
    OUTPUT_DIR / "synthetic_returns.csv",
    synthetic_returns,
    delimiter=",",
    header=",".join([f"t+{i+1}" for i in range(WINDOW_SIZE)]),
    comments="",
)
print(f"Saved {OUTPUT_DIR / 'synthetic_returns.csv'}")
