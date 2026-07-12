import csv
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_checkpoint(model: torch.nn.Module, path: Path):
    """Save model state dict to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device):
    """Load model state dict from a checkpoint file."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def set_seed(seed: int):
    """Set random seed for reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_loss_curve(loss_history: list[float], path: Path):
    """Plot and save the training loss curve as a PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, loss in enumerate(loss_history, 1):
            writer.writerow([i, loss])


def save_generated_data(
    synthetic_windows: np.ndarray,
    synthetic_returns: np.ndarray,
    output_dir: Path,
):
    """Save generated synthetic windows and denormalized returns to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "synthetic_windows.npy", synthetic_windows)

    np.savetxt(
        output_dir / "synthetic_returns.csv",
        synthetic_returns,
        delimiter=",",
        header=",".join([f"t+{i+1}" for i in range(synthetic_returns.shape[1])]),
        comments="",
    )
