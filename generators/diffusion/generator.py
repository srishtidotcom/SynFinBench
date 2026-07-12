import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import DiffusionModel
from scheduler import DDPMScheduler


class FinancialDiffusionGenerator:
    def __init__(
        self,
        window_size: int = 30,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 500,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ):
        self.window_size = window_size
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DiffusionModel(input_dim=window_size).to(self.device)
        self.scheduler = DDPMScheduler(num_timesteps, beta_start, beta_end)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        self.data_mean: float | None = None
        self.data_std: float | None = None
        self.loss_history: list[float] = []

    def fit(
        self,
        train_windows: np.ndarray,
        verbose: bool = True,
    ) -> list[float]:
        self.data_mean = float(train_windows.mean())
        self.data_std = float(train_windows.std())

        tensor_data = torch.tensor(train_windows, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        if verbose:
            print(f"Data shape: {train_windows.shape}")
            print(
                f"Mean: {self.data_mean:.6f}, Std: {self.data_std:.6f}"
            )
            print(f"Device: {self.device}")
            print(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        best_loss = float("inf")
        self.loss_history = []

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = dataloader
            if verbose:
                pbar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch+1:3d}/{self.num_epochs}",
                    leave=False,
                )

            for batch in pbar:
                x0 = batch[0].to(self.device)
                B = x0.size(0)

                t = torch.randint(
                    0, self.num_timesteps, (B,), device=self.device
                )
                noise = torch.randn_like(x0)
                x_t = self.scheduler.add_noise(x0, noise, t)

                noise_pred = self.model(x_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                if verbose:
                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose:
                marker = " (best)" if avg_loss == best_loss else ""
                print(f"  Epoch {epoch+1:3d} — Loss: {avg_loss:.6f}{marker}")

        if verbose:
            print(f"Training complete. Best loss: {best_loss:.6f}")

        return self.loss_history

    def generate(
        self,
        num_samples: int,
        batch_size: int = 256,
        verbose: bool = False,
    ) -> np.ndarray:
        self.model.eval()
        synthetic_windows = []

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                B = end - start

                x_t = torch.randn(B, self.window_size, device=self.device)

                for t in reversed(range(self.num_timesteps)):
                    t_tensor = torch.full(
                        (B,), t, device=self.device, dtype=torch.long
                    )
                    noise_pred = self.model(x_t, t_tensor)
                    noise = torch.randn_like(x_t) if t > 0 else None
                    x_t = self.scheduler.sample_timestep(
                        x_t, noise_pred, t_tensor, noise
                    )

                synthetic_windows.append(x_t.cpu().numpy())

        result = np.concatenate(synthetic_windows, axis=0)

        if verbose:
            print(f"Generated {result.shape[0]} windows, shape {result.shape}")

        return result

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_dir / "model.pt")

        meta = {
            "window_size": self.window_size,
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "loss_history": self.loss_history,
        }
        torch.save(meta, output_dir / "meta.pt")

        csv_path = output_dir / "loss_history.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(self.loss_history, 1):
                writer.writerow([i, loss])

    def load(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        ckpt_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent

        meta_path = ckpt_dir / "meta.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=self.device, weights_only=False)
            self.window_size = meta.get("window_size", self.window_size)
            self.num_timesteps = meta.get("num_timesteps", self.num_timesteps)
            self.beta_start = meta.get("beta_start", self.beta_start)
            self.beta_end = meta.get("beta_end", self.beta_end)
            self.data_mean = meta.get("data_mean", self.data_mean)
            self.data_std = meta.get("data_std", self.data_std)
            self.loss_history = meta.get("loss_history", [])

        model_path = checkpoint_path if checkpoint_path.suffix == ".pt" else ckpt_dir / "model.pt"
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = DiffusionModel(input_dim=self.window_size).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.scheduler = DDPMScheduler(
            self.num_timesteps, self.beta_start, self.beta_end
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
