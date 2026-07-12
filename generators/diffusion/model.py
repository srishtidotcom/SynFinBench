import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embedding used in Transformer / DDPM architectures.

    Maps a scalar timestep tensor (B,) to a sinusoidal embedding (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class MLPDenoiser(nn.Module):
    """Lightweight MLP that predicts noise given noisy input and timestep.

    Concatenates sinusoidal timestep embedding with the noisy input
    and passes the result through a stack of hidden layers with GELU activations.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        in_dim = input_dim + hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, t_emb], dim=1)
        return self.net(h)


class DiffusionModel(nn.Module):
    """DDPM denoising model wrapping the MLPDenoiser.

    Forward pass:  noise_pred = model(x_t, t)
        x_t: (B, input_dim)  — noisy window
        t:   (B,)            — timestep indices
    Returns: (B, input_dim) — predicted Gaussian noise
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.denoiser = MLPDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t, t)
