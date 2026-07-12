"""DDPM forward/reverse process scheduler.

Provides:
    - Linear beta schedule
    - Forward diffusion:  x_t = add_noise(x0, noise, t)
    - Reverse step:       x_{t-1} = sample_timestep(x_t, noise_pred, t)
    - x0 estimation:      x0_pred = predict_x0(x_t, noise_pred, t)
    - Posterior:          mean, var = q_posterior_mean_variance(x0, x_t, t)
"""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from generators.diffusion.config import NUM_TIMESTEPS, BETA_START, BETA_END


def _linear_beta_schedule(
    num_timesteps: int = NUM_TIMESTEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end over num_timesteps."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


class DDPMScheduler:
    def __init__(
        self,
        num_timesteps: int = NUM_TIMESTEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
    ):
        self.num_timesteps = num_timesteps

        # (num_timesteps,) — noise variance at each timestep
        betas = _linear_beta_schedule(num_timesteps, beta_start, beta_end)
        self.betas = betas

        # (num_timesteps,) — 1 - beta_t
        alphas = 1.0 - betas
        self.alphas = alphas

        # (num_timesteps,) — cumulative product of alphas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.alpha_bars = alpha_bars

    def add_noise(
        self,
        x0: torch.Tensor,      # (B, 30) — clean data
        noise: torch.Tensor,   # (B, 30) — sampled Gaussian noise
        t: torch.Tensor,       # (B,)    — timestep indices (0 <= t < num_timesteps)
    ) -> torch.Tensor:         # (B, 30) — noisy data at timestep t
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(1)      # (B, 1)
        sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bars[t]).sqrt().unsqueeze(1)  # (B, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def predict_x0(
        self,
        x_t: torch.Tensor,      # (B, 30) — noisy data at timestep t
        noise_pred: torch.Tensor,  # (B, 30) — predicted noise
        t: torch.Tensor,        # (B,)    — timestep indices
    ) -> torch.Tensor:          # (B, 30) — estimated clean data
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(1)                    # (B, 1)
        sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bars[t]).sqrt().unsqueeze(1)  # (B, 1)
        return (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,      # (B, 30) — estimated clean data
        x_t: torch.Tensor,     # (B, 30) — noisy data at timestep t
        t: torch.Tensor,       # (B,)    — timestep indices
    ):
        """Compute mean and variance of q(x_{t-1} | x_t, x0)."""
        # Gather coefficients indexed by t, shape (B,) then unsqueeze to (B, 1)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)              # (B, 1)

        # ᾱ_{t-1}: at t=0 (last step) this should be 1 (empty product),
        # matching the standard DDPM convention ᾱ_0 = 1.
        alpha_bar_t_minus_1 = torch.where(
            t.unsqueeze(1) > 0,
            self.alpha_bars[(t - 1).clamp(min=0)].unsqueeze(1),
            torch.ones_like(alpha_bar_t),
        )

        beta_t = self.betas[t].unsqueeze(1)                        # (B, 1)
        alpha_t = self.alphas[t].unsqueeze(1)                      # (B, 1)

        # Posterior mean: mu = (sqrt(alpha_bar_{t-1}) * beta_t) / (1 - alpha_bar_t) * x0
        #                + (sqrt(alpha_t) * (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t) * x_t
        coef_x0 = (alpha_bar_t_minus_1.sqrt() * beta_t) / (1.0 - alpha_bar_t)
        coef_xt = (alpha_t.sqrt() * (1.0 - alpha_bar_t_minus_1)) / (1.0 - alpha_bar_t)
        posterior_mean = coef_x0 * x0 + coef_xt * x_t              # (B, 30)

        # Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t)  # (B, 1)

        return posterior_mean, posterior_variance

    def sample_timestep(
        self,
        x_t: torch.Tensor,      # (B, 30) — noisy data at timestep t
        noise_pred: torch.Tensor,  # (B, 30) — predicted noise
        t: torch.Tensor,        # (B,)    — timestep indices
        noise: torch.Tensor | None = None,  # (B, 30) — random noise for x_{t-1} sampling
    ) -> torch.Tensor:          # (B, 30) — x_{t-1}
        """Single reverse denoising step: p(x_{t-1} | x_t)."""
        if noise is None:
            noise = torch.randn_like(x_t)

        x0_pred = self.predict_x0(x_t, noise_pred, t)              # (B, 30)
        posterior_mean, posterior_variance = self.q_posterior_mean_variance(x0_pred, x_t, t)

        # At t=0 there is no noise added
        mask = (t == 0).float().unsqueeze(1)                       # (B, 1)
        return posterior_mean + (1.0 - mask) * posterior_variance.sqrt() * noise
