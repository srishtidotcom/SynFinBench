"""
Standalone script to run the full GAN training in the background,
so it survives closing VS Code / notebook restarts.

UPDATE: increased NOISE_DIM (3->8), reduced N_CRITIC (3->2), and added
a per-epoch diversity check to monitor mode collapse directly during
training instead of only finding out at the end.

Usage (from inside generators/gan/):
    nohup python3 run_training.py > training_log.txt 2>&1 &

Then check progress anytime with:
    tail -f training_log.txt
"""

import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader

from model import Generator, Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

NOISE_DIM = 8          # was 3 -- more room for the generator to vary output
SEQ_LEN = 30
BATCH_SIZE = 64
N_CRITIC = 2            # was 3 -- give the generator more relative training time
GP_LAMBDA = 10.0
LR = 1e-4
EPOCHS = 150            # a bit longer than last time since this is the "real" tuned run

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = np.load("../../data/processed/train_windows.npy")
train = train[..., None].astype(np.float32)
train_tensor = torch.from_numpy(train)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

print(f"Loaded train data: {train_tensor.shape}", flush=True)


def gradient_penalty(D, real, fake):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, device=DEVICE)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = D(interpolated)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def check_diversity(G, n_samples=20):
    """
    Generates n_samples fake sequences and reports the average per-timestep
    stddev across them. A LOW number here (close to 0) is the mode-collapse
    warning sign -- it means the generator is producing near-identical
    outputs regardless of the input noise.
    """
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, SEQ_LEN, NOISE_DIM, device=DEVICE)
        samples = G(z)  # (n_samples, seq_len, 1)
        diversity = samples.std(dim=0).mean().item()
    G.train()
    return diversity


def main():
    G = Generator(noise_dim=NOISE_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

    os.makedirs("outputs", exist_ok=True)

    real_diversity = train_tensor.std(dim=0).mean().item()
    print(f"Real data diversity (reference): {real_diversity:.4f}", flush=True)

    for epoch in range(EPOCHS):
        for (real_batch,) in train_loader:
            real_batch = real_batch.to(DEVICE)
            bsz = real_batch.size(0)

            for _ in range(N_CRITIC):
                z = torch.randn(bsz, SEQ_LEN, NOISE_DIM, device=DEVICE)
                fake_batch = G(z).detach()

                d_real = D(real_batch)
                d_fake = D(fake_batch)
                gp = gradient_penalty(D, real_batch, fake_batch)

                d_loss = d_fake.mean() - d_real.mean() + GP_LAMBDA * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z = torch.randn(bsz, SEQ_LEN, NOISE_DIM, device=DEVICE)
            fake_batch = G(z)
            g_loss = -D(fake_batch).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        diversity = check_diversity(G)
        print(f"epoch {epoch}  d_loss={d_loss.item():.4f}  g_loss={g_loss.item():.4f}  "
              f"gen_diversity={diversity:.4f}  (real={real_diversity:.4f})", flush=True)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(G.state_dict(), f"outputs/generator_epoch{epoch}.pt")
            torch.save(D.state_dict(), f"outputs/discriminator_epoch{epoch}.pt")
            print(f"  -> checkpoint saved at epoch {epoch}", flush=True)

    torch.save(G.state_dict(), "outputs/generator_final.pt")
    torch.save(D.state_dict(), "outputs/discriminator_final.pt")
    print("Training complete. Final model saved.", flush=True)


if __name__ == "__main__":
    main()