"""Evaluate the diffusion generator after training and sampling.

Usage:
    # From repository root:
    python -m generators.diffusion.evaluate

Loads data/processed/train_windows.npy and
generators/diffusion/outputs/synthetic_windows.npy.

Output (written to generators/diffusion/outputs/):
    evaluation.csv                — basic statistics
    summary.txt                   — benchmark report
    figures/histogram.png         — overlayed histogram
    figures/cdf_comparison.png    — ECDF comparison
    figures/qq_plot.png           — QQ plot
    figures/boxplot.png           — boxplots
    figures/acf_real.png          — real data autocorrelation
    figures/acf_synthetic.png     — synthetic data autocorrelation
    figures/correlation_heatmap_real.png
    figures/correlation_heatmap_synthetic.png
    figures/loss_curve.png        — training loss (if loss.csv exists)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
real_path = _PROJECT_ROOT / "data" / "processed" / "train_windows.npy"
synth_path = Path(__file__).resolve().parent / "outputs" / "synthetic_windows.npy"
out_dir = Path(__file__).resolve().parent / "outputs"
fig_dir = out_dir / "figures"

out_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
real = np.load(real_path)
synth = np.load(synth_path)

N_real, L_real = real.shape
N_synth, L_synth = synth.shape
WINDOW = L_real

# Flatten for distribution-level comparisons
real_flat = real.ravel()
synth_flat = synth.ravel()


# ===================================================================
# PART 1 — Basic Statistics
# ===================================================================
def _describe(name: str, arr: np.ndarray) -> dict:
    return {
        "dataset": name,
        "shape": str(arr.shape),
        "num_samples": arr.shape[0],
        "window_length": arr.shape[1],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "variance": float(arr.var(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "skewness": float(stats.skew(arr.ravel())),
        "kurtosis": float(stats.kurtosis(arr.ravel())),
    }


stats_real = _describe("real", real)
stats_synth = _describe("synthetic", synth)
stats_df = pd.DataFrame([stats_real, stats_synth])
stats_df.to_csv(out_dir / "evaluation.csv", index=False)

# ===================================================================
# PART 2 — Statistical Fidelity Metrics
# ===================================================================


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    # Subsample to avoid O(n²) memory blowup
    MAX_MMD = 5_000
    rng = np.random.default_rng(42)
    X = rng.choice(X.ravel().astype(np.float64), size=MAX_MMD, replace=False)[:, None]
    Y = rng.choice(Y.ravel().astype(np.float64), size=MAX_MMD, replace=False)[:, None]
    gamma = 1.0 / (2.0 * sigma ** 2)
    K_XX = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    K_YY = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    K_XY = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    n = X.shape[0]
    m = Y.shape[0]
    return float(
        K_XX.sum() / (n * n)
        + K_YY.sum() / (m * m)
        - 2.0 * K_XY.sum() / (n * m)
    )


wasserstein = float(stats.wasserstein_distance(real_flat, synth_flat))
ks_stat, ks_pval = stats.ks_2samp(real_flat, synth_flat)
mmd = _mmd_rbf(real_flat, synth_flat)
# Pearson correlation between the flattened arrays
pearson_r, pearson_p = stats.pearsonr(real_flat, synth_flat)
mad = float(np.abs(real_flat - synth_flat).mean())
rmse = float(np.sqrt(((real_flat - synth_flat) ** 2).mean()))

metrics = {
    "wasserstein_distance": wasserstein,
    "ks_statistic": float(ks_stat),
    "ks_pvalue": float(ks_pval),
    "mmd_rbf": mmd,
    "pearson_correlation": float(pearson_r),
    "pearson_pvalue": float(pearson_p),
    "mean_absolute_difference": mad,
    "rmse": rmse,
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(out_dir / "evaluation.csv", mode="a", header=True, index=False)

# ===================================================================
# PART 3 — Distribution Figures
# ===================================================================
sns.set_style("whitegrid")

# -- histogram.png --
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(real_flat, bins=80, alpha=0.6, density=True, label="Real", color="C0")
ax.hist(synth_flat, bins=80, alpha=0.6, density=True, label="Synthetic", color="C1")
ax.set_xlabel("Normalised Return")
ax.set_ylabel("Density")
ax.set_title("Real vs Synthetic Distribution")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "histogram.png", dpi=150)
plt.close(fig)

# -- cdf_comparison.png --
fig, ax = plt.subplots(figsize=(8, 5))
x_plot = np.sort(real_flat)
y_plot = np.arange(1, len(x_plot) + 1) / len(x_plot)
ax.plot(x_plot, y_plot, label="Real", color="C0")
x_plot_s = np.sort(synth_flat)
y_plot_s = np.arange(1, len(x_plot_s) + 1) / len(x_plot_s)
ax.plot(x_plot_s, y_plot_s, label="Synthetic", color="C1", linestyle="--")
ax.set_xlabel("Normalised Return")
ax.set_ylabel("ECDF")
ax.set_title("Empirical Cumulative Distribution Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "cdf_comparison.png", dpi=150)
plt.close(fig)

# -- qq_plot.png --
fig, ax = plt.subplots(figsize=(8, 5))
real_sorted = np.sort(real_flat)
synth_sorted = np.sort(synth_flat)
ax.scatter(real_sorted, synth_sorted, s=2, alpha=0.5, color="C2")
lo = min(real_sorted.min(), synth_sorted.min())
hi = max(real_sorted.max(), synth_sorted.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
ax.set_xlabel("Real Quantiles")
ax.set_ylabel("Synthetic Quantiles")
ax.set_title("QQ Plot")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "qq_plot.png", dpi=150)
plt.close(fig)

# -- boxplot.png --
fig, ax = plt.subplots(figsize=(8, 5))
bp_data = [real_flat, synth_flat]
ax.boxplot(bp_data, tick_labels=["Real", "Synthetic"], showmeans=True)
ax.set_ylabel("Normalised Return")
ax.set_title("Real vs Synthetic Boxplot")
fig.tight_layout()
fig.savefig(fig_dir / "boxplot.png", dpi=150)
plt.close(fig)

# ===================================================================
# PART 4 — Autocorrelation (manual, lags 1–30)
# ===================================================================


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    var = (x ** 2).sum()
    if var == 0:
        return 0.0
    acov = (x[: n - lag] * x[lag:]).sum()
    return float(acov / var)


def _mean_autocorr(windows: np.ndarray, max_lag: int) -> np.ndarray:
    lags = np.arange(1, max_lag + 1)
    acf = np.array([_autocorr(w, lag) for w in windows for lag in lags])
    acf = acf.reshape(len(windows), max_lag).mean(axis=0)
    return acf


max_lag = 30
acf_real_vals = _mean_autocorr(real, max_lag)
acf_synth_vals = _mean_autocorr(synth, max_lag)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, max_lag + 1), acf_real_vals, color="C0", width=0.6)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Mean Autocorrelation — Real Data")
fig.tight_layout()
fig.savefig(fig_dir / "acf_real.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, max_lag + 1), acf_synth_vals, color="C1", width=0.6)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Mean Autocorrelation — Synthetic Data")
fig.tight_layout()
fig.savefig(fig_dir / "acf_synthetic.png", dpi=150)
plt.close(fig)

# ===================================================================
# PART 5 — Correlation (30x30)
# ===================================================================
corr_real = np.corrcoef(real.T)
corr_synth = np.corrcoef(synth.T)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_real, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
            square=True, cbar_kws={"shrink": 0.75})
ax.set_title("Correlation Matrix — Real Data")
ax.set_xlabel("Lag")
ax.set_ylabel("Lag")
fig.tight_layout()
fig.savefig(fig_dir / "correlation_heatmap_real.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_synth, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
            square=True, cbar_kws={"shrink": 0.75})
ax.set_title("Correlation Matrix — Synthetic Data")
ax.set_xlabel("Lag")
ax.set_ylabel("Lag")
fig.tight_layout()
fig.savefig(fig_dir / "correlation_heatmap_synthetic.png", dpi=150)
plt.close(fig)

# ===================================================================
# PART 6 — Training Curve
# ===================================================================
loss_path = out_dir / "loss.csv"
if loss_path.exists():
    loss_df = pd.read_csv(loss_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_df["epoch"], loss_df["loss"], color="C2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

# ===================================================================
# PART 7 — Summary Report
# ===================================================================
mean_diff = stats_real["mean"] - stats_synth["mean"]
std_diff = stats_real["std"] - stats_synth["std"]
skew_diff = stats_real["skewness"] - stats_synth["skewness"]
kurt_diff = stats_real["kurtosis"] - stats_synth["kurtosis"]

# Interpretation logic
parts = []
if abs(mean_diff) < 0.1 * stats_real["std"]:
    parts.append("The generated data closely matches the real distribution in terms of first-order statistics")
else:
    parts.append("The generated data shows a shift in mean relative to the real distribution")

if abs(ks_stat) < 0.15:
    parts.append("Distribution overlap is good according to the KS statistic")
elif abs(ks_stat) < 0.3:
    parts.append("Distribution overlap is moderate according to the KS statistic")
else:
    parts.append("The KS statistic indicates notable distributional divergence")

if wasserstein < 0.5:
    parts.append("Wasserstein distance indicates low deviation")
elif wasserstein < 1.5:
    parts.append("Wasserstein distance indicates moderate deviation")
else:
    parts.append("Wasserstein distance indicates large deviation")

if abs(pearson_r) > 0.8:
    parts.append("Strong linear correspondence is observed between real and synthetic samples")
elif abs(pearson_r) > 0.5:
    parts.append("Moderate linear correspondence is observed between real and synthetic samples")
else:
    parts.append("Limited linear correspondence is observed between real and synthetic samples")

parts.append("Long-range dependence should be inspected using the autocorrelation plots")

interpretation = ". ".join(parts) + "."

lines = [
    "=" * 27,
    "DIFFUSION BENCHMARK REPORT",
    "=" * 27,
    "",
    f"Training Samples:      {N_real}",
    f"Synthetic Samples:     {N_synth}",
    f"Window Length:         {WINDOW}",
    "",
    f"Mean Difference:       {mean_diff:+.8f}",
    f"Std Difference:        {std_diff:+.8f}",
    "",
    f"KS Statistic:          {ks_stat:.6f}",
    f"Wasserstein Distance:  {wasserstein:.6f}",
    f"MMD:                   {mmd:.8e}",
    "",
    f"Skewness Difference:   {skew_diff:+.6f}",
    f"Kurtosis Difference:   {kurt_diff:+.6f}",
    "",
    f"RMSE:                  {rmse:.6f}",
    f"Pearson Correlation:   {pearson_r:.6f}",
    "",
    "---",
    "Interpretation:",
    interpretation,
    "",
]

with open(out_dir / "summary.txt", "w") as f:
    f.write("\n".join(lines))

# ===================================================================
# PART 8 — Console Output
# ===================================================================
files = [
    "evaluation.csv",
    "summary.txt",
    "loss_curve.png",
    "histogram.png",
    "cdf_comparison.png",
    "qq_plot.png",
    "boxplot.png",
    "acf_real.png",
    "acf_synthetic.png",
    "correlation_heatmap_real.png",
    "correlation_heatmap_synthetic.png",
]

print("=" * 33)
print("DIFFUSION EVALUATION COMPLETE")
print("=" * 33)
print()
print("Generated files:")
for f in files:
    print(f"  {f}")
