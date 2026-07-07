# SynFinBench

**SynFinBench** is a modular and reproducible benchmark for evaluating synthetic financial data generators.

## Project Goal

### Research Question

> **How can synthetic financial data generators be evaluated through a standardized, reproducible benchmark that measures statistical fidelity, robustness across market regimes, long-horizon stability, and downstream utility?**

Our goal is **not** to build a new synthetic data generator.

Our goal is **not** to identify a universally "best" model.

Instead, we aim to develop a **standardized benchmark** that future researchers can use to evaluate any synthetic financial data generator under the same experimental conditions.

---

# Core Contribution

The benchmark provides:

- Standard datasets
- Standard preprocessing pipeline
- Standard train/test splits
- Representative generator families
- Standard evaluation metrics
- Fixed downstream evaluation protocol

The benchmark is designed to be **model-agnostic**.

Any future synthetic data generator should be able to plug into the benchmark without modifying the evaluation pipeline.

---

# Benchmark Scope

## Representative Generator Families

Rather than benchmarking dozens of individual papers, SynFinBench benchmarks one representative implementation from each major methodological family.

| Family | Representative Model |
|---------|----------------------|
| Classical Statistical | ARIMA-GARCH + Residual Bootstrap |
| GAN | TimeGAN (or QuantGAN) |
| Diffusion | Financial Diffusion Model |
| Hybrid | GAN-Diffusion |

The objective is to compare **generator paradigms**, not individual implementations.

---

## Datasets

The benchmark evaluates generators across multiple financial markets.

| Dataset | Motivation |
|----------|------------|
| S&P 500 | Standard equity benchmark |
| Bitcoin (BTC/USD) | High volatility and heavy-tailed market |
| EUR/USD | Stable foreign exchange market |

Using multiple asset classes allows us to evaluate how well generators generalize beyond a single market.

---

# Standard Experimental Pipeline

Every generator follows the exact same workflow.

```
Real Dataset
      │
      ▼
Standard Preprocessing
      │
      ▼
Train Generator
      │
      ▼
Generate Synthetic Dataset
      │
      ▼
Evaluation
```

The only component that changes is the generator itself.

Everything else remains fixed.

---

# Standardized Preprocessing

Every dataset undergoes exactly the same preprocessing pipeline.

1. Download historical daily prices
2. Handle missing values
3. Compute daily log returns
4. Chronological train/test split
5. Normalize using **training statistics only**
6. Create fixed-length sliding windows
7. Save processed datasets

No generator is allowed to modify the preprocessing pipeline.

This guarantees fair comparison across all models.

---

# Why Save Sliding Windows?

Different generator families expect different input formats.

| Generator | Expected Input |
|-----------|----------------|
| ARIMA | Return series |
| GARCH | Return series |
| TimeGAN | Sliding windows |
| Diffusion | Sliding windows |
| Hybrid | Sliding windows |

If only the return series is saved, every deep learning model would need to recreate its own sliding windows.

This introduces unnecessary implementation differences and makes reproducibility more difficult.

Instead, the preprocessing pipeline saves **both**:

- Normalized return series
- Standardized sliding windows

Every generator therefore begins from exactly the same processed inputs.

The preprocessing stage produces:

```
train.csv
test.csv

train_windows.npy
test_windows.npy
```

These files become the canonical inputs for all experiments.

---

# Repository Structure

```
SynFinBench/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   │
│   ├── raw/
│   │   └── sp500.csv
│   │
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── train_windows.npy
│       └── test_windows.npy
│
├── notebooks/
│   ├── 01_download_data.ipynb
│   └── 02_preprocessing.ipynb
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocessing.py
│   │   └── windowing.py
│   │
│   └── utils/
│       └── seed.py
│
├── generators/
│   │
│   ├── classical/
│   │   ├── train.ipynb
│   │   ├── model.py
│   │   └── outputs/
│   │
│   ├── gan/
│   │   ├── train.ipynb
│   │   ├── model.py
│   │   └── outputs/
│   │
│   ├── diffusion/
│   │   ├── train.ipynb
│   │   ├── model.py
│   │   └── outputs/
│   │
│   └── hybrid/
│       ├── train.ipynb
│       ├── model.py
│       └── outputs/
│
└── reports/
```

---

# Team Workflow

Only one person is responsible for downloading the raw dataset.

```
Yahoo Finance
      │
      ▼
data/raw/sp500.csv
```

The preprocessing notebook is then executed once.

```
Raw Data
      │
      ▼
Log Returns
      │
      ▼
Train/Test Split
      │
      ▼
Normalization
      │
      ▼
Sliding Windows
      │
      ▼
Processed Dataset
```

The processed outputs are shared by everyone.

Each generator team then independently trains their assigned model using the exact same processed inputs.

```
Processed Dataset
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Classical          GAN
        │               │
        ▼               ▼
 Diffusion        Hybrid
```

No generator should modify the preprocessing pipeline.

---

# Generator Interface

Every generator should follow the same high-level workflow.

```
Processed Data
      │
      ▼
Train Model
      │
      ▼
Generate Synthetic Returns
      │
      ▼
Save Outputs
```

Each generator should save its generated data to:

```
generators/<generator_name>/outputs/
```

This standardized output structure allows the evaluation pipeline to compare generators without requiring generator-specific code.

---

# Evaluation Framework

The benchmark consists of four evaluation pillars.

## 1. Statistical Fidelity

Measures whether synthetic data reproduces the statistical properties of real financial markets.

Includes evaluation of:

- Return distributions
- Heavy tails
- Volatility clustering
- Autocorrelation
- Stylized facts
- Wasserstein Distance
- Kolmogorov-Smirnov Test
- Maximum Mean Discrepancy (MMD)
- QQ Plots
- ACF/PACF

---

## 2. Regime Robustness

Evaluates generators under different market conditions.

Examples include:

- Bull markets
- Bear markets
- Market crashes
- Recovery periods
- High-volatility regimes

---

## 3. Long-Horizon Stability

Evaluates whether generated statistical properties remain stable over long simulated trajectories.

Metrics include:

- Distribution drift
- Variance drift
- Volatility persistence
- Autocorrelation decay
- Stationarity
- Tail behavior

---

## 4. Downstream Utility

Train-on-Synthetic, Test-on-Real (TSTR) forecasting benchmark.

Pipeline:

```
Synthetic Data
      │
      ▼
Train Forecasting Model
      │
      ▼
Evaluate on Real Test Data
```

Metrics include:

- RMSE
- MAE
- Directional Accuracy

---

# Current Development Phase

The project is currently in **Phase 1**, focusing on the equity market (S&P 500).

Current objectives:

- Download S&P 500 historical data from Yahoo Finance.
- Build the standardized preprocessing pipeline.
- Freeze the train/test split.
- Generate standardized sliding windows.
- Implement the four representative generator families in parallel.

Once the preprocessing pipeline is finalized, it should remain unchanged for all subsequent experiments to ensure fairness and reproducibility.