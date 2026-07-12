# Classical Statistical Generator: ARIMA-GARCH + Residual Bootstrap Explanation

This document provides a comprehensive, readable guide to the econometric theory, mathematical formulations, grid search order selections, statistical tests, and diagnostic validations used in the Classical Statistical synthetic financial time series generator.

---

## 1. Stationarity & Pre-Fit Diagnostics

Before applying any time series models, we must assess the statistical properties of the asset returns.

### A. Autocorrelation Function (ACF) & Partial Autocorrelation Function (PACF)
*   **ACF (Autocorrelation Function)**: Measures the total linear correlation between $y_t$ and its lag $y_{t-k}$. It captures both direct correlation and indirect correlation (propagation through intermediate lags).
*   **PACF (Partial Autocorrelation Function)**: Measures the direct correlation between $y_t$ and $y_{t-k}$ *after removing* the linear influence of all intermediate lags. It is the primary tool for identifying the order of an Autoregressive (AR) process.
*   **Squared Returns ACF/PACF**: Used to inspect **volatility clustering**. If raw returns show minimal autocorrelation but squared returns show persistent and slowly decaying ACF, it indicates that volatility is time-varying and clustered (periods of high volatility follow high volatility).

### B. Augmented Dickey-Fuller (ADF) Test
To fit an ARIMA model, the input series must be **stationary** (having a constant mean, variance, and autocovariance over time). The ADF test checks for the presence of a unit root:
*   **Null Hypothesis ($H_0$)**: The series is non-stationary (has a unit root).
*   **Alternative Hypothesis ($H_1$)**: The series is stationary.
*   **p-value**: If the $p$-value is less than the significance level (typically $0.05$), we reject the null hypothesis of non-stationarity.
*   **Critical Values**: Represent the thresholds for the ADF statistic at 1%, 5%, and 10% significance levels. If the test statistic is **more negative** (less than) the critical value, we reject $H_0$. 
    *   *Our Result*: The ADF statistic of our normalized returns is `-13.428`, which is far below the 1% critical value (`-3.431`). The $p$-value is `4.06e-25` ($\approx 0$), confirming that the series is highly stationary, so no differencing is needed ($d=0$).

---

## 2. Conditional Mean Modeling: ARIMA

We model the short-term linear dependencies of returns using an $\text{ARIMA}(p, d, q)$ model.

### A. Model Equation
For a stationary return series $r_t$ (with $d=0$):
$$r_t = c + \sum_{i=1}^p \phi_i r_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

Where:
*   $c$ is the constant drift.
*   $p$ is the **Autoregressive (AR) order** (number of lag observations of returns).
*   $q$ is the **Moving Average (MA) order** (number of lag observations of residual shocks).
*   $\phi_i$ and $\theta_j$ are the estimated parameters.
*   $\epsilon_t$ is the residual error (shock) at time $t$.

### B. ARIMA Grid Search & AIC
To find the optimal $(p, q)$ order, we run a grid search over $p \in [0, 5]$ and $q \in [0, 5]$ and select the combination that minimizes the **Akaike Information Criterion (AIC)**:
$$\text{AIC} = 2k - 2\ln(\hat{L})$$
where $k = p + q + 1$ is the number of parameters and $\hat{L}$ is the maximum likelihood estimation. AIC rewards model fit but penalizes complexity to prevent overfitting.

### C. Overfitting and Train-Test Gap Analysis
When searching up to order 5, we compare:
*   **ARIMA(3, 0, 3)**: AIC = `14764.07`, Train Ljung-Box p-value = `0.1535`.
*   **ARIMA(5, 0, 4)**: AIC = `14759.99` (lowest AIC), Train Ljung-Box p-value = `0.4372`.
*   **Analysis**: Higher order ARIMA models can overfit. We monitor the **RMSE Gap** between train residuals and out-of-sample test residuals ($|\text{RMSE}_{\text{test}} - \text{RMSE}_{\text{train}}|$). For both models, the test RMSE remains extremely stable ($\approx 0.851$), indicating no generalization breakdown. However, **ARIMA(5, 0, 4)** is chosen because it minimizes Train AIC while achieving a significantly higher Ljung-Box p-value on both Train (`0.4372`) and Test (`0.0125`) sets than ARIMA(3,3).

### D. Ljung-Box Test on ARIMA Residuals
Before fitting GARCH, we must ensure that the ARIMA model has fully captured the conditional mean dynamics.
*   **Null Hypothesis ($H_0$)**: The residuals $\hat{\epsilon}_t$ are independent (white noise).
*   We want a $p$-value $> 0.05$ at lag 10 to fail to reject $H_0$, indicating that no significant linear autocorrelation remains.

---

## 3. Volatility Modeling: GARCH

Financial returns exhibit conditional heteroskedasticity (volatility clustering). We model this variance using a GARCH model.

### A. ARCH Effects Check (Engle's LM Test)
Before fitting GARCH, we verify the presence of ARCH effects in the ARIMA residuals $\hat{\epsilon}_t$. The test runs a regression of squared residuals on their lags:
$$\hat{\epsilon}_t^2 = \alpha_0 + \sum_{i=1}^m \alpha_i \hat{\epsilon}_{t-i}^2 + u_t$$
*   **Null Hypothesis ($H_0$)**: $\alpha_1 = \dots = \alpha_m = 0$ (homoskedasticity / no ARCH effects).
*   If $p$-value $< 0.05$, we reject $H_0$, confirming volatility clustering and justifying GARCH.

### B. GARCH Model Equation
The $\text{GARCH}(P, Q)$ model represents the conditional variance $\sigma_t^2$ of the ARIMA residuals $\epsilon_t$:
$$\epsilon_t = \sigma_t z_t \quad \text{where } z_t \sim \text{i.i.d. with } E[z_t]=0, Var[z_t]=1$$
$$\sigma_t^2 = \omega + \sum_{i=1}^P \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^Q \beta_j \sigma_{t-j}^2$$

Where:
*   $\sigma_t^2$ is the **conditional variance** (volatility at time $t$).
*   $\omega > 0$ is the baseline variance.
*   $\alpha_i \ge 0$ (ARCH parameters) represent the impact of past squared shocks on current volatility.
*   $\beta_j \ge 0$ (GARCH parameters) represent the persistence of volatility over time.

### C. GARCH Stability Safeguard
For a GARCH model to be stable and covariance-stationary, the persistence parameters must sum to less than one:
$$\sum_{i=1}^P \alpha_i + \sum_{j=1}^Q \beta_j < 1.0$$
If this sum is $\ge 1.0$, the conditional variance will explode during simulation. We implement a stability check: if the sum is $\ge 1.0$, we scale the $\alpha$ and $\beta$ parameters by a factor of $0.99 / (\sum \alpha + \sum \beta)$ to ensure stability.

### D. GARCH Post-Fit Model Verification
We extract the standardized residuals:
$$\hat{z}_t = \frac{\hat{\epsilon}_t}{\hat{\sigma}_t}$$
We run the Ljung-Box test on:
1.  **Standardized Residuals ($z_t$)**: Checks if any mean correlation remains (we want $p > 0.05$).
2.  **Squared Standardized Residuals ($z_t^2$)**: Checks if all volatility clustering has been removed. A $p$-value $> 0.05$ at lag 10 confirms that the standardized residuals are homoskedastic (white noise in variance).

---

## 4. Residual Bootstrap Simulation

Traditional parametric simulation assumes the standardized residuals $z_t$ follow a standard Normal or Student-$t$ distribution. However, financial return shocks are typically asymmetric and heavy-tailed. 

To preserve these empirical characteristics, we perform a **Residual Bootstrap**:
1.  Extract the empirical standardized residuals $\{\hat{z}_t\}_{t=1}^T$ from the fitted model.
2.  Set up a simulation horizon $T_{\text{total}} = T + B$, where $T = 5230$ and $B = 500$ is a **burn-in period** designed to remove any dependency on initial values.
3.  For each step $t$ in the simulation:
    *   Sample $z_t^*$ from $\{\hat{z}_t\}$ **with replacement**.
    *   Recursively calculate the conditional variance: $\sigma_t^2 = \omega + \sum \alpha_i \epsilon_{t-i}^2 + \sum \beta_j \sigma_{t-j}^2$.
    *   Generate the GARCH shock: $\epsilon_t = \sigma_t z_t^*$.
    *   Recursively calculate the ARIMA return: $y_t = c + \sum \phi_i y_{t-i} + \sum \theta_j \epsilon_{t-j} + \epsilon_t$.
4.  Discard the first $B = 500$ steps and keep the remaining $T$ steps.

---

## 5. Output: Synthetic Returns Generation

**What is "Synthetic" data?**

"Synthetic" refers to the **artificially generated financial returns** produced by your trained ARIMA-GARCH model. This is the PRIMARY OUTPUT and goal of this generator.

- **NOT** random noise
- **NOT** test data
- **IS** new, realistic return series that preserves the statistical properties learned from historical data

**How it's created:**
1. Fit ARIMA-GARCH on historical train data (2008-2023)
2. Bootstrap sample from empirical standardized residuals z_t
3. Recursively generate 5,230 new returns using fitted model parameters
4. Result: synthetic returns that are statistically similar to real returns but are NEW sequences

**Why generate synthetic data?**
- Training deep learning models (GANs, Diffusions) - need more data
- Backtesting trading strategies on alternative scenarios
- Stress testing with realistic but unseen market conditions
- Benchmark comparison: "classical statistical model vs. deep learning"

The generated returns are un-normalized back to raw log returns:
$$\text{LogReturn}_t = (\text{Normalized}_t \times \sigma_{\text{train}}) + \mu_{\text{train}}$$

**Output files** saved in `outputs/`:
1.  `synthetic_returns.csv`: Contains columns `Normalized` (standardized returns) and `LogReturn` (raw log returns). These are 5,230 NEW artificial returns.
2.  `synthetic_windows.npy`: Sliding windows of size $30$, shape `(5200, 30)`. Ready to feed into GANs/Diffusions for benchmark comparisons.

---

## 6. Evaluation: Comparing Synthetic vs. Real Returns

We evaluate the quality of **synthetic (generated)** returns against both training and test returns to ensure they are statistically realistic.

**What are we comparing?**
- **Train**: Historical S&P 500 returns (2008-2023) used to fit the model
- **Test**: Held-out historical returns (2023-2024) not used in training
- **Synthetic**: NEW artificial returns generated by the ARIMA-GARCH model (this is the output!)

The synthetic data should match the statistical properties of BOTH train and test data - showing the model learned realistic patterns, not just memorized training data.

### A. Moments Comparison
*   **Mean & Std Dev**: Confirms scale conservation.
*   **Skewness**: Assesses return asymmetry (financial returns are typically negatively skewed).
*   **Kurtosis**: Measures the thickness of the tails (excess kurtosis $> 3.0$ indicates fat tails).

### B. Distributional & Correlation Distances
*   **Wasserstein Distance**: The Earth Mover's Distance, measuring the minimum cost of turning the synthetic distribution shape into the real distribution shape (lower is better).
*   **Kolmogorov-Smirnov (KS) Test**:
    *   Checks if the synthetic returns and real returns come from identical distributions.
    *   We want the **KS Statistic** to be close to $0$ and the **KS p-value** to be $> 0.05$ to confirm statistical similarity.
*   **RMSE of ACF / Abs ACF**: Measures the error between real and synthetic autocorrelation functions up to 20 lags. Lower RMSE confirms the synthetic returns accurately reproduce both linear correlation (ACF) and volatility clustering (Absolute ACF).
