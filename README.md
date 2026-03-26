# Deep Learning for Asset Management

> **Reference implementation** for *"Deep Learning"* — Bilokon & Simonian, CFA Institute Research Foundation (2025).

Six deep-learning models built entirely from scratch with **NumPy only**, applied to real S&P 500 equity data downloaded live via [yfinance](https://github.com/ranaroussi/yfinance).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Workflow](#project-workflow)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Models](#models)
   - [Perceptron](#1-perceptron)
   - [Feed-Forward Neural Network (FFNN)](#2-feed-forward-neural-network-ffnn)
   - [Long Short-Term Memory (LSTM)](#3-long-short-term-memory-lstm)
   - [Gated Recurrent Unit (GRU)](#4-gated-recurrent-unit-gru)
   - [Generative Adversarial Network (GAN)](#5-generative-adversarial-network-gan)
   - [Variational Autoencoder (VAE)](#6-variational-autoencoder-vae)
5. [Model Comparison & Portfolio Backtest](#model-comparison--portfolio-backtest)
6. [Data Pipeline](#data-pipeline)
7. [Outputs](#outputs)
8. [References](#references)

---

## Overview

Each model is a self-contained Python script that:

- **Downloads** 3 years of real OHLCV data for a screened subset of S&P 500 stocks.
- **Engineers** 7 technical features (momentum, MA-ratio, volume z-score, volatility, RSI, Bollinger %B).
- **Trains** and **evaluates** the model, printing metrics to the console.
- **Saves** a multi-panel PNG dashboard to the current directory.

All neural-network maths (forward pass, backpropagation, gradient updates) are implemented in pure NumPy — no PyTorch, TensorFlow, or Keras.

---

## Project Workflow

For a visual overview of the end-to-end pipeline open **`project_workflow.html`** in any browser.

```
Raw OHLCV  →  Auto Screener  →  Feature Engineering  →  Pre-processing
                                                               │
              ┌────────────────────────────────────────────────┘
              ▼
    Perceptron · FFNN · LSTM · GRU · GAN · VAE
              │
              ▼
    Evaluation (Acc / AUC / F1 / Edge)  →  PNG dashboards
              │
              ▼
    Model Comparison  →  Portfolio Backtest (long/flat vs buy-and-hold)
```

---

## Prerequisites & Installation

**Python 3.9+** is required.

```bash
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|---|---|
| `numpy` | All neural-network maths |
| `pandas` | Data wrangling |
| `yfinance` | Live OHLCV download |
| `scikit-learn` | StandardScaler, ROC-AUC |
| `matplotlib` | PNG dashboard generation |

---

## Models

All models share the same data pipeline:

- **Universe**: 28 large-cap S&P 500 tickers
- **Screener**: ranks by Sharpe, 6-month momentum and annualised volatility; selects top-N
- **Features**: 7 technical indicators (see [Data Pipeline](#data-pipeline))
- **Split**: 80 % train / 20 % test (no shuffle — time order preserved); 15 % of train used as validation
- **Label**: binary — next-day return direction (1 = up, 0 = down/flat)

---

### 1. Perceptron

```
File : perceptron_yfinance.py
```

The simplest possible classifier — a single neuron with a step activation function trained with the classic Perceptron learning rule.

**Architecture**: `[7 inputs → 1 output]`

| Hyperparameter | Default |
|---|---|
| Learning rate | 0.005 |
| Epochs | 200 |

```bash
python perceptron_yfinance.py                 # auto-screen top-5
python perceptron_yfinance.py --ticker AAPL   # single stock
python perceptron_yfinance.py --top 3         # screen top-3
```

**Output**: `perceptron_results.png` — 6-panel dashboard (training curve, confusion matrix, feature weights, accuracy vs baseline, per-stock breakdown, ROC placeholder).

> *The simulated-data variant `perceptron_stock_prediction.py` runs the same Perceptron on GARCH-synthetic prices (no internet required).*

---

### 2. Feed-Forward Neural Network (FFNN)

```
File : ffnn_stock_prediction.py
```

A multi-layer perceptron trained with mini-batch SGD + momentum and binary cross-entropy loss.

**Architecture**: `[7 → 64 → 32 → 1]`  (Xavier initialisation, sigmoid activations)

| Hyperparameter | Default |
|---|---|
| Hidden layers | `[64, 32]` |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Batch size | 32 |
| Epochs | 200 |

```bash
python ffnn_stock_prediction.py
python ffnn_stock_prediction.py --ticker MSFT --epochs 300
python ffnn_stock_prediction.py --top 3 --hidden 128 64
```

**Output**: `ffnn_results.png` — loss/accuracy curves, ROC-AUC bars, confusion matrices.

---

### 3. Long Short-Term Memory (LSTM)

```
File : lstm_stock_prediction.py
```

A single-layer LSTM with explicit forget / input / output gates and cell state, trained with truncated BPTT. All gate gradients verified numerically (error < 1 × 10⁻¹⁰).

**Architecture**: `LSTM(hidden=32) → Dense(1, sigmoid)`

| Hyperparameter | Default |
|---|---|
| Sequence length (T) | 20 |
| Hidden units (H) | 32 |
| Optimiser | Adam |
| Gradient clipping | ≤ 5 |
| Epochs | 100 |

```bash
python lstm_stock_prediction.py
python lstm_stock_prediction.py --ticker NVDA --seq_len 30 --hidden 64
```

**Output**: `lstm_results.png`

---

### 4. Gated Recurrent Unit (GRU)

```
File : gru_stock_prediction.py
```

A lighter recurrent model with update and reset gates (~25 % fewer parameters than LSTM). Gradient check error < 1 × 10⁻¹¹.

**Architecture**: `GRU(hidden=32) → Dense(1, sigmoid)`

| Hyperparameter | Default |
|---|---|
| Sequence length (T) | 20 |
| Hidden units (H) | 32 |
| Optimiser | Adam |
| Gradient clipping | ≤ 5 |
| Epochs | 100 |

```bash
python gru_stock_prediction.py
python gru_stock_prediction.py --ticker AMZN --hidden 64 --epochs 150
```

**Output**: `gru_results.png`

---

### 5. Generative Adversarial Network (GAN)

```
File : gan_synthetic_market.py
```

A minimax GAN where the Generator learns to produce realistic-looking feature vectors from Gaussian noise, and the Discriminator learns to distinguish real from synthetic.

**Application**: synthetic market scenario generation for portfolio stress-testing and data augmentation (Exhibit 7 of the PDF).

| Hyperparameter | Default |
|---|---|
| Latent dimension | 16 |
| Hidden units | 64 |
| Epochs | 200 |

```bash
python gan_synthetic_market.py
python gan_synthetic_market.py --ticker NVDA --latent 32 --hidden 128 --epochs 300
```

**Output**: `gan_results.png` — real vs fake feature distributions, D(real)/D(fake) training curves.

---

### 6. Variational Autoencoder (VAE)

```
File : vae_anomaly_detection.py
```

An unsupervised anomaly detector. The VAE is trained on normal market days; at inference, days with high reconstruction error are flagged as anomalous. Gradient check error < 1 × 10⁻¹².

**Loss**: `BCE_reconstruction + β · KL(q(z|x) ‖ p(z))`

| Hyperparameter | Default |
|---|---|
| Latent dimension | 4 |
| Hidden units | 32 |
| β (KL weight) | 1 |
| Epochs | 100 |
| Anomaly threshold | top 10 % reconstruction error |

```bash
python vae_anomaly_detection.py
python vae_anomaly_detection.py --ticker SPY --latent 8 --epochs 150
python vae_anomaly_detection.py --anomaly_pct 5      # stricter threshold
```

**Output**: `vae_results.png` — ELBO training curve, latent-space scatter, anomaly timeline.

---

## Model Comparison & Portfolio Backtest

```
File : model_comparison.py
```

Runs **Perceptron, FFNN, LSTM and GRU** head-to-head on the same screened stocks and produces a unified benchmark.

```bash
python model_comparison.py               # screen + top-5, 80 epochs
python model_comparison.py --top 3
python model_comparison.py --ticker AAPL
python model_comparison.py --epochs 150
```

**Outputs**:

| File | Contents |
|---|---|
| `model_comparison.png` | 6-panel dashboard: accuracy bars, AUC bars, edge heatmap, equity curves, average rank, summary table |
| `model_backtests.png` | Average equity curves (long/flat signal vs buy-and-hold), Sharpe ratio, max drawdown per model |

Console table example:

```
  Model        Acc      AUC      Edge
  ─────────────────────────────────────
  Perceptron  52.40%   0.521   +2.40%
  FFNN        55.10%   0.558   +5.10%
  LSTM        54.30%   0.546   +4.30%
  GRU         54.80%   0.551   +4.80%
```

---

## Data Pipeline

| Step | Detail |
|---|---|
| **Download** | `yfinance` · 3-year history · daily OHLCV · auto-adjusted |
| **Universe** | 28 S&P 500 large-cap tickers |
| **Screener** | Composite rank of Sharpe, 6-month momentum, annualised volatility → top-N |
| **Features** | 5-day momentum, 10-day momentum, price / MA-20, volume z-score, rolling volatility (20d), RSI (14d), Bollinger %B |
| **Scaling** | `StandardScaler` (fit on train, applied to test — no leakage) |
| **Split** | 80 % train / 20 % test (temporal, no shuffle) |

---

## Outputs

| Script | Output PNG |
|---|---|
| `perceptron_yfinance.py` | `perceptron_results.png` |
| `ffnn_stock_prediction.py` | `ffnn_results.png` |
| `lstm_stock_prediction.py` | `lstm_results.png` |
| `gru_stock_prediction.py` | `gru_results.png` |
| `gan_synthetic_market.py` | `gan_results.png` |
| `vae_anomaly_detection.py` | `vae_results.png` |
| `model_comparison.py` | `model_comparison.png`, `model_backtests.png` |

All PNG files are written to the current working directory and ignored by `.gitignore`.

---

## References

- Bilokon, P. & Simonian, J. (2025). *Deep Learning*. CFA Institute Research Foundation.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014*.
- Goodfellow, I. et al. (2014). Generative adversarial nets. *NeurIPS 2014*.
- Kingma, D. P. & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR 2014*.
