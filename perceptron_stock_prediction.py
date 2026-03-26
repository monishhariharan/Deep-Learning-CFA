"""
Perceptron for Stock Return Prediction
=======================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)

Architecture : Single-layer Perceptron (from-scratch, NumPy only)
Task         : Binary classification — predict whether next-day
               stock return is positive (1) or negative (0)
Features     : 5 technical indicators derived from simulated OHLCV data
               1. 5-day return momentum
               2. Price relative to 20-day moving average
               3. Normalised volume change
               4. Volatility (10-day rolling std of returns)
               5. RSI proxy (ratio of avg up-days to avg down-days)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_stock_data(n_days: int = 1000) -> np.ndarray:
    """
    Simulate a daily price series with a mild upward drift and
    realistic volatility clustering (GARCH-lite).
    Returns an array of closing prices.
    """
    rets = []
    vol, prev_ret = 0.015, 0.0
    for _ in range(n_days - 1):
        # Weak momentum autocorrelation — gives the perceptron a real signal
        shock    = np.random.normal(0.10 * prev_ret, vol)
        vol      = 0.94 * vol + 0.06 * abs(shock)
        rets.append(shock)
        prev_ret = shock
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices)


def build_features(prices: np.ndarray, lookback: int = 20) -> tuple:
    """
    Engineer 5 technical features and a binary return label.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5)
    y : np.ndarray, shape (n_samples,)   — 1 if next-day return > 0 else 0
    """
    returns = np.diff(np.log(prices))            # log returns
    n = len(returns)
    features, labels = [], []

    for t in range(lookback, n - 1):
        # Feature 1 — 5-day momentum
        mom5 = np.sum(returns[t - 5 : t])

        # Feature 2 — price vs 20-day moving average (z-score style)
        ma20 = np.mean(prices[t - lookback : t])
        price_vs_ma = (prices[t] - ma20) / ma20

        # Feature 3 — normalised volume proxy (random walk volume)
        #   We approximate volume with the square of returns (activity proxy)
        vol_now  = returns[t] ** 2
        vol_past = np.mean(returns[t - 5 : t] ** 2)
        norm_vol = (vol_now - vol_past) / (vol_past + 1e-8)

        # Feature 4 — rolling volatility (10-day std of returns)
        roll_vol = np.std(returns[t - 10 : t])

        # Feature 5 — RSI proxy
        up_days   = returns[t - 14 : t][returns[t - 14 : t] > 0]
        down_days = returns[t - 14 : t][returns[t - 14 : t] < 0]
        avg_up    = np.mean(up_days)   if len(up_days)   > 0 else 1e-8
        avg_down  = abs(np.mean(down_days)) if len(down_days) > 0 else 1e-8
        rsi_proxy = avg_up / (avg_up + avg_down)   # 0–1, >0.5 = bullish

        features.append([mom5, price_vs_ma, norm_vol, roll_vol, rsi_proxy])
        labels.append(1.0 if returns[t + 1] > 0 else 0.0)

    return np.array(features), np.array(labels)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PERCEPTRON CLASS  (faithful to PDF, extended for multi-feature input)
# ═════════════════════════════════════════════════════════════════════════════

class Perceptron:
    """
    Single-layer Perceptron trained with the Perceptron learning rule
    (gradient descent on misclassified samples), exactly as presented
    in Bilokon & Simonian (2025).

    Parameters
    ----------
    dim          : int   — number of input features
    learning_rate: float — step size γ for weight updates
    epochs       : int   — number of full passes over the training set
    """

    def __init__(self, dim: int, learning_rate: float = 0.01, epochs: int = 100):
        self.dim           = dim
        self.learning_rate = learning_rate
        self.epochs        = epochs
        # Initialise weights & bias from N(0,1)  — as in the PDF
        self.weights = np.random.normal(size=(self.dim, 1))
        self.bias    = np.random.normal()
        self.loss_history: list[float] = []

    # ── Core methods (directly from PDF) ─────────────────────────────────────

    def _activation(self, v: float) -> float:
        """Step function: 0 if v < 0, else 1."""
        return 0.0 if v < 0.0 else 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train using the perceptron update rule:
            if target ≠ ŷ:
                bias    -= lr * (ŷ − target)
                weights -= lr * (ŷ − target) * x
        """
        for epoch in range(self.epochs):
            errors = 0
            for x_i, target in zip(X, y):
                x_col  = x_i.reshape(-1, 1)
                v      = float((self.weights.T @ x_col).item()) + self.bias
                y_pred = self._activation(v)
                if target != y_pred:
                    delta         = y_pred - target
                    self.bias    -= self.learning_rate * delta
                    self.weights -= self.learning_rate * delta * x_col
                    errors       += 1
            self.loss_history.append(errors / len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions for each row in X."""
        y_pred = []
        for x_i in X:
            x_col  = x_i.reshape(-1, 1)
            v      = float((self.weights.T @ x_col).item()) + self.bias
            y_pred.append(self._activation(v))
        return np.array(y_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 60)
    print("  Perceptron — Stock Return Direction Prediction")
    print("  Bilokon & Simonian (2025) | CFA Research Foundation")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Simulating price series …")
    prices = simulate_stock_data(n_days=1200)
    X, y   = build_features(prices)
    print(f"    Samples: {len(y)}  |  Features: {X.shape[1]}")
    print(f"    Class balance — Up: {y.mean():.1%}  Down: {(1-y).mean():.1%}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False   # keep temporal order
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2] Training Perceptron …")
    model = Perceptron(dim=X.shape[1], learning_rate=0.01, epochs=200)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    print(f"\n[3] Results")
    print(f"    Train accuracy : {train_acc:.2%}")
    print(f"    Test  accuracy : {test_acc:.2%}")
    print(f"\n    Learned weights (one per feature):")
    feature_names = ["Momentum-5d", "Price vs MA20", "Vol change",
                     "Roll volatility", "RSI proxy"]
    for name, w in zip(feature_names, model.weights.flatten()):
        print(f"      {name:<18}  {w:+.4f}")
    print(f"      {'Bias':<18}  {model.bias:+.4f}")

    # ── Classification report ─────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("\n[4] Classification Report (test set)")
    print(classification_report(y_test, y_pred,
                                labels=[0.0, 1.0],
                                target_names=["Down (0)", "Up (1)"]))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print("    Confusion Matrix:")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Plot training curve ───────────────────────────────────────────────────
    print("\n[5] Saving training curve …")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_history, color="#4C72B0", linewidth=1.8)
    ax.set_title("Perceptron Training — Misclassification Rate per Epoch",
                 fontsize=13, pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassification Rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/training_curve.png", dpi=150)
    plt.close()
    print("    Saved → training_curve.png")
    print("\nDone.")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()
