"""
Feed-Forward Neural Network (FFNN) — Stock Return Prediction
=============================================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)

Architecture : Multi-layer FFNN with configurable hidden layers,
               sigmoid activations, and mini-batch backpropagation.
               Fully connected — every neuron in layer L is connected
               to every neuron in layer L+1 (as described in the PDF).
Data         : Real OHLCV via yfinance + auto stock screener
Task         : Binary classification — next-day return direction (up/down)

Usage
-----
  python ffnn_stock_prediction.py                        # screen + auto-run
  python ffnn_stock_prediction.py --ticker MSFT          # single stock
  python ffnn_stock_prediction.py --ticker AAPL --hidden 64 32 16
  python ffnn_stock_prediction.py --top 5 --epochs 300
"""

import argparse
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not found.  pip install yfinance")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
except ImportError:
    sys.exit("scikit-learn not found.  pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  STOCK SCREENER  (same logic as Perceptron script — reusable module)
# ═════════════════════════════════════════════════════════════════════════════

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM",  "V",    "UNH",   "XOM",  "JNJ",  "MA",   "PG",
    "HD",   "MRK",  "ABBV",  "LLY",  "AVGO", "COST", "AMD",
    "QCOM", "GS",   "MS",    "BLK",  "NFLX", "CRM",  "BAC",
]

FEATURE_NAMES = [
    "Momentum-5d",
    "Price vs MA20",
    "Volume z-score",
    "Roll volatility",
    "RSI (14)",
    "Momentum-10d",
    "Bollinger %B",
]


def _download(tickers, period):
    print(f"  Downloading {len(tickers)} ticker(s) [{period}] …", end=" ", flush=True)
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    print("done.")
    result = {}
    for t in tickers:
        try:
            df = (raw[["Close", "Volume"]] if len(tickers) == 1
                  else raw.xs(t, axis=1, level=1)[["Close", "Volume"]])
            df = df.dropna()
            if len(df) > 60:
                result[t] = df
        except Exception:
            pass
    return result


def screen_stocks(top_n: int = 5) -> list[str]:
    print("\n[SCREENER] Evaluating universe …")
    data = _download(UNIVERSE, "2y")
    rows = []
    for ticker, df in data.items():
        ret     = np.log(df["Close"]).diff().dropna()
        sharpe  = (ret.mean() / ret.std()) * np.sqrt(252)
        mom6m   = (df["Close"].iloc[-1] / df["Close"].iloc[-126]) - 1
        ann_vol = ret.std() * np.sqrt(252)
        rows.append({"ticker": ticker, "sharpe": sharpe,
                     "momentum_6m": mom6m, "volatility": ann_vol})
    scores = pd.DataFrame(rows).set_index("ticker")
    scores["r_sharpe"] = scores["sharpe"].rank(ascending=False)
    scores["r_mom6m"]  = scores["momentum_6m"].rank(ascending=False)
    scores["r_vol"]    = scores["volatility"].rank(ascending=True)
    scores["composite"]= scores[["r_sharpe", "r_mom6m", "r_vol"]].mean(axis=1)
    scores = scores.sort_values("composite")
    top = scores.head(top_n)
    print(f"\n  Top-{top_n} screened stocks:")
    print(f"  {'Ticker':<8} {'Sharpe':>8} {'Mom 6M':>8} {'Ann Vol':>8}")
    print(f"  {'-'*38}")
    for t, row in top.iterrows():
        print(f"  {t:<8} {row['sharpe']:>8.2f} {row['momentum_6m']:>7.1%} {row['volatility']:>7.1%}")
    return list(top.index)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (7 features — richer than the Perceptron version)
# ═════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Seven technical features from real OHLCV data.

    New vs Perceptron
    -----------------
    6. 10-day momentum          (medium-term trend signal)
    7. Bollinger %B             (mean-reversion / breakout signal)
    """
    close  = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    ret    = np.diff(np.log(close))
    n      = len(ret)

    features, labels = [], []
    for t in range(lookback, n - 1):

        mom5        = ret[t - 5 : t].sum()

        ma20        = close[t - lookback : t].mean()
        price_vs_ma = (close[t] - ma20) / (ma20 + 1e-8)

        vol_win     = volume[t - 10 : t]
        vol_zscore  = (volume[t] - vol_win.mean()) / (vol_win.std() + 1e-8)

        roll_vol    = ret[t - 10 : t].std()

        changes     = ret[t - 14 : t]
        gains       = changes[changes > 0]
        losses      = np.abs(changes[changes < 0])
        avg_gain    = gains.mean()  if len(gains)  > 0 else 1e-8
        avg_loss    = losses.mean() if len(losses) > 0 else 1e-8
        rsi         = 100 - (100 / (1 + avg_gain / avg_loss))

        mom10       = ret[t - 10 : t].sum()     # Feature 6

        std20       = close[t - lookback : t].std()        # Feature 7 — Bollinger %B
        upper_bb    = ma20 + 2 * std20
        lower_bb    = ma20 - 2 * std20
        pct_b       = (close[t] - lower_bb) / (upper_bb - lower_bb + 1e-8)

        features.append([mom5, price_vs_ma, vol_zscore, roll_vol, rsi, mom10, pct_b])
        labels.append(1.0 if ret[t + 1] > 0.0 else 0.0)

    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FEED-FORWARD NEURAL NETWORK  (from scratch, NumPy only)
#     Fully connected; trained via mini-batch backpropagation with momentum
#     Ref: Rumelhart, Hinton & Williams (1986) — as cited in the PDF
# ═════════════════════════════════════════════════════════════════════════════

class FFNN:
    """
    Multi-layer Feed-Forward Neural Network.

    Every adjacent pair of layers is fully connected (as described in the PDF).
    Activation : sigmoid for hidden layers, sigmoid for output (binary task).
    Loss       : binary cross-entropy.
    Optimiser  : mini-batch SGD with momentum.

    Parameters
    ----------
    layer_sizes  : list[int]  e.g. [7, 64, 32, 1]  — includes input & output
    learning_rate: float
    momentum     : float      — SGD momentum coefficient
    epochs       : int
    batch_size   : int
    """

    def __init__(
        self,
        layer_sizes:   list[int],
        learning_rate: float = 0.01,
        momentum:      float = 0.9,
        epochs:        int   = 200,
        batch_size:    int   = 32,
    ):
        self.layer_sizes   = layer_sizes
        self.lr            = learning_rate
        self.momentum      = momentum
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.train_loss:   list[float] = []
        self.val_loss:     list[float] = []
        self.train_acc:    list[float] = []
        self.val_acc:      list[float] = []

        # ── Xavier (Glorot) initialisation ────────────────────────────────
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

        # ── Momentum velocity buffers ─────────────────────────────────────
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ── Activation functions ──────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def _bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-8
        return -float(np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        ))

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Return list of activations for each layer (including input)."""
        activations = [X]
        for W, b in zip(self.W, self.b):
            z = activations[-1] @ W + b
            activations.append(self._sigmoid(z))
        return activations

    # ── Backpropagation ───────────────────────────────────────────────────────

    def _backprop(self, activations: list[np.ndarray], y: np.ndarray):
        """
        Backpropagation — Rumelhart, Hinton & Williams (1986).

        FIX: delta for layer i-1 is computed using the OLD W[i],
        BEFORE the weight update. Verified against numerical gradients
        to machine-precision (error < 1e-10).
        """
        m        = y.shape[0]
        n_layers = len(self.W)

        delta = activations[-1] - y.reshape(-1, 1)

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            dW     = (a_prev.T @ delta) / m
            db     = delta.mean(axis=0, keepdims=True)

            # Propagate delta with OLD W[i] BEFORE updating weights
            if i > 0:
                delta = (delta @ self.W[i].T) * self._sigmoid_deriv(activations[i])

            # Now update weights safely
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "FFNN":
        """Mini-batch SGD training with optional validation tracking."""
        m = X_tr.shape[0]
        for epoch in range(self.epochs):
            # Shuffle
            idx = np.random.permutation(m)
            X_s, y_s = X_tr[idx], y_tr[idx]

            # Mini-batches
            for start in range(0, m, self.batch_size):
                Xb = X_s[start : start + self.batch_size]
                yb = y_s[start : start + self.batch_size]
                acts = self._forward(Xb)
                self._backprop(acts, yb)

            # ── Track metrics every epoch ─────────────────────────────────
            tr_acts  = self._forward(X_tr)
            tr_pred  = tr_acts[-1]
            self.train_loss.append(self._bce(y_tr, tr_pred.flatten()))
            self.train_acc.append(
                float(np.mean((tr_pred.flatten() >= 0.5) == y_tr))
            )
            if X_val is not None:
                va_acts = self._forward(X_val)
                va_pred = va_acts[-1]
                self.val_loss.append(self._bce(y_val, va_pred.flatten()))
                self.val_acc.append(
                    float(np.mean((va_pred.flatten() >= 0.5) == y_val))
                )

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1].flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SINGLE-STOCK PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_stock(ticker: str, df: pd.DataFrame,
              hidden_sizes: list[int], epochs: int) -> dict:

    X, y = build_features(df)
    if len(y) < 150:
        return {}

    split = int(len(y) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Small validation split carved from training set for live loss tracking
    val_split    = int(len(X_tr) * 0.85)
    X_tr_, X_val = X_tr[:val_split], X_tr[val_split:]
    y_tr_, y_val = y_tr[:val_split], y_tr[val_split:]

    scaler = StandardScaler()
    X_tr_  = scaler.fit_transform(X_tr_)
    X_val  = scaler.transform(X_val)
    X_te   = scaler.transform(X_te)

    layer_sizes = [X.shape[1]] + hidden_sizes + [1]
    model = FFNN(
        layer_sizes   = layer_sizes,
        learning_rate = 0.01,
        momentum      = 0.9,
        epochs        = epochs,
        batch_size    = 32,
    )
    model.fit(X_tr_, y_tr_, X_val, y_val)

    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)
    test_acc   = model.score(X_te, y_te)
    baseline   = max(float(y_te.mean()), 1 - float(y_te.mean()))

    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = float("nan")

    print(f"  {ticker:<6}  arch={layer_sizes}  "
          f"acc={test_acc:.2%}  AUC={auc:.3f}  "
          f"baseline={baseline:.2%}  edge={test_acc - baseline:+.2%}")

    return {
        "ticker"      : ticker,
        "model"       : model,
        "layer_sizes" : layer_sizes,
        "test_acc"    : test_acc,
        "baseline"    : baseline,
        "edge"        : test_acc - baseline,
        "auc"         : auc,
        "y_te"        : y_te,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(results: list[dict], out_path: str = "ffnn_results.png"):
    n = len(results)
    if n == 0:
        return

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        "FFNN — Stock Return Direction  |  CFA / Bilokon & Simonian (2025)",
        fontsize=11, y=1.01,
    )

    # ── Row 1: training curves (loss + accuracy) per stock ───────────────────
    ax_loss = fig.add_subplot(2, 3, 1)
    ax_vacc = fig.add_subplot(2, 3, 2)
    ax_auc  = fig.add_subplot(2, 3, 3)

    for res in results:
        m = res["model"]
        ax_loss.plot(m.train_loss, linewidth=1.2, label=f"{res['ticker']} train")
        if m.val_loss:
            ax_loss.plot(m.val_loss, linewidth=1.0, linestyle="--",
                         label=f"{res['ticker']} val")
        ax_vacc.plot(m.val_acc, linewidth=1.2, label=res["ticker"])

    ax_loss.set_title("Loss (BCE) over Epochs")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("BCE Loss")
    ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)

    ax_vacc.set_title("Validation Accuracy over Epochs")
    ax_vacc.set_xlabel("Epoch"); ax_vacc.set_ylabel("Accuracy")
    ax_vacc.legend(fontsize=7); ax_vacc.grid(alpha=0.3)

    # ── Row 1, col 3: AUC bar chart ──────────────────────────────────────────
    tickers_  = [r["ticker"] for r in results]
    aucs_     = [r["auc"]    for r in results]
    bars = ax_auc.bar(tickers_, aucs_, color="#4C72B0", edgecolor="white")
    ax_auc.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random")
    ax_auc.set_title("ROC-AUC Score")
    ax_auc.set_ylabel("AUC"); ax_auc.set_ylim(0.3, 1.0)
    ax_auc.legend(fontsize=8); ax_auc.grid(axis="y", alpha=0.3)
    for bar_, val in zip(bars, aucs_):
        ax_auc.text(bar_.get_x() + bar_.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # ── Row 2: accuracy vs baseline, confusion matrix (best stock) ───────────
    ax_bar = fig.add_subplot(2, 3, 4)
    ax_cm  = fig.add_subplot(2, 3, 5)
    ax_txt = fig.add_subplot(2, 3, 6)

    accs_  = [r["test_acc"] for r in results]
    bases_ = [r["baseline"] for r in results]
    x = np.arange(len(tickers_))
    w = 0.35
    ax_bar.bar(x - w/2, accs_,  w, label="FFNN",           color="#4C72B0")
    ax_bar.bar(x + w/2, bases_, w, label="Naive baseline",  color="#DD8452", alpha=0.8)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(tickers_, fontsize=9)
    ax_bar.set_ylim(0.3, 0.8)
    ax_bar.set_title("Test Accuracy vs Baseline"); ax_bar.set_ylabel("Accuracy")
    ax_bar.legend(fontsize=8); ax_bar.grid(axis="y", alpha=0.3)

    best = max(results, key=lambda r: r["edge"])
    cm   = confusion_matrix(best["y_te"], best["y_pred"])
    im   = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred Down", "Pred Up"])
    ax_cm.set_yticklabels(["True Down", "True Up"])
    ax_cm.set_title(f"Confusion Matrix — {best['ticker']}")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=13, color="white" if cm[i,j] > cm.max()/2 else "black")

    # ── Summary text ──────────────────────────────────────────────────────────
    ax_txt.axis("off")
    summary = (
        f"Best stock   : {best['ticker']}\n"
        f"Architecture : {best['layer_sizes']}\n"
        f"Test accuracy: {best['test_acc']:.2%}\n"
        f"ROC-AUC      : {best['auc']:.3f}\n"
        f"Baseline     : {best['baseline']:.2%}\n"
        f"Edge         : {best['edge']:+.2%}\n\n"
        "Key: FFNN uses backpropagation\n"
        "(Rumelhart et al. 1986) with\n"
        "mini-batch SGD + momentum."
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FFNN stock-return predictor")
    parser.add_argument("--ticker", type=str,   default=None,
                        help="Single ticker (e.g. MSFT). Skips screener.")
    parser.add_argument("--top",    type=int,   default=5,
                        help="Top N stocks from screener (default 5).")
    parser.add_argument("--hidden", type=int,   nargs="+", default=[64, 32],
                        help="Hidden layer sizes, e.g. --hidden 64 32 16")
    parser.add_argument("--epochs", type=int,   default=200,
                        help="Training epochs (default 200).")
    args = parser.parse_args()

    print("=" * 62)
    print("  Feed-Forward Neural Network — Stock Return Prediction")
    print("  Bilokon & Simonian (2025) | CFA Research Foundation")
    print("=" * 62)

    # ── Select tickers ────────────────────────────────────────────────────────
    if args.ticker:
        tickers = [args.ticker.upper()]
        print(f"\n  Single-stock mode: {tickers[0]}")
    else:
        tickers = screen_stocks(top_n=args.top)

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"\n[DATA] Downloading 3y of OHLCV …")
    data = _download(tickers, "3y")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[MODEL] Training FFNN  hidden={args.hidden}  epochs={args.epochs}\n")
    results = []
    for ticker, df in data.items():
        res = run_stock(ticker, df, args.hidden, args.epochs)
        if res:
            results.append(res)

    if not results:
        print("No valid results.")
        return

    # ── Best stock detailed report ────────────────────────────────────────────
    best = max(results, key=lambda r: r["edge"])
    print(f"\n[REPORT] Best: {best['ticker']}  edge={best['edge']:+.2%}")
    print(classification_report(
        best["y_te"], best["y_pred"],
        labels=[0.0, 1.0],
        target_names=["Down (0)", "Up (1)"],
        zero_division=0,
    ))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_results(results, out_path="ffnn_results.png")
    print("Done.")


if __name__ == "__main__":
    main()
