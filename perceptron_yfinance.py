"""
Perceptron — Stock Return Prediction with yfinance & Auto Screener
==================================================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)

Architecture : Single-layer Perceptron (from-scratch, NumPy only)
Data         : Real OHLCV prices via yfinance
Screener     : Ranks a candidate universe by Sharpe-momentum score,
               then trains & evaluates the Perceptron on each selected stock.

Usage
-----
  python perceptron_yfinance.py                  # auto-screen & run all
  python perceptron_yfinance.py --ticker AAPL    # single stock mode
  python perceptron_yfinance.py --top 5          # screen & run top-5 stocks
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

# ── Optional import guard ────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not found. Run:  pip install yfinance")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    sys.exit("scikit-learn not found. Run:  pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  AUTO STOCK SCREENER
# ═════════════════════════════════════════════════════════════════════════════

# S&P 500 mega/large-cap candidate universe
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V",    "UNH",   "XOM",  "JNJ",  "MA",   "PG",   "HD",
    "CVX", "MRK",  "ABBV",  "LLY",  "PEP",  "KO",   "AVGO", "COST",
    "TMO", "MCD",  "WMT",   "ACN",  "BAC",  "CRM",  "NEE",  "NFLX",
    "AMD", "INTC", "QCOM",  "GS",   "MS",   "BLK",  "SPGI", "CAT",
]

SCREEN_PERIOD  = "2y"    # data window for screening
TRAIN_PERIOD   = "3y"    # data window for model training
SCREEN_METRICS = ["sharpe", "momentum_6m", "volatility"]


def _download(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Download adjusted-close + volume for a list of tickers. Returns {ticker: df}."""
    print(f"  Downloading {len(tickers)} ticker(s) [{period}] …", end=" ", flush=True)
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    print("done.")
    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw[["Close", "Volume"]].copy()
            else:
                df = raw.xs(t, axis=1, level=1)[["Close", "Volume"]].copy()
            df = df.dropna()
            if len(df) > 60:
                result[t] = df
        except Exception:
            pass
    return result


def screen_stocks(top_n: int = 5) -> list[str]:
    """
    Score every ticker in UNIVERSE on three factors, then return the top_n
    by composite rank.

    Factors
    -------
    1. Annualised Sharpe ratio  (higher = better)
    2. 6-month price momentum   (higher = better)
    3. Realised volatility      (lower  = better — prefer stable stocks)
    """
    print("\n[SCREENER] Evaluating universe …")
    data = _download(UNIVERSE, SCREEN_PERIOD)
    rows = []
    for ticker, df in data.items():
        ret  = np.log(df["Close"]).diff().dropna()
        sharpe  = (ret.mean() / ret.std()) * np.sqrt(252)
        mom6m   = (df["Close"].iloc[-1] / df["Close"].iloc[-126]) - 1
        ann_vol = ret.std() * np.sqrt(252)
        rows.append({"ticker": ticker, "sharpe": sharpe,
                     "momentum_6m": mom6m, "volatility": ann_vol})

    scores = pd.DataFrame(rows).set_index("ticker")

    # Rank each metric (1 = best)
    scores["r_sharpe"]   = scores["sharpe"].rank(ascending=False)
    scores["r_mom6m"]    = scores["momentum_6m"].rank(ascending=False)
    scores["r_vol"]      = scores["volatility"].rank(ascending=True)   # low vol → good rank
    scores["composite"]  = scores[["r_sharpe", "r_mom6m", "r_vol"]].mean(axis=1)
    scores = scores.sort_values("composite")

    top = scores.head(top_n)
    print(f"\n  Top-{top_n} screened stocks:")
    print(f"  {'Ticker':<8} {'Sharpe':>8} {'Mom 6M':>8} {'Ann Vol':>8}")
    print(f"  {'-'*36}")
    for t, row in top.iterrows():
        print(f"  {t:<8} {row['sharpe']:>8.2f} {row['momentum_6m']:>7.1%} {row['volatility']:>7.1%}")
    return list(top.index)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "Momentum-5d",
    "Price vs MA20",
    "Volume z-score",
    "Roll volatility",
    "RSI (14)",
]


def build_features(df: pd.DataFrame, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 5 technical features from a real OHLCV DataFrame and binary labels.

    Features
    --------
    1. 5-day log-return momentum
    2. Price deviation from 20-day MA (normalised)
    3. Volume z-score (10-day window)
    4. 10-day rolling return volatility
    5. Wilder RSI (14-period)

    Label : 1 if next-day log-return > 0, else 0
    """
    close  = df["Close"].values
    volume = df["Volume"].values
    ret    = np.diff(np.log(close))
    n      = len(ret)

    features, labels = [], []
    for t in range(lookback, n - 1):

        # 1. Momentum
        mom5 = ret[t - 5 : t].sum()

        # 2. Price vs 20-day MA
        ma20         = close[t - lookback : t].mean()
        price_vs_ma  = (close[t] - ma20) / (ma20 + 1e-8)

        # 3. Volume z-score
        vol_window   = volume[t - 10 : t].astype(float)
        vol_zscore   = (volume[t] - vol_window.mean()) / (vol_window.std() + 1e-8)

        # 4. Rolling volatility
        roll_vol = ret[t - 10 : t].std()

        # 5. Wilder RSI (14-period)
        changes  = ret[t - 14 : t]
        gains    = changes[changes > 0]
        losses   = np.abs(changes[changes < 0])
        avg_gain = gains.mean()  if len(gains)  > 0 else 1e-8
        avg_loss = losses.mean() if len(losses) > 0 else 1e-8
        rsi      = 100 - (100 / (1 + avg_gain / avg_loss))

        features.append([mom5, price_vs_ma, vol_zscore, roll_vol, rsi])
        labels.append(1.0 if ret[t + 1] > 0.0 else 0.0)

    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PERCEPTRON  (identical to Bilokon & Simonian 2025)
# ═════════════════════════════════════════════════════════════════════════════

class Perceptron:
    """
    Single-layer Perceptron with the Perceptron learning rule,
    exactly as presented in the CFA Institute Research Foundation chapter.
    """

    def __init__(self, dim: int, learning_rate: float = 0.005, epochs: int = 200):
        self.dim            = dim
        self.learning_rate  = learning_rate
        self.epochs         = epochs
        self.weights        = np.random.normal(size=(dim, 1))
        self.bias           = float(np.random.normal())
        self.loss_history: list[float] = []

    def _step(self, v: float) -> float:
        return 0.0 if v < 0.0 else 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                xc     = xi.reshape(-1, 1)
                v      = float((self.weights.T @ xc).item()) + self.bias
                y_hat  = self._step(v)
                if target != y_hat:
                    delta         = y_hat - target
                    self.bias    -= self.learning_rate * delta
                    self.weights -= self.learning_rate * delta * xc
                    errors       += 1
            self.loss_history.append(errors / len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for xi in X:
            xc = xi.reshape(-1, 1)
            v  = float((self.weights.T @ xc).item()) + self.bias
            preds.append(self._step(v))
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SINGLE-STOCK PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_stock(ticker: str, df: pd.DataFrame, ax_curve=None) -> dict:
    """Train + evaluate a Perceptron on one stock. Returns a results dict."""
    X, y = build_features(df)
    if len(y) < 100:
        return {}

    split = int(len(y) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    model = Perceptron(dim=X.shape[1], learning_rate=0.005, epochs=200)
    model.fit(X_tr, y_tr)

    y_pred     = model.predict(X_te)
    test_acc   = model.score(X_te, y_te)
    baseline   = float(y_te.mean())          # always-predict-up baseline

    if ax_curve is not None:
        ax_curve.plot(model.loss_history, linewidth=1.4, label=ticker)

    return {
        "ticker"    : ticker,
        "samples"   : len(y),
        "test_acc"  : test_acc,
        "baseline"  : max(baseline, 1 - baseline),
        "edge"      : test_acc - max(baseline, 1 - baseline),
        "weights"   : model.weights.flatten(),
        "bias"      : model.bias,
        "y_te"      : y_te,
        "y_pred"    : y_pred,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Perceptron stock-return predictor")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker symbol (e.g. AAPL). Skips screener.")
    parser.add_argument("--top",    type=int, default=5,
                        help="Number of top stocks to screen (default 5).")
    args = parser.parse_args()

    print("=" * 62)
    print("  Perceptron — Real-Data Stock Return Direction Prediction")
    print("  Bilokon & Simonian (2025) | CFA Research Foundation")
    print("=" * 62)

    # ── Select tickers ────────────────────────────────────────────────────────
    if args.ticker:
        tickers = [args.ticker.upper()]
        print(f"\n  Single-stock mode: {tickers[0]}")
    else:
        tickers = screen_stocks(top_n=args.top)

    # ── Download training data ────────────────────────────────────────────────
    print(f"\n[DATA] Downloading {TRAIN_PERIOD} of OHLCV for selected tickers …")
    data = _download(tickers, TRAIN_PERIOD)

    # ── Train & evaluate ──────────────────────────────────────────────────────
    print(f"\n[MODEL] Training Perceptron on {len(data)} stock(s) …\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax_curve, ax_bar = axes

    results = []
    for ticker, df in data.items():
        res = run_stock(ticker, df, ax_curve=ax_curve)
        if res:
            results.append(res)
            print(f"  {ticker:<6}  acc={res['test_acc']:.2%}  "
                  f"baseline={res['baseline']:.2%}  "
                  f"edge={res['edge']:+.2%}  "
                  f"n={res['samples']}")

    if not results:
        print("No valid results. Check tickers / internet connection.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["edge"])
    print(f"\n  Best stock : {best['ticker']}  "
          f"edge={best['edge']:+.2%} over naive baseline")

    # ── Feature importance (best stock) ──────────────────────────────────────
    print(f"\n  Feature weights for {best['ticker']}:")
    for name, w in zip(FEATURE_NAMES, best["weights"]):
        bar = "█" * int(abs(w) * 200)
        sign = "+" if w >= 0 else "-"
        print(f"    {name:<20} {sign}{abs(w):.4f}  {bar}")

    # ── Classification report (best stock) ───────────────────────────────────
    print(f"\n  Classification Report — {best['ticker']} (test set)")
    print(classification_report(
        best["y_te"], best["y_pred"],
        labels=[0.0, 1.0],
        target_names=["Down (0)", "Up (1)"],
        zero_division=0,
    ))

    # ── Plot 1 : training curves ──────────────────────────────────────────────
    ax_curve.set_title("Training Curve — Misclassification Rate", fontsize=11)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Error Rate")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)

    # ── Plot 2 : accuracy vs baseline bar chart ───────────────────────────────
    labels_bar = [r["ticker"] for r in results]
    acc_vals   = [r["test_acc"]  for r in results]
    base_vals  = [r["baseline"]  for r in results]
    x = np.arange(len(labels_bar))
    w = 0.35
    ax_bar.bar(x - w/2, acc_vals,  w, label="Perceptron",    color="#4C72B0")
    ax_bar.bar(x + w/2, base_vals, w, label="Naive baseline", color="#DD8452", alpha=0.7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_bar, fontsize=9)
    ax_bar.set_ylim(0.3, 0.8)
    ax_bar.set_title("Test Accuracy vs Baseline", fontsize=11)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle("Perceptron — Stock Return Direction  |  CFA / Bilokon & Simonian (2025)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out_path = "perceptron_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
