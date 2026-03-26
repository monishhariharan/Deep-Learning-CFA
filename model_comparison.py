"""
Model Comparison — All Classifiers Head-to-Head
================================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)

Runs all four from-scratch classifiers on the SAME screened stocks
and produces a unified 6-panel comparison dashboard + equity curves.

Models compared
---------------
  1. Perceptron   — single-layer, step activation     (Bilokon §3)
  2. FFNN         — [7→64→32→1], backprop             (Rumelhart 1986)
  3. LSTM         — forget/input/output gates, BPTT   (Hochreiter 1997)
  4. GRU          — update/reset gates, BPTT          (Cho et al. 2014)

Outputs
-------
  · model_comparison.png  — 6-panel dashboard
  · model_backtests.png   — equity curves + Sharpe / max drawdown
  · Console table: accuracy, AUC, edge per model per stock

Usage
-----
  python model_comparison.py                    # screen + top-5
  python model_comparison.py --ticker AAPL
  python model_comparison.py --top 3 --epochs 100
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
    sys.exit("pip install yfinance")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  SHARED DATA UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "JPM","V","UNH","XOM","JNJ","MA","PG","HD",
    "MRK","ABBV","LLY","AVGO","COST","AMD","QCOM",
    "GS","MS","BLK","NFLX","CRM","BAC",
]


def _download(tickers, period):
    print(f"  Downloading {len(tickers)} ticker(s) [{period}] …", end=" ", flush=True)
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    print("done.")
    out = {}
    for t in tickers:
        try:
            df = (raw[["Close", "Volume"]] if len(tickers) == 1
                  else raw.xs(t, axis=1, level=1)[["Close", "Volume"]])
            df = df.dropna()
            if len(df) > 80:
                out[t] = df
        except Exception:
            pass
    return out


def screen_stocks(top_n=5):
    print("\n[SCREENER] Evaluating universe …")
    data = _download(UNIVERSE, "2y")
    rows = []
    for t, df in data.items():
        ret = np.log(df["Close"]).diff().dropna()
        rows.append(dict(ticker=t,
                         sharpe=(ret.mean() / ret.std()) * np.sqrt(252),
                         momentum_6m=(df["Close"].iloc[-1] / df["Close"].iloc[-126]) - 1,
                         volatility=ret.std() * np.sqrt(252)))
    sc = pd.DataFrame(rows).set_index("ticker")
    for col, asc in [("sharpe", False), ("momentum_6m", False), ("volatility", True)]:
        sc[f"r_{col}"] = sc[col].rank(ascending=asc)
    sc["composite"] = sc[["r_sharpe", "r_momentum_6m", "r_volatility"]].mean(1)
    top = sc.sort_values("composite").head(top_n)
    print(f"\n  Top-{top_n}: " + "  ".join(top.index.tolist()))
    return list(top.index)


def build_features(df, lookback=20):
    close  = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    ret    = np.diff(np.log(close))
    n      = len(ret)
    feats, labels = [], []
    for t in range(lookback, n - 1):
        ma20  = close[t - lookback:t].mean()
        std20 = close[t - lookback:t].std()
        vw    = volume[t - 10:t]
        ch    = ret[t - 14:t]
        g     = ch[ch > 0]; lo = np.abs(ch[ch < 0])
        ag    = g.mean()  if len(g)  > 0 else 1e-8
        al    = lo.mean() if len(lo) > 0 else 1e-8
        ub, lb = ma20 + 2 * std20, ma20 - 2 * std20
        feats.append([
            ret[t - 5:t].sum(),
            (close[t] - ma20) / (ma20 + 1e-8),
            (volume[t] - vw.mean()) / (vw.std() + 1e-8),
            ret[t - 10:t].std(),
            100 - (100 / (1 + ag / al)),
            ret[t - 10:t].sum(),
            (close[t] - lb) / (ub - lb + 1e-8),
        ])
        labels.append(1.0 if ret[t + 1] > 0.0 else 0.0)
    return np.array(feats, dtype=np.float64), np.array(labels, dtype=np.float64)


def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for t in range(seq_len, len(y)):
        Xs.append(X[t - seq_len:t])
        ys.append(y[t])
    return np.array(Xs), np.array(ys)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PERCEPTRON  (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

class Perceptron:
    def __init__(self, n_features=5, lr=0.005, epochs=200):
        self.w  = np.zeros(n_features)
        self.b  = 0.0
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        # Use only first 5 features to match original perceptron
        X5 = X[:, :5]
        for _ in range(self.epochs):
            for xi, yi in zip(X5, y):
                pred = 1.0 if (np.dot(self.w, xi) + self.b) >= 0 else 0.0
                err  = yi - pred
                self.w += self.lr * err * xi
                self.b += self.lr * err
        return self

    def predict(self, X):
        X5 = X[:, :5]
        return (X5 @ self.w + self.b >= 0).astype(float)

    def predict_proba(self, X):
        # Soft score via sigmoid of linear activation
        X5 = X[:, :5]
        z  = X5 @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FFNN  (from scratch — copied core from ffnn_stock_prediction.py)
# ═════════════════════════════════════════════════════════════════════════════

class FFNN:
    def __init__(self, layer_sizes, lr=0.01, momentum=0.9,
                 epochs=150, batch_size=32):
        self.lr        = lr
        self.momentum  = momentum
        self.epochs    = epochs
        self.batch_size= batch_size
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            lim = np.sqrt(6.0 / (fi + fo))
            self.W.append(np.random.uniform(-lim, lim, (fi, fo)))
            self.b.append(np.zeros((1, fo)))
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    @staticmethod
    def _bce(yt, yp): return -float(np.mean(yt*np.log(yp+1e-8)+(1-yt)*np.log(1-yp+1e-8)))

    def _forward(self, X):
        acts = [X]
        for W, b in zip(self.W, self.b):
            acts.append(self._sig(acts[-1] @ W + b))
        return acts

    def _backprop(self, acts, y):
        m = y.shape[0]
        delta = acts[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.W))):
            dW = (acts[i].T @ delta) / m
            db = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] * (1 - acts[i]))
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    def fit(self, X, y, **_):
        m = X.shape[0]
        for _ in range(self.epochs):
            idx = np.random.permutation(m)
            Xs, ys = X[idx], y[idx]
            for s in range(0, m, self.batch_size):
                acts = self._forward(Xs[s:s+self.batch_size])
                self._backprop(acts, ys[s:s+self.batch_size])
        return self

    def predict_proba(self, X): return self._forward(X)[-1].flatten()
    def predict(self, X):       return (self.predict_proba(X) >= 0.5).astype(float)
    def score(self, X, y):      return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  LSTM  (from scratch — core from lstm_stock_prediction.py)
# ═════════════════════════════════════════════════════════════════════════════

class LSTM:
    def __init__(self, input_size, hidden_size, seq_len,
                 lr=0.005, epochs=80, batch_size=32, clip=5.0):
        self.F=input_size; self.H=hidden_size; self.T=seq_len
        self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.clip=clip
        self._init_params()

    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

    def _init_params(self):
        n=self.F+self.H; H=self.H
        def X(a,b): l=np.sqrt(6/(a+b)); return np.random.uniform(-l,l,(a,b))
        self.Wf=X(n,H); self.bf=np.zeros((1,H))
        self.Wi=X(n,H); self.bi=np.zeros((1,H))
        self.Wg=X(n,H); self.bg=np.zeros((1,H))
        self.Wo=X(n,H); self.bo=np.zeros((1,H))
        self.Wy=X(H,1); self.by=np.zeros((1,1))
        names=['Wf','Wi','Wg','Wo','bf','bi','bg','bo','Wy','by']
        self._m={k:np.zeros_like(getattr(self,k)) for k in names}
        self._v={k:np.zeros_like(getattr(self,k)) for k in names}
        self._t=0

    def _adam(self, name, grad):
        self._t+=1; b1,b2,eps=0.9,0.999,1e-8
        self._m[name]=b1*self._m[name]+(1-b1)*grad
        self._v[name]=b2*self._v[name]+(1-b2)*grad**2
        mh=self._m[name]/(1-b1**self._t)
        vh=self._v[name]/(1-b2**self._t)
        setattr(self,name,getattr(self,name)-self.lr*mh/(np.sqrt(vh)+eps))

    def _forward_batch(self, X_batch):
        B=X_batch.shape[0]
        h=np.zeros((B,self.H)); c=np.zeros((B,self.H)); cache=[]
        for t in range(self.T):
            x=X_batch[:,t,:]; xh=np.concatenate([x,h],1)
            f=self._sig(xh@self.Wf+self.bf); i=self._sig(xh@self.Wi+self.bi)
            g=np.tanh(xh@self.Wg+self.bg);  o=self._sig(xh@self.Wo+self.bo)
            c=f*c+i*g; ht=o*np.tanh(c)
            cache.append(dict(x=x,h_prev=h,f=f,i=i,g=g,o=o,xh=xh,ht=ht,c=c))
            h=ht
        return self._sig(h@self.Wy+self.by), cache, h

    def _bptt(self, y_hat, y, cache):
        B=y.shape[0]
        dy=(y_hat-y.reshape(-1,1))/B
        h_last=cache[-1]["ht"]
        dWy=h_last.T@dy; dby=dy.mean(0,keepdims=True)
        dh_out=dy@self.Wy.T
        dWf=np.zeros_like(self.Wf); dbf=np.zeros_like(self.bf)
        dWi=np.zeros_like(self.Wi); dbi=np.zeros_like(self.bi)
        dWg=np.zeros_like(self.Wg); dbg=np.zeros_like(self.bg)
        dWo=np.zeros_like(self.Wo); dbo=np.zeros_like(self.bo)
        dh_next=np.zeros((B,self.H)); dc_next=np.zeros((B,self.H))
        for t in reversed(range(self.T)):
            s=cache[t]; f,i,g,o=s["f"],s["i"],s["g"],s["o"]
            c_t=s["c"]; xh=s["xh"]
            c_prev=cache[t-1]["c"] if t>0 else np.zeros((B,self.H))
            dh_t=(dh_out if t==self.T-1 else 0.0)+dh_next
            do=dh_t*np.tanh(c_t)
            dc=dh_t*o*(1-np.tanh(c_t)**2)+dc_next
            df=dc*c_prev; di=dc*g; dg=dc*i
            df_pre=df*f*(1-f); di_pre=di*i*(1-i)
            dg_pre=dg*(1-g**2);  do_pre=do*o*(1-o)
            dWf+=xh.T@df_pre; dbf+=df_pre.mean(0,keepdims=True)
            dWi+=xh.T@di_pre; dbi+=di_pre.mean(0,keepdims=True)
            dWg+=xh.T@dg_pre; dbg+=dg_pre.mean(0,keepdims=True)
            dWo+=xh.T@do_pre; dbo+=do_pre.mean(0,keepdims=True)
            dxh=(df_pre@self.Wf.T+di_pre@self.Wi.T+dg_pre@self.Wg.T+do_pre@self.Wo.T)
            dh_next=dxh[:,self.F:]; dc_next=dc*f
        grads=dict(Wf=dWf,bf=dbf,Wi=dWi,bi=dbi,Wg=dWg,bg=dbg,Wo=dWo,bo=dbo,Wy=dWy,by=dby)
        for k,g in grads.items():
            n=np.linalg.norm(g)
            if n>self.clip: grads[k]=g*(self.clip/n)
        for k,g in grads.items(): self._adam(k,g)

    def fit(self, X, y, **_):
        N=X.shape[0]; self._t=0
        for _ in range(self.epochs):
            idx=np.random.permutation(N); Xs,ys=X[idx],y[idx]
            for s in range(0,N,self.batch_size):
                yh,cache,_=self._forward_batch(Xs[s:s+self.batch_size])
                self._bptt(yh,ys[s:s+self.batch_size],cache)
        return self

    def predict_proba(self,X): return self._forward_batch(X)[0].flatten()
    def predict(self,X):       return (self.predict_proba(X)>=0.5).astype(float)
    def score(self,X,y):       return float(np.mean(self.predict(X)==y))


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GRU  (from scratch — core from gru_stock_prediction.py)
# ═════════════════════════════════════════════════════════════════════════════

class GRU:
    def __init__(self, input_size, hidden_size, seq_len,
                 lr=0.005, epochs=80, batch_size=32, clip=5.0):
        self.F=input_size; self.H=hidden_size; self.T=seq_len
        self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.clip=clip
        self._init_params()

    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

    def _init_params(self):
        n=self.F+self.H; H=self.H
        def X(a,b): l=np.sqrt(6/(a+b)); return np.random.uniform(-l,l,(a,b))
        self.Wz=X(n,H); self.bz=np.zeros((1,H))
        self.Wr=X(n,H); self.br=np.zeros((1,H))
        self.Wh=X(n,H); self.bh=np.zeros((1,H))
        self.Wy=X(H,1); self.by=np.zeros((1,1))
        names=['Wz','bz','Wr','br','Wh','bh','Wy','by']
        self._m={k:np.zeros_like(getattr(self,k)) for k in names}
        self._v={k:np.zeros_like(getattr(self,k)) for k in names}
        self._t=0

    def _adam(self,name,grad):
        self._t+=1; b1,b2,eps=0.9,0.999,1e-8
        self._m[name]=b1*self._m[name]+(1-b1)*grad
        self._v[name]=b2*self._v[name]+(1-b2)*grad**2
        mh=self._m[name]/(1-b1**self._t)
        vh=self._v[name]/(1-b2**self._t)
        setattr(self,name,getattr(self,name)-self.lr*mh/(np.sqrt(vh)+eps))

    def _forward(self, X_batch):
        B=X_batch.shape[0]; h=np.zeros((B,self.H)); cache=[]
        for t in range(self.T):
            x=X_batch[:,t,:]; xh=np.concatenate([x,h],1)
            z=self._sig(xh@self.Wz+self.bz); r=self._sig(xh@self.Wr+self.br)
            xrh=np.concatenate([x,r*h],1); hc=np.tanh(xrh@self.Wh+self.bh)
            hn=(1-z)*h+z*hc
            cache.append(dict(h_prev=h,z=z,r=r,hc=hc,xh=xh,xrh=xrh,hn=hn))
            h=hn
        return self._sig(h@self.Wy+self.by), cache, h

    def _bptt(self,y_hat,y,cache):
        B=y.shape[0]; dy=(y_hat-y.reshape(-1,1))/B
        h_last=cache[-1]["hn"]
        dWy=h_last.T@dy; dby=dy.mean(0,keepdims=True); dh_out=dy@self.Wy.T
        dWz=np.zeros_like(self.Wz); dbz=np.zeros_like(self.bz)
        dWr=np.zeros_like(self.Wr); dbr=np.zeros_like(self.br)
        dWh=np.zeros_like(self.Wh); dbh=np.zeros_like(self.bh)
        dh_next=np.zeros((B,self.H))
        for t in reversed(range(self.T)):
            s=cache[t]; h_prev,z,r,hc,xh,xrh=s["h_prev"],s["z"],s["r"],s["hc"],s["xh"],s["xrh"]
            dh=(dh_out if t==self.T-1 else 0)+dh_next
            dhc=dh*z; dz=dh*(hc-h_prev)
            dz_pre=dz*z*(1-z); dhc_pre=dhc*(1-hc**2)
            dxrh=dhc_pre@self.Wh.T
            dr=dxrh[:,self.F:]*h_prev; dr_pre=dr*r*(1-r)
            dWh+=xrh.T@dhc_pre; dbh+=dhc_pre.mean(0,keepdims=True)
            dWz+=xh.T@dz_pre;   dbz+=dz_pre.mean(0,keepdims=True)
            dWr+=xh.T@dr_pre;   dbr+=dr_pre.mean(0,keepdims=True)
            dxh_z=dz_pre@self.Wz.T; dxh_r=dr_pre@self.Wr.T
            dh_next=(dh*(1-z)+dxh_z[:,self.F:]+dxh_r[:,self.F:]+dxrh[:,self.F:]*r)
        grads=dict(Wz=dWz,bz=dbz,Wr=dWr,br=dbr,Wh=dWh,bh=dbh,Wy=dWy,by=dby)
        for k,g in grads.items():
            n=np.linalg.norm(g)
            if n>self.clip: grads[k]=g*(self.clip/n)
        for k,g in grads.items(): self._adam(k,g)

    def fit(self,X,y,**_):
        N=X.shape[0]; self._t=0
        for _ in range(self.epochs):
            idx=np.random.permutation(N); Xs,ys=X[idx],y[idx]
            for s in range(0,N,self.batch_size):
                yh,cache,_=self._forward(Xs[s:s+self.batch_size])
                self._bptt(yh,ys[s:s+self.batch_size],cache)
        return self

    def predict_proba(self,X): return self._forward(X)[0].flatten()
    def predict(self,X):       return (self.predict_proba(X)>=0.5).astype(float)
    def score(self,X,y):       return float(np.mean(self.predict(X)==y))


# ═════════════════════════════════════════════════════════════════════════════
# 6.  SINGLE-STOCK RUNNER  — returns metrics for all 4 models
# ═════════════════════════════════════════════════════════════════════════════

SEQ_LEN    = 20   # timesteps for LSTM / GRU
LOOKBACK   = 20   # feature lookback window
HIDDEN     = 32   # recurrent hidden units
FFNN_ARCH  = [7, 64, 32, 1]


def run_stock(ticker, df, epochs):
    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close))
    # Align future returns with labels produced by build_features:
    # label at t uses return at t+1, with LOOKBACK warmup.
    future_ret = log_ret[LOOKBACK + 1 :]

    X_flat, y_flat = build_features(df, lookback=LOOKBACK)
    if len(y_flat) < SEQ_LEN + 120:
        print(f"  {ticker}: not enough data — skipping.")
        return None

    # ── Flat split (Perceptron, FFNN) ────────────────────────────────────────
    split_f = int(len(y_flat) * 0.8)
    X_tr_f, X_te_f = X_flat[:split_f], X_flat[split_f:]
    y_tr_f, y_te_f = y_flat[:split_f], y_flat[split_f:]
    ret_te_f       = future_ret[split_f:]
    sc = StandardScaler()
    X_tr_fs = sc.fit_transform(X_tr_f)
    X_te_fs  = sc.transform(X_te_f)

    # ── Sequence split (LSTM, GRU) ───────────────────────────────────────────
    X_seq, y_seq = build_sequences(X_flat, y_flat, SEQ_LEN)
    split_s = int(len(y_seq) * 0.8)
    X_tr_s, X_te_s = X_seq[:split_s], X_seq[split_s:]
    y_tr_s, y_te_s = y_seq[:split_s], y_seq[split_s:]
    ret_seq        = future_ret[SEQ_LEN:]
    ret_te_s       = ret_seq[split_s:]
    sc2 = StandardScaler()
    N, T, Fv = X_tr_s.shape
    X_tr_ss  = sc2.fit_transform(X_tr_s.reshape(-1, Fv)).reshape(N, T, Fv)
    X_te_ss  = sc2.transform(X_te_s.reshape(-1, Fv)).reshape(X_te_s.shape[0], T, Fv)

    baseline_f = max(float(y_te_f.mean()), 1 - float(y_te_f.mean()))
    baseline_s = max(float(y_te_s.mean()), 1 - float(y_te_s.mean()))

    def _metrics(model, X_te, y_te):
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)
        acc    = float(np.mean(y_pred == y_te))
        try:   auc = roc_auc_score(y_te, y_prob)
        except: auc = float("nan")
        return acc, auc, y_pred, y_prob

    results = {}

    # ── Perceptron ────────────────────────────────────────────────────────────
    m_p = Perceptron(n_features=5, lr=0.005, epochs=min(epochs, 200))
    m_p.fit(X_tr_fs, y_tr_f)
    acc, auc, pred, prob = _metrics(m_p, X_te_fs, y_te_f)
    results["Perceptron"] = dict(acc=acc, auc=auc, edge=acc-baseline_f,
                                  y_te=y_te_f, y_pred=pred, y_prob=prob,
                                  backtest=None)  # placeholder, filled below

    # ── FFNN ─────────────────────────────────────────────────────────────────
    m_f = FFNN(FFNN_ARCH, lr=0.01, momentum=0.9, epochs=epochs, batch_size=32)
    m_f.fit(X_tr_fs, y_tr_f)
    acc, auc, pred, prob = _metrics(m_f, X_te_fs, y_te_f)
    results["FFNN"] = dict(acc=acc, auc=auc, edge=acc-baseline_f,
                            y_te=y_te_f, y_pred=pred, y_prob=prob,
                            backtest=None)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    m_l = LSTM(Fv, HIDDEN, SEQ_LEN, lr=0.005, epochs=epochs, batch_size=32, clip=5.0)
    m_l.fit(X_tr_ss, y_tr_s)
    acc, auc, pred, prob = _metrics(m_l, X_te_ss, y_te_s)
    results["LSTM"] = dict(acc=acc, auc=auc, edge=acc-baseline_s,
                            y_te=y_te_s, y_pred=pred, y_prob=prob,
                            backtest=None)

    # ── GRU ──────────────────────────────────────────────────────────────────
    m_g = GRU(Fv, HIDDEN, SEQ_LEN, lr=0.005, epochs=epochs, batch_size=32, clip=5.0)
    m_g.fit(X_tr_ss, y_tr_s)
    acc, auc, pred, prob = _metrics(m_g, X_te_ss, y_te_s)
    results["GRU"] = dict(acc=acc, auc=auc, edge=acc-baseline_s,
                           y_te=y_te_s, y_pred=pred, y_prob=prob,
                           backtest=None)

    # ── Backtests (long/flat vs buy-and-hold) ────────────────────────────────
    results["Perceptron"]["backtest"] = backtest_metrics(results["Perceptron"]["y_pred"], ret_te_f)
    results["FFNN"]["backtest"]       = backtest_metrics(results["FFNN"]["y_pred"], ret_te_f)
    results["LSTM"]["backtest"]       = backtest_metrics(results["LSTM"]["y_pred"], ret_te_s)
    results["GRU"]["backtest"]        = backtest_metrics(results["GRU"]["y_pred"], ret_te_s)

    return dict(ticker=ticker, results=results,
                baseline_f=baseline_f, baseline_s=baseline_s)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  EQUITY CURVE  — simple signal-based backtest (long/flat)
# ═════════════════════════════════════════════════════════════════════════════

def equity_curve(y_pred, y_true_ret):
    """
    Long next day when model predicts Up (1), flat otherwise.
    y_true_ret: actual log-returns of the test period.
    Returns cumulative equity series.
    """
    n = min(len(y_pred), len(y_true_ret))
    strategy = y_pred[:n] * y_true_ret[:n]   # only take long positions
    return np.exp(np.cumsum(strategy))


def buy_hold_curve(y_true_ret):
    return np.exp(np.cumsum(y_true_ret))


def sharpe_ratio(returns):
    if len(returns) == 0:
        return float("nan")
    vol = returns.std()
    if vol < 1e-12:
        return 0.0
    return float((returns.mean() / vol) * np.sqrt(252))


def max_drawdown(equity):
    if len(equity) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(dd.min())


def backtest_metrics(y_pred, y_true_ret):
    """
    Compute long/flat strategy metrics against buy-and-hold.
    """
    n = min(len(y_pred), len(y_true_ret))
    if n == 0:
        return dict(
            returns=np.array([]),
            benchmark_returns=np.array([]),
            equity=np.array([]),
            buy_hold=np.array([]),
            sharpe=float("nan"),
            max_drawdown=float("nan"),
            final=1.0,
            benchmark_final=1.0,
        )
    strategy_ret = y_pred[:n] * y_true_ret[:n]
    equity = np.exp(np.cumsum(strategy_ret))
    buy_hold = np.exp(np.cumsum(y_true_ret[:n]))
    return dict(
        returns=strategy_ret,
        benchmark_returns=y_true_ret[:n],
        equity=equity,
        buy_hold=buy_hold,
        sharpe=sharpe_ratio(strategy_ret),
        max_drawdown=max_drawdown(equity),
        final=float(equity[-1]),
        benchmark_final=float(buy_hold[-1]),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

MODEL_NAMES  = ["Perceptron", "FFNN", "LSTM", "GRU"]
MODEL_COLORS = ["#55A868", "#4C72B0", "#C44E52", "#DD8452"]


def plot_comparison(all_stocks, out_path="model_comparison.png"):
    if not all_stocks:
        return

    tickers = [s["ticker"] for s in all_stocks]
    n_stocks = len(tickers)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Model Comparison — Perceptron vs FFNN vs LSTM vs GRU\n"
        "Bilokon & Simonian (2025) | CFA Institute Research Foundation",
        fontsize=12, y=1.01,
    )

    ax_acc  = fig.add_subplot(2, 3, 1)   # Accuracy by model × stock
    ax_auc  = fig.add_subplot(2, 3, 2)   # AUC by model × stock
    ax_edge = fig.add_subplot(2, 3, 3)   # Edge by model × stock
    ax_heat = fig.add_subplot(2, 3, 4)   # Heatmap: best model per stock
    ax_rank = fig.add_subplot(2, 3, 5)   # Average rank per model
    ax_txt  = fig.add_subplot(2, 3, 6)   # Summary table

    x    = np.arange(n_stocks)
    w    = 0.18
    offs = [-1.5, -0.5, 0.5, 1.5]

    # Build matrix: shape (4 models × n_stocks)
    acc_mat  = np.zeros((4, n_stocks))
    auc_mat  = np.zeros((4, n_stocks))
    edge_mat = np.zeros((4, n_stocks))

    for si, stock in enumerate(all_stocks):
        for mi, name in enumerate(MODEL_NAMES):
            r = stock["results"][name]
            acc_mat[mi, si]  = r["acc"]
            auc_mat[mi, si]  = r["auc"] if not np.isnan(r["auc"]) else 0.5
            edge_mat[mi, si] = r["edge"]

    # ── Accuracy bars ──────────────────────────────────────────────────────
    for mi, (name, col) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        ax_acc.bar(x + offs[mi]*w, acc_mat[mi], w, label=name,
                   color=col, alpha=0.85, edgecolor="white")
    ax_acc.set_xticks(x); ax_acc.set_xticklabels(tickers, fontsize=9)
    ax_acc.set_ylim(0.3, 0.8); ax_acc.set_title("Test Accuracy", fontsize=11)
    ax_acc.set_ylabel("Accuracy"); ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", alpha=0.3)

    # ── AUC bars ───────────────────────────────────────────────────────────
    for mi, (name, col) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        ax_auc.bar(x + offs[mi]*w, auc_mat[mi], w, label=name,
                   color=col, alpha=0.85, edgecolor="white")
    ax_auc.axhline(0.5, color="red", ls="--", lw=1, label="Random")
    ax_auc.set_xticks(x); ax_auc.set_xticklabels(tickers, fontsize=9)
    ax_auc.set_ylim(0.3, 1.0); ax_auc.set_title("ROC-AUC", fontsize=11)
    ax_auc.set_ylabel("AUC"); ax_auc.legend(fontsize=8)
    ax_auc.grid(axis="y", alpha=0.3)

    # ── Edge bars ──────────────────────────────────────────────────────────
    for mi, (name, col) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        ax_edge.bar(x + offs[mi]*w, edge_mat[mi], w, label=name,
                    color=col, alpha=0.85, edgecolor="white")
    ax_edge.axhline(0, color="gray", lw=0.8)
    ax_edge.set_xticks(x); ax_edge.set_xticklabels(tickers, fontsize=9)
    ax_edge.set_title("Edge vs Baseline", fontsize=11)
    ax_edge.set_ylabel("Acc − Baseline"); ax_edge.legend(fontsize=8)
    ax_edge.grid(axis="y", alpha=0.3)

    # ── Heatmap: edge per model × stock ────────────────────────────────────
    im = ax_heat.imshow(edge_mat, cmap="RdYlGn", aspect="auto",
                         vmin=-0.15, vmax=0.15)
    ax_heat.set_xticks(range(n_stocks)); ax_heat.set_xticklabels(tickers, fontsize=9)
    ax_heat.set_yticks(range(4));        ax_heat.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax_heat.set_title("Edge Heatmap (green = above baseline)", fontsize=10)
    for mi in range(4):
        for si in range(n_stocks):
            ax_heat.text(si, mi, f"{edge_mat[mi,si]:+.1%}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(edge_mat[mi,si]) > 0.05 else "black")
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # ── Average rank per model (lower = better) ────────────────────────────
    avg_acc  = acc_mat.mean(axis=1)
    avg_auc  = auc_mat.mean(axis=1)
    avg_edge = edge_mat.mean(axis=1)
    yy = np.arange(4)
    ax_rank.barh(yy, avg_edge, color=MODEL_COLORS, alpha=0.85, edgecolor="white")
    ax_rank.set_yticks(yy); ax_rank.set_yticklabels(MODEL_NAMES, fontsize=10)
    ax_rank.axvline(0, color="gray", lw=0.8)
    ax_rank.set_title("Average Edge (all stocks)", fontsize=11)
    ax_rank.set_xlabel("Mean Edge vs Baseline")
    ax_rank.grid(axis="x", alpha=0.3)
    for yi, v in enumerate(avg_edge):
        ax_rank.text(v + (0.002 if v >= 0 else -0.002), yi, f"{v:+.2%}",
                     va="center", ha="left" if v >= 0 else "right", fontsize=9)

    # ── Summary text ───────────────────────────────────────────────────────
    ax_txt.axis("off")
    best_model = MODEL_NAMES[int(np.argmax(avg_edge))]
    best_stock_overall = max(
        [(s["ticker"], max(s["results"].values(), key=lambda r: r["edge"]))
         for s in all_stocks],
        key=lambda x: x[1]["edge"]
    )
    summary = (
        f"{'Model':<12} {'Avg Acc':>8} {'Avg AUC':>8} {'Avg Edge':>9}\n"
        f"{'─'*40}\n"
    )
    for mi, name in enumerate(MODEL_NAMES):
        summary += (f"{name:<12} {avg_acc[mi]:>8.2%} "
                    f"{avg_auc[mi]:>8.3f} {avg_edge[mi]:>+9.2%}\n")
    summary += (
        f"\n{'─'*40}\n"
        f"Best model   : {best_model}\n"
        f"Best stock   : {best_stock_overall[0]}\n"
        f"Best edge    : {best_stock_overall[1]['edge']:+.2%}\n\n"
        f"All models: NumPy from scratch\n"
        f"Seq models: T={SEQ_LEN} days, H={HIDDEN} units\n"
        f"FFNN arch : {FFNN_ARCH}"
    )
    ax_txt.text(0.03, 0.97, summary, transform=ax_txt.transAxes,
                fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Dashboard saved → {out_path}")


def plot_backtests(all_stocks, out_path="model_backtests.png"):
    if not all_stocks:
        return

    fig, (ax_eq, ax_txt) = plt.subplots(1, 2, figsize=(14, 5))
    ax_eq.set_title("Average Equity Curves (long/flat signal)")
    ax_eq.set_xlabel("Test days")
    ax_eq.set_ylabel("Cumulative equity (×)")

    agg_metrics = {}
    avg_bh_curve = None

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        eqs, bhs, sharpes, mdds, finals = [], [], [], [], []
        for stock in all_stocks:
            bt = stock["results"][name].get("backtest")
            if not bt or len(bt["equity"]) == 0:
                continue
            eqs.append(bt["equity"])
            bhs.append(bt["buy_hold"])
            sharpes.append(bt["sharpe"])
            mdds.append(bt["max_drawdown"])
            finals.append(bt["final"])
        if not eqs:
            continue
        min_len = min(len(e) for e in eqs)
        mean_eq = np.mean([e[:min_len] for e in eqs], axis=0)
        mean_bh = np.mean([b[:min_len] for b in bhs], axis=0)
        if avg_bh_curve is None:
            avg_bh_curve = mean_bh
        ax_eq.plot(mean_eq, label=name, color=color, lw=2)
        agg_metrics[name] = dict(
            sharpe=np.nanmean(sharpes) if sharpes else float("nan"),
            mdd=np.nanmean(mdds) if mdds else float("nan"),
            final=np.nanmean(finals) if finals else float("nan"),
            bh_final=float(mean_bh[-1]) if len(mean_bh) else float("nan"),
        )

    if avg_bh_curve is not None:
        ax_eq.plot(avg_bh_curve, label="Buy & Hold", color="black",
                   ls="--", lw=1.8)
    ax_eq.legend(fontsize=9)
    ax_eq.grid(alpha=0.3)

    ax_txt.axis("off")
    lines = [
        f"{'Model':<12} {'Sharpe':>8} {'MaxDD':>9} {'Final Eq':>10} {'B&H Eq':>10}",
        "─" * 52,
    ]
    for name in MODEL_NAMES:
        if name not in agg_metrics:
            continue
        m = agg_metrics[name]
        lines.append(
            f"{name:<12} "
            f"{m['sharpe']:>8.2f} "
            f"{m['mdd']:>+9.2%} "
            f"{m['final']:>10.3f} "
            f"{m['bh_final']:>10.3f}"
        )
    ax_txt.text(0.02, 0.98, "\n".join(lines), va="top",
                family="monospace", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Backtest saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare all 4 DL classifiers")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--top",    type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80,
                        help="Epochs for FFNN / LSTM / GRU (default 80)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Model Comparison: Perceptron | FFNN | LSTM | GRU")
    print("  Bilokon & Simonian (2025) | CFA Research Foundation")
    print("=" * 64)

    tickers = ([args.ticker.upper()] if args.ticker
               else screen_stocks(top_n=args.top))

    print(f"\n[DATA] Downloading 3y OHLCV …")
    data = _download(tickers, "3y")

    print(f"\n[MODEL] Running 4 models × {len(data)} stock(s)  epochs={args.epochs}\n")
    all_stocks = []
    for ticker, df in data.items():
        print(f"\n── {ticker} ──")
        res = run_stock(ticker, df, epochs=args.epochs)
        if res:
            all_stocks.append(res)
            # Console table
            print(f"  {'Model':<12} {'Acc':>7} {'AUC':>7} {'Edge':>8}")
            print(f"  {'─'*38}")
            for name in MODEL_NAMES:
                r = res["results"][name]
                print(f"  {name:<12} {r['acc']:>7.2%} "
                      f"{r['auc']:>7.3f} {r['edge']:>+8.2%}")

    if not all_stocks:
        print("No valid results.")
        return

    # ── Overall summary ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  SUMMARY — averaged over {len(all_stocks)} stock(s)")
    print("=" * 64)
    print(f"  {'Model':<12} {'Avg Acc':>8} {'Avg AUC':>8} {'Avg Edge':>9}")
    print(f"  {'─'*42}")
    for mi, name in enumerate(MODEL_NAMES):
        accs  = [s["results"][name]["acc"]  for s in all_stocks]
        aucs  = [s["results"][name]["auc"]  for s in all_stocks if not np.isnan(s["results"][name]["auc"])]
        edges = [s["results"][name]["edge"] for s in all_stocks]
        print(f"  {name:<12} {np.mean(accs):>8.2%} "
              f"{np.mean(aucs):>8.3f} {np.mean(edges):>+9.2%}")

    print("\n[BACKTEST] Long/flat signal vs buy-and-hold (test period)")
    print(f"  {'Model':<12} {'Sharpe':>8} {'MaxDD':>9} {'Final Eq':>10} {'B&H Eq':>10}")
    print(f"  {'─'*52}")
    for name in MODEL_NAMES:
        sharpes, mdds, finals, bh_finals = [], [], [], []
        for s in all_stocks:
            bt = s["results"][name].get("backtest")
            if not bt or len(bt["equity"]) == 0:
                continue
            sharpes.append(bt["sharpe"])
            mdds.append(bt["max_drawdown"])
            finals.append(bt["final"])
            bh_finals.append(bt["benchmark_final"])
        if not sharpes:
            continue
        print(f"  {name:<12} {np.nanmean(sharpes):>8.2f} "
              f"{np.nanmean(mdds):>+9.2%} {np.nanmean(finals):>10.3f} "
              f"{np.nanmean(bh_finals):>10.3f}")

    plot_comparison(all_stocks, out_path="model_comparison.png")
    plot_backtests(all_stocks, out_path="model_backtests.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
