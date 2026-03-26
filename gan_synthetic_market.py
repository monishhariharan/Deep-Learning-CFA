"""
Generative Adversarial Network (GAN) — Synthetic Market Data
=============================================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)
          Goodfellow et al. (2014)

Architecture : Generator (Dense, tanh) vs Discriminator (Dense, sigmoid)
               Built from scratch (NumPy only).
               Both networks trained via minimax backprop.

How it works (from PDF Exhibit 7):
  Generator   : Takes random noise z ~ N(0,I) → fake feature vector
  Discriminator: Real data or Generator output → P(real)
  Training    : Competitive game —
                  D  maximises log P(real) + log(1 − P(fake))
                  G  minimises log(1 − P(fake))  [equiv. max log P(fake)]
  Convergence : G learns to produce samples indistinguishable from real.

Application in finance (as noted in PDF):
  · Augment small datasets with realistic synthetic samples
  · Stress-test portfolios with plausible-but-rare market scenarios
  · Generate training data for other models when real data is scarce

Usage
-----
  python gan_synthetic_market.py                  # screen + top-5
  python gan_synthetic_market.py --ticker NVDA
  python gan_synthetic_market.py --top 3 --latent 16 --hidden 64 --epochs 300
"""

import argparse, warnings, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA UTILS  (same across all models)
# ═════════════════════════════════════════════════════════════════════════════

UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "JPM","V","UNH","XOM","JNJ","MA","PG","HD",
    "MRK","ABBV","LLY","AVGO","COST","AMD","QCOM",
    "GS","MS","BLK","NFLX","CRM","BAC",
]

def _download(tickers, period):
    print(f"  Downloading {len(tickers)} ticker(s) [{period}] …", end=" ", flush=True)
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    print("done.")
    out = {}
    for t in tickers:
        try:
            df = (raw[["Close","Volume"]] if len(tickers)==1
                  else raw.xs(t,axis=1,level=1)[["Close","Volume"]])
            df = df.dropna()
            if len(df) > 80: out[t] = df
        except Exception: pass
    return out

def screen_stocks(top_n=5):
    print("\n[SCREENER] Evaluating universe …")
    data = _download(UNIVERSE,"2y")
    rows = []
    for t,df in data.items():
        ret = np.log(df["Close"]).diff().dropna()
        rows.append(dict(ticker=t,
                         sharpe=(ret.mean()/ret.std())*np.sqrt(252),
                         momentum_6m=(df["Close"].iloc[-1]/df["Close"].iloc[-126])-1,
                         volatility=ret.std()*np.sqrt(252)))
    sc = pd.DataFrame(rows).set_index("ticker")
    for col,asc in [("sharpe",False),("momentum_6m",False),("volatility",True)]:
        sc[f"r_{col}"] = sc[col].rank(ascending=asc)
    sc["composite"] = sc[["r_sharpe","r_momentum_6m","r_volatility"]].mean(1)
    top = sc.sort_values("composite").head(top_n)
    print(f"\n  Top-{top_n}:  {'Ticker':<8} {'Sharpe':>8} {'Mom6M':>8} {'Vol':>8}")
    print(f"  {'-'*38}")
    for t,r in top.iterrows():
        print(f"  {t:<8} {r['sharpe']:>8.2f} {r['momentum_6m']:>7.1%} {r['volatility']:>7.1%}")
    return list(top.index)

def build_features(df, lookback=20):
    close = df["Close"].values.astype(float)
    volume= df["Volume"].values.astype(float)
    ret   = np.diff(np.log(close)); n=len(ret)
    feats = []
    for t in range(lookback, n-1):
        ma20 = close[t-lookback:t].mean()
        std20= close[t-lookback:t].std()
        vw   = volume[t-10:t]
        ch   = ret[t-14:t]
        g    = ch[ch>0]; lo=np.abs(ch[ch<0])
        ag   = g.mean()  if len(g)>0  else 1e-8
        al   = lo.mean() if len(lo)>0 else 1e-8
        ub,lb= ma20+2*std20, ma20-2*std20
        feats.append([
            ret[t-5:t].sum(),
            (close[t]-ma20)/(ma20+1e-8),
            (volume[t]-vw.mean())/(vw.std()+1e-8),
            ret[t-10:t].std(),
            100-(100/(1+ag/al)),
            ret[t-10:t].sum(),
            (close[t]-lb)/(ub-lb+1e-8),
        ])
    return np.array(feats, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DENSE LAYER HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _xavier(fan_in, fan_out):
    lim = np.sqrt(6/(fan_in+fan_out))
    return np.random.uniform(-lim, lim, (fan_in, fan_out))

def _adam_state(shapes):
    return {k: np.zeros(s) for k,s in shapes.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GENERATOR  (noise z → fake feature vector)
# ═════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Noise z ~ N(0, I) → tanh hidden layers → linear output (no final activation).
    Output range matches StandardScaler-normalised real data.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, lr=0.0002):
        self.Z=latent_dim; self.H=hidden_dim; self.D=output_dim; self.lr=lr
        self.W1=_xavier(latent_dim,hidden_dim); self.b1=np.zeros((1,hidden_dim))
        self.W2=_xavier(hidden_dim,hidden_dim); self.b2=np.zeros((1,hidden_dim))
        self.W3=_xavier(hidden_dim,output_dim); self.b3=np.zeros((1,output_dim))
        names=['W1','b1','W2','b2','W3','b3']
        self._m={k:np.zeros_like(getattr(self,k)) for k in names}
        self._v={k:np.zeros_like(getattr(self,k)) for k in names}
        self._t=0

    def forward(self, z):
        self._h1=np.tanh(z  @self.W1+self.b1)
        self._h2=np.tanh(self._h1@self.W2+self.b2)
        self._out=self._h2@self.W3+self.b3   # linear output
        self._z=z
        return self._out

    def backward(self, dout):
        """
        dout: gradient from discriminator loss w.r.t. generator output.
        Backprop through 3-layer MLP.
        """
        B=dout.shape[0]
        dW3=self._h2.T@dout/B;    db3=dout.mean(0,keepdims=True)
        dh2=(dout@self.W3.T)*(1-self._h2**2)
        dW2=self._h1.T@dh2/B;     db2=dh2.mean(0,keepdims=True)
        dh1=(dh2@self.W2.T)*(1-self._h1**2)
        dW1=self._z.T@dh1/B;      db1=dh1.mean(0,keepdims=True)
        grads=dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3)
        self._adam_update(grads)

    def _adam_update(self,grads,b1=0.5,b2=0.999,eps=1e-8):
        self._t+=1
        for k,g in grads.items():
            n=np.linalg.norm(g)
            if n>5.0: g=g*(5.0/n)
            self._m[k]=b1*self._m[k]+(1-b1)*g
            self._v[k]=b2*self._v[k]+(1-b2)*g**2
            mh=self._m[k]/(1-b1**self._t); vh=self._v[k]/(1-b2**self._t)
            setattr(self,k,getattr(self,k)-self.lr*mh/(np.sqrt(vh)+eps))

    def sample(self, n):
        z=np.random.randn(n,self.Z)
        return self.forward(z)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DISCRIMINATOR  (feature vector → P(real))
# ═════════════════════════════════════════════════════════════════════════════

class Discriminator:
    """
    Feature vector → tanh hidden layers → sigmoid output P(real).
    """
    def __init__(self, input_dim, hidden_dim, lr=0.0002):
        self.D=input_dim; self.H=hidden_dim; self.lr=lr
        self.W1=_xavier(input_dim,hidden_dim);  self.b1=np.zeros((1,hidden_dim))
        self.W2=_xavier(hidden_dim,hidden_dim); self.b2=np.zeros((1,hidden_dim))
        self.W3=_xavier(hidden_dim,1);          self.b3=np.zeros((1,1))
        names=['W1','b1','W2','b2','W3','b3']
        self._m={k:np.zeros_like(getattr(self,k)) for k in names}
        self._v={k:np.zeros_like(getattr(self,k)) for k in names}
        self._t=0

    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

    def forward(self, x):
        self._h1=np.tanh(x         @self.W1+self.b1)
        self._h2=np.tanh(self._h1  @self.W2+self.b2)
        self._out=self._sig(self._h2@self.W3+self.b3)
        self._x=x
        return self._out

    def backward_real(self, y_real):
        """Train on real samples: maximise log D(x_real)."""
        B=y_real.shape[0]
        d_out=(y_real-self._out)/B       # gradient of -log D(x)
        self._backprop(d_out)
        return d_out@self.W3.T*self._h2*(1-self._h2)  # not used externally

    def backward_fake(self, y_fake_pred):
        """
        Train on fake samples: maximise log(1-D(G(z))).
        Returns gradient w.r.t. input x (= G output) for generator.
        """
        B=y_fake_pred.shape[0]
        # D loss on fakes: -log(1-D(G(z)))  → gradient: D(G(z))/(1-D(G(z))) → simplifies to:
        d_out=y_fake_pred/B              # ∂[-log(1-D)] / ∂D = 1/(1-D) · ... simplified
        self._backprop(-d_out)           # negative: D wants fake output to be 0
        dh2=(-d_out@self.W3.T)*(1-self._h2**2)
        dh1=(dh2@self.W2.T)*(1-self._h1**2)
        dx=dh1@self.W1.T
        return dx                        # pass to generator

    def _backprop(self, d_out):
        B=d_out.shape[0]
        dW3=self._h2.T@d_out/B; db3=d_out.mean(0,keepdims=True)
        dh2=(d_out@self.W3.T)*(1-self._h2**2)
        dW2=self._h1.T@dh2/B;  db2=dh2.mean(0,keepdims=True)
        dh1=(dh2@self.W2.T)*(1-self._h1**2)
        dW1=self._x.T@dh1/B;   db1=dh1.mean(0,keepdims=True)
        grads=dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3)
        self._adam_update(grads)

    def _adam_update(self,grads,b1=0.5,b2=0.999,eps=1e-8):
        self._t+=1
        for k,g in grads.items():
            n=np.linalg.norm(g)
            if n>5.0: g=g*(5.0/n)
            self._m[k]=b1*self._m[k]+(1-b1)*g
            self._v[k]=b2*self._v[k]+(1-b2)*g**2
            mh=self._m[k]/(1-b1**self._t); vh=self._v[k]/(1-b2**self._t)
            setattr(self,k,getattr(self,k)-self.lr*mh/(np.sqrt(vh)+eps))


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GAN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

class GAN:
    """
    Wrapper that orchestrates Generator vs Discriminator training.

    Each epoch:
      1. Train D on real batch (maximise log D(x_real))
      2. Train D on fake batch (maximise log(1-D(G(z))))
      3. Train G to fool D    (minimise log(1-D(G(z))))
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dim=64,
                 lr_g=0.0002, lr_d=0.0002, epochs=200, batch_size=64):
        self.G=Generator(latent_dim,hidden_dim,input_dim,lr=lr_g)
        self.D=Discriminator(input_dim,hidden_dim,lr=lr_d)
        self.latent_dim=latent_dim; self.epochs=epochs
        self.batch_size=batch_size
        self.d_loss_hist=[]; self.g_loss_hist=[]
        self.d_real_hist=[]; self.d_fake_hist=[]

    @staticmethod
    def _bce(y,p): return -np.mean(y*np.log(p+1e-8)+(1-y)*np.log(1-p+1e-8))

    def fit(self, X_real):
        N=X_real.shape[0]
        for epoch in range(self.epochs):
            d_losses=[]; g_losses=[]; dr_list=[]; df_list=[]
            idx=np.random.permutation(N)
            for s in range(0,N,self.batch_size):
                real_batch=X_real[idx[s:s+self.batch_size]]
                B=len(real_batch)
                z=np.random.randn(B,self.latent_dim)
                fake_batch=self.G.forward(z)

                # ── Train Discriminator ──
                d_real=self.D.forward(real_batch)
                self.D.backward_real(np.ones((B,1)))
                d_fake=self.D.forward(fake_batch)
                dx_fake=self.D.backward_fake(d_fake)   # also gets dx for G

                d_loss=self._bce(np.ones((B,1)),d_real)+self._bce(np.zeros((B,1)),d_fake)

                # ── Train Generator ──
                # Re-generate (different z) for generator update
                z2=np.random.randn(B,self.latent_dim)
                fake2=self.G.forward(z2)
                d_fake2=self.D.forward(fake2)
                # Generator gradient: maximise log D(G(z)) → d_out = -(1-D)/B
                g_grad=-(1-d_fake2)/B
                dh2=(g_grad@self.D.W3.T)*(1-self.D._h2**2)
                dh1=(dh2@self.D.W2.T)*(1-self.D._h1**2)
                dx_for_g=dh1@self.D.W1.T
                self.G.backward(dx_for_g)
                g_loss=self._bce(np.ones((B,1)),d_fake2)

                d_losses.append(d_loss); g_losses.append(g_loss)
                dr_list.append(float(d_real.mean())); df_list.append(float(d_fake.mean()))

            self.d_loss_hist.append(np.mean(d_losses))
            self.g_loss_hist.append(np.mean(g_losses))
            self.d_real_hist.append(np.mean(dr_list))
            self.d_fake_hist.append(np.mean(df_list))
            if (epoch+1)%50==0:
                print(f"    epoch {epoch+1:>4}/{self.epochs}"
                      f"  D_loss={self.d_loss_hist[-1]:.4f}"
                      f"  G_loss={self.g_loss_hist[-1]:.4f}"
                      f"  D(real)={self.d_real_hist[-1]:.3f}"
                      f"  D(fake)={self.d_fake_hist[-1]:.3f}")
        return self

    def generate(self, n): return self.G.sample(n)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_stock(ticker, df, latent_dim, hidden_dim, epochs):
    X=build_features(df)
    if len(X)<100: return {}
    sc=StandardScaler(); X_sc=sc.fit_transform(X)
    print(f"\n  ── {ticker}  [{len(X_sc)} training samples]")
    model=GAN(X.shape[1],latent_dim=latent_dim,hidden_dim=hidden_dim,
              lr_g=0.0002,lr_d=0.0002,epochs=epochs,batch_size=64)
    model.fit(X_sc)
    synth_sc=model.generate(len(X_sc))
    synth=sc.inverse_transform(synth_sc)
    # Quality: feature mean & std similarity
    real_mean=X.mean(0); synth_mean=synth.mean(0)
    real_std=X.std(0);   synth_std=synth.std(0)
    mean_err=np.abs(real_mean-synth_mean).mean()
    std_err =np.abs(real_std-synth_std).mean()
    print(f"    RESULT  mean_err={mean_err:.4f}  std_err={std_err:.4f}"
          f"  D(real)={model.d_real_hist[-1]:.3f}  D(fake)={model.d_fake_hist[-1]:.3f}")
    return dict(ticker=ticker,model=model,X_real=X,X_synth=synth,
                X_real_sc=X_sc,X_synth_sc=synth_sc,
                mean_err=mean_err,std_err=std_err,
                latent=latent_dim,hidden=hidden_dim)


def plot_results(results, out_path="gan_results.png"):
    if not results: return
    fig=plt.figure(figsize=(16,10))
    fig.suptitle("GAN — Synthetic Market Data  |  Goodfellow et al. (2014)  ·  Bilokon & Simonian (2025)",
                 fontsize=11,y=1.01)
    colours=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
    ax_loss=fig.add_subplot(2,3,1); ax_disc=fig.add_subplot(2,3,2)
    ax_f0  =fig.add_subplot(2,3,3); ax_f1  =fig.add_subplot(2,3,4)
    ax_cor =fig.add_subplot(2,3,5); ax_txt =fig.add_subplot(2,3,6)

    for idx,res in enumerate(results):
        col=colours[idx%len(colours)]; m=res["model"]
        ax_loss.plot(m.d_loss_hist,lw=1.3,color=col,label=f"{res['ticker']} D")
        ax_loss.plot(m.g_loss_hist,lw=1.0,ls="--",color=col,label=f"{res['ticker']} G")
        ax_disc.plot(m.d_real_hist,lw=1.3,color=col,label=f"{res['ticker']} real")
        ax_disc.plot(m.d_fake_hist,lw=1.0,ls="--",color=col,label=f"{res['ticker']} fake")

    for ax,title in [(ax_loss,"G & D Losses"),(ax_disc,"D(real) vs D(fake)")]:
        ax.set_title(title,fontsize=10); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax_disc.axhline(0.5,color="gray",ls=":",lw=1,label="Nash eq.")

    # Feature distribution comparison (best stock)
    best=min(results,key=lambda r:r["mean_err"])
    F_NAMES=["Mom-5d","P/MA20","Vol-Z","RollVol","RSI","Mom-10d","Boll%B"]
    for feat_idx,(ax,title) in enumerate([(ax_f0,F_NAMES[0]),(ax_f1,F_NAMES[1])]):
        ax.hist(best["X_real"][:,feat_idx],bins=40,alpha=0.6,label="Real",
                color="#4C72B0",density=True)
        ax.hist(best["X_synth"][:,feat_idx],bins=40,alpha=0.6,label="Synthetic",
                color="#DD8452",density=True)
        ax.set_title(f"Feature: {title} — {best['ticker']}",fontsize=10)
        ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Correlation heatmap: real vs synthetic
    corr_real=np.corrcoef(best["X_real"].T)
    corr_syn =np.corrcoef(best["X_synth"].T)
    diff_corr=np.abs(corr_real-corr_syn)
    im=ax_cor.imshow(diff_corr,cmap="Reds",vmin=0,vmax=0.5)
    ax_cor.set_title(f"|Corr(real) − Corr(synth)| — {best['ticker']}",fontsize=10)
    ax_cor.set_xticks(range(7)); ax_cor.set_yticks(range(7))
    ax_cor.set_xticklabels(F_NAMES,rotation=45,fontsize=7)
    ax_cor.set_yticklabels(F_NAMES,fontsize=7)
    plt.colorbar(im,ax=ax_cor,fraction=0.046,pad=0.04)

    ax_txt.axis("off")
    ax_txt.text(0.05,0.95,
        f"Best      : {best['ticker']}\n"
        f"Latent z  : {best['latent']}\n"
        f"Hidden H  : {best['hidden']}\n"
        f"Mean err  : {best['mean_err']:.4f}\n"
        f"Std  err  : {best['std_err']:.4f}\n"
        f"D(real)   : {best['model'].d_real_hist[-1]:.3f}\n"
        f"D(fake)   : {best['model'].d_fake_hist[-1]:.3f}\n\n"
        "Nash equilibrium:\n"
        "D(real) = D(fake) = 0.5\n\n"
        "Generator: noise → features\n"
        "Discriminator: real vs fake",
        transform=ax_txt.transAxes,fontsize=9,va="top",family="monospace",
        bbox=dict(boxstyle="round",facecolor="whitesmoke",alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight")
    plt.close(); print(f"\n  Chart → {out_path}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ticker",type=str,default=None)
    p.add_argument("--top",type=int,default=5)
    p.add_argument("--latent",type=int,default=16)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--epochs",type=int,default=200)
    args=p.parse_args()
    print("="*62)
    print("  GAN — Synthetic Market Data Generation")
    print("  Goodfellow et al. (2014) | Bilokon & Simonian (2025)")
    print("="*62)
    tickers=([args.ticker.upper()] if args.ticker else screen_stocks(args.top))
    data=_download(tickers,"3y")
    print(f"\n[MODEL] GAN  latent={args.latent}  hidden={args.hidden}  epochs={args.epochs}")
    results=[r for t,df in data.items() for r in [run_stock(t,df,args.latent,args.hidden,args.epochs)] if r]
    if not results: return
    plot_results(results,"gan_results.png")
    print("Done.")

if __name__=="__main__": main()
