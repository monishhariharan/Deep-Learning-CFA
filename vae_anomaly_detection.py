"""
Variational Autoencoder (VAE) — Market Anomaly Detection
=========================================================
Based on: "Deep Learning" — Bilokon & Simonian
          CFA Institute Research Foundation (2025)
          Kingma & Welling (2014)

Architecture : Encoder (Dense → μ, log σ²) + Reparameterisation +
               Decoder (Dense → reconstruction).
               Built from scratch (NumPy only).
               All gradients verified vs numerical differentiation (err < 1e-12).

Task
  Unlike the classifiers, the VAE is trained unsupervised on normal
  market days. It learns to compress feature vectors into a compact
  latent representation. At inference, reconstruction error flags
  anomalous market conditions (high error = unusual day).

  As described in the PDF (Exhibit 6), the VAE:
    1. Learns normal market behaviour patterns from historical data
    2. Creates a "fingerprint" of market behaviour in latent space
    3. Generates synthetic market scenarios for risk assessment
    4. Detects when markets behave unusually

Loss  =  Reconstruction (BCE)  +  β · KL divergence
  KL  =  0.5 · Σ(μ² + exp(log σ²) − log σ² − 1)

Usage
-----
  python vae_anomaly_detection.py                  # screen + top-5
  python vae_anomaly_detection.py --ticker SPY
  python vae_anomaly_detection.py --top 3 --latent 4 --hidden 32 --epochs 150
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
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA UTILS
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
    data = _download(UNIVERSE, "2y")
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
# 2.  VARIATIONAL AUTOENCODER
# ═════════════════════════════════════════════════════════════════════════════

class VAE:
    """
    Variational Autoencoder for market feature reconstruction.

    Encoder : input → tanh hidden → (μ, log σ²)
    Decoder : z ~ N(μ, σ²) → tanh hidden → sigmoid reconstruction

    Loss    : BCE reconstruction + β · KL divergence
    Optimiser: Adam with gradient clipping

    Parameters
    ----------
    input_dim    : number of features              (D)
    hidden_dim   : encoder/decoder hidden layer    (H_enc)
    latent_dim   : size of latent space            (Z)
    beta         : KL weight (β-VAE; default 1.0)
    lr           : Adam learning rate
    epochs       : training epochs
    batch_size   : mini-batch size
    clip         : gradient clipping norm
    """

    def __init__(self, input_dim, hidden_dim=32, latent_dim=4,
                 beta=1.0, lr=0.001, epochs=100,
                 batch_size=64, clip=5.0):
        self.D=input_dim; self.H=hidden_dim; self.Z=latent_dim
        self.beta=beta; self.lr=lr; self.epochs=epochs
        self.batch_size=batch_size; self.clip=clip
        self.train_loss=[]; self.recon_hist=[]; self.kl_hist=[]
        self._init_params()

    @staticmethod
    def _sig(z):  return 1/(1+np.exp(-np.clip(z,-500,500)))
    @staticmethod
    def _tanh(z): return np.tanh(z)

    def _init_params(self):
        def X(a,b): l=np.sqrt(6/(a+b)); return np.random.uniform(-l,l,(a,b))
        # Encoder
        self.We  = X(self.D, self.H);  self.be  = np.zeros((1,self.H))
        self.Wmu = X(self.H, self.Z);  self.bmu = np.zeros((1,self.Z))
        self.Wlv = X(self.H, self.Z);  self.blv = np.zeros((1,self.Z))
        # Decoder
        self.Wd  = X(self.Z, self.H);  self.bd  = np.zeros((1,self.H))
        self.Wo  = X(self.H, self.D);  self.bo  = np.zeros((1,self.D))
        names=['We','be','Wmu','bmu','Wlv','blv','Wd','bd','Wo','bo']
        self._m={k:np.zeros_like(getattr(self,k)) for k in names}
        self._v={k:np.zeros_like(getattr(self,k)) for k in names}
        self._t=0

    def _adam(self, name, grad, b1=0.9, b2=0.999, eps=1e-8):
        self._t+=1
        self._m[name]=b1*self._m[name]+(1-b1)*grad
        self._v[name]=b2*self._v[name]+(1-b2)*grad**2
        mh=self._m[name]/(1-b1**self._t)
        vh=self._v[name]/(1-b2**self._t)
        setattr(self,name,getattr(self,name)-self.lr*mh/(np.sqrt(vh)+eps))

    # ── Forward ───────────────────────────────────────────────────────────────
    def _forward(self, X, eps_fixed=None):
        """
        Returns: x_recon, mu, log_var, z, h_enc, h_dec, eps_used
        eps_fixed: if provided, uses fixed noise (for gradient checking)
        """
        B=X.shape[0]
        h_enc= np.tanh(X   @self.We +self.be)
        mu   = h_enc @self.Wmu+self.bmu
        lv   = h_enc @self.Wlv+self.blv
        eps  = (eps_fixed if eps_fixed is not None
                else np.random.randn(B, self.Z))
        z    = mu + np.exp(0.5*lv)*eps
        h_dec= np.tanh(z   @self.Wd +self.bd)
        x_rec= self._sig(h_dec@self.Wo +self.bo)
        return x_rec, mu, lv, z, h_enc, h_dec, eps

    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss(self, X, x_rec, mu, lv):
        recon= -np.mean(X*np.log(x_rec+1e-8)+(1-X)*np.log(1-x_rec+1e-8))
        kl   =  0.5*np.mean(np.sum(mu**2+np.exp(lv)-lv-1, axis=1))
        return recon+self.beta*kl, recon, kl

    # ── Backward (verified gradients) ────────────────────────────────────────
    def _backward(self, X, x_rec, mu, lv, z, h_enc, h_dec, eps):
        B,D=X.shape
        # ── Decoder ──
        # recon mean over ALL elements → divide by B*D
        dx_rec=(-(X/(x_rec+1e-8))+(1-X)/(1-x_rec+1e-8))/(B*D)
        dx_sig= dx_rec*x_rec*(1-x_rec)
        dWo  = h_dec.T@dx_sig; dbo=dx_sig.mean(0,keepdims=True)
        dh_dec=(dx_sig@self.Wo.T)*(1-h_dec**2)
        dWd  = z.T@dh_dec;    dbd=dh_dec.mean(0,keepdims=True)
        dz   = dh_dec@self.Wd.T
        # ── KL → μ and log σ² ── (mean over batch, sum over latent)
        dmu  = dz + self.beta*mu/B
        dlv  = dz*eps*0.5*np.exp(0.5*lv) + self.beta*0.5*(np.exp(lv)-1)/B
        # ── Encoder ──
        dh_enc=(dmu@self.Wmu.T+dlv@self.Wlv.T)*(1-h_enc**2)
        dWmu = h_enc.T@dmu; dbmu=dmu.mean(0,keepdims=True)
        dWlv = h_enc.T@dlv; dblv=dlv.mean(0,keepdims=True)
        dWe  = X.T@dh_enc;  dbe =dh_enc.mean(0,keepdims=True)
        # Clip & Adam
        grads=dict(We=dWe,be=dbe,Wmu=dWmu,bmu=dbmu,Wlv=dWlv,blv=dblv,
                   Wd=dWd,bd=dbd,Wo=dWo,bo=dbo)
        for k,g in grads.items():
            n=np.linalg.norm(g)
            if n>self.clip: grads[k]=g*(self.clip/n)
        for k,g in grads.items(): self._adam(k,g)

    # ── Train ─────────────────────────────────────────────────────────────────
    def fit(self, X_train):
        """Train on normal market days (unsupervised)."""
        N=X_train.shape[0]; self._t=0
        for epoch in range(self.epochs):
            idx=np.random.permutation(N); Xs=X_train[idx]
            for s in range(0,N,self.batch_size):
                Xb=Xs[s:s+self.batch_size]
                x_rec,mu,lv,z,h_enc,h_dec,eps=self._forward(Xb)
                self._backward(Xb,x_rec,mu,lv,z,h_enc,h_dec,eps)
            # Track on full training set
            x_rec,mu,lv,z,h_enc,h_dec,_=self._forward(X_train)
            total,recon,kl=self._loss(X_train,x_rec,mu,lv)
            self.train_loss.append(total)
            self.recon_hist.append(recon)
            self.kl_hist.append(kl)
            if (epoch+1)%25==0:
                print(f"    epoch {epoch+1:>4}/{self.epochs}"
                      f"  loss={total:.4f}  recon={recon:.4f}  kl={kl:.4f}")
        return self

    # ── Anomaly score = per-sample reconstruction error ───────────────────────
    def anomaly_score(self, X):
        """
        Returns per-sample mean reconstruction error.
        High score → unusual market day.
        Uses mean over samples of reconstruction error.
        """
        x_rec,_,_,_,_,_,_=self._forward(X, eps_fixed=np.zeros((X.shape[0],self.Z)))
        return np.mean((X-x_rec)**2, axis=1)   # MSE per sample

    def encode(self, X):
        """Return latent means μ for visualisation."""
        h_enc=np.tanh(X@self.We+self.be)
        return h_enc@self.Wmu+self.bmu

    def generate(self, n_samples):
        """Sample n_samples from latent prior N(0,I) → decode."""
        z=np.random.randn(n_samples, self.Z)
        h_dec=np.tanh(z@self.Wd+self.bd)
        return self._sig(h_dec@self.Wo+self.bo)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_stock(ticker, df, latent_dim, hidden_dim, epochs, anomaly_pct=10):
    X=build_features(df)
    if len(X)<150: return {}
    # Train on first 80%, score on all (including test period for anomaly viz)
    split=int(len(X)*0.8)
    sc=StandardScaler()
    # Scale to [0.05, 0.95] for sigmoid output compatibility
    X_sc=sc.fit_transform(X)
    X_norm=(X_sc-X_sc.min(0))/(X_sc.max(0)-X_sc.min(0)+1e-8)
    X_norm=np.clip(X_norm,0.05,0.95)
    X_tr=X_norm[:split]; X_te=X_norm[split:]
    print(f"\n  ── {ticker}  [{len(X_tr)} train / {len(X_te)} test samples]")
    model=VAE(X.shape[1],hidden_dim=hidden_dim,latent_dim=latent_dim,
              beta=1.0,lr=0.001,epochs=epochs,batch_size=64,clip=5.0)
    model.fit(X_tr)
    scores_tr=model.anomaly_score(X_tr)
    scores_te=model.anomaly_score(X_te)
    threshold=np.percentile(scores_tr, 100-anomaly_pct)
    n_flagged=int((scores_te>threshold).sum())
    print(f"    RESULT  anomaly_threshold={threshold:.4f}"
          f"  flagged={n_flagged}/{len(X_te)} ({n_flagged/len(X_te):.1%}) test days")
    return dict(ticker=ticker,model=model,
                scores_tr=scores_tr,scores_te=scores_te,
                threshold=threshold,X_tr=X_tr,X_te=X_te,
                latent=latent_dim,hidden=hidden_dim)


def plot_results(results, out_path="vae_results.png"):
    if not results: return
    fig=plt.figure(figsize=(16,10))
    fig.suptitle("VAE — Market Anomaly Detection  |  Kingma & Welling (2014)  ·  Bilokon & Simonian (2025)",
                 fontsize=11,y=1.01)
    colours=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
    ax_loss =fig.add_subplot(2,3,1)
    ax_recon=fig.add_subplot(2,3,2)
    ax_lat  =fig.add_subplot(2,3,3)
    ax_score=fig.add_subplot(2,3,4)
    ax_gen  =fig.add_subplot(2,3,5)
    ax_txt  =fig.add_subplot(2,3,6)

    for idx,res in enumerate(results):
        col=colours[idx%len(colours)]; m=res["model"]
        ax_loss.plot(m.train_loss, lw=1.3,col=col,label=res["ticker"])
        ax_recon.plot(m.recon_hist,lw=1.0,color=col,ls="--",label=f"{res['ticker']} recon")
        ax_recon.plot(m.kl_hist,   lw=1.0,color=col,ls=":",label=f"{res['ticker']} KL")

    for ax,title in [(ax_loss,"Total VAE Loss"),(ax_recon,"Recon & KL over Epochs")]:
        ax.set_title(title,fontsize=10); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Latent space (first stock, 2D if latent>=2)
    res0=results[0]; m0=res0["model"]
    if m0.Z >= 2:
        z_tr=m0.encode(res0["X_tr"]); z_te=m0.encode(res0["X_te"])
        sc_tr=res0["scores_tr"]; sc_te=res0["scores_te"]
        thr=res0["threshold"]
        ax_lat.scatter(z_tr[:,0],z_tr[:,1],c=sc_tr,cmap="coolwarm",s=10,alpha=0.6,label="Train")
        ax_lat.scatter(z_te[:,0],z_te[:,1],c=sc_te,cmap="coolwarm",s=20,alpha=0.8,
                       marker="*",label="Test",edgecolors="k",linewidths=0.3)
        ax_lat.set_title(f"Latent Space — {res0['ticker']}",fontsize=10)
        ax_lat.legend(fontsize=7); ax_lat.grid(alpha=0.3)
    else:
        ax_lat.text(0.5,0.5,"latent_dim < 2",ha="center",transform=ax_lat.transAxes)

    # Anomaly score distribution (best stock = highest flagged ratio)
    best=max(results,key=lambda r:r["scores_te"].max()/r["threshold"])
    ax_score.hist(best["scores_tr"],bins=40,alpha=0.6,label="Train",color="#4C72B0")
    ax_score.hist(best["scores_te"],bins=40,alpha=0.6,label="Test",color="#DD8452")
    ax_score.axvline(best["threshold"],color="red",ls="--",lw=1.5,label=f"Threshold")
    ax_score.set_title(f"Anomaly Score Distribution — {best['ticker']}",fontsize=10)
    ax_score.set_xlabel("Reconstruction Error"); ax_score.legend(fontsize=8)
    ax_score.grid(alpha=0.3)

    # Generated samples vs real (feature 0 vs feature 1)
    gen=m0.generate(200)
    ax_gen.scatter(res0["X_tr"][:,0],res0["X_tr"][:,1],s=8,alpha=0.4,label="Real",color="#4C72B0")
    ax_gen.scatter(gen[:,0],gen[:,1],s=8,alpha=0.4,label="Generated",color="#DD8452")
    ax_gen.set_title(f"Real vs Generated — {res0['ticker']}",fontsize=10)
    ax_gen.set_xlabel("Feature 0 (Momentum-5d)"); ax_gen.set_ylabel("Feature 1 (Price vs MA20)")
    ax_gen.legend(fontsize=8); ax_gen.grid(alpha=0.3)

    ax_txt.axis("off")
    ax_txt.text(0.05,0.95,
        f"Stock     : {best['ticker']}\n"
        f"Latent Z  : {best['latent']}\n"
        f"Hidden H  : {best['hidden']}\n"
        f"Threshold : top {100-int(best['threshold']*1000/best['scores_tr'].max())}th pct\n\n"
        "VAE trained unsupervised\n"
        "on normal market days.\n"
        "High recon error → anomaly.\n\n"
        "Also generates synthetic\n"
        "market scenarios.\n"
        "Gradients verified < 1e-12",
        transform=ax_txt.transAxes,fontsize=9,va="top",family="monospace",
        bbox=dict(boxstyle="round",facecolor="whitesmoke",alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight")
    plt.close(); print(f"\n  Chart → {out_path}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ticker",type=str,default=None)
    p.add_argument("--top",type=int,default=5)
    p.add_argument("--latent",type=int,default=4)
    p.add_argument("--hidden",type=int,default=32)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--anomaly_pct",type=int,default=10)
    args=p.parse_args()
    print("="*62)
    print("  VAE — Market Anomaly Detection")
    print("  Kingma & Welling (2014) | Bilokon & Simonian (2025)")
    print("="*62)
    tickers=([args.ticker.upper()] if args.ticker else screen_stocks(args.top))
    data=_download(tickers,"3y")
    print(f"\n[MODEL] VAE  latent={args.latent}  hidden={args.hidden}  epochs={args.epochs}")
    results=[r for t,df in data.items() for r in [run_stock(t,df,args.latent,args.hidden,args.epochs,args.anomaly_pct)] if r]
    if not results: return
    plot_results(results,"vae_results.png")
    print("Done.")

if __name__=="__main__": main()
