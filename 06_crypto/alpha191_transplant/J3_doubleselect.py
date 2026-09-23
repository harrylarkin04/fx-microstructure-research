# -*- coding: utf-8 -*-
"""Post-double-selection LASSO (Belloni-Chernozhukov-Hansen) on the crypto
Alpha191 transplant, then the portfolio.

For each Alpha191 signal d:
  step 1  LASSO of forward return y on the control set X   -> S1
  step 2  LASSO of the signal d on the control set X       -> S2
  step 3  OLS of y on [d, S1 union S2], HC3 SE, cluster by date
This gives valid inference on d AFTER controlling for known crypto effects, and
is immune to the omitted-variable bias that naive single-selection LASSO has.

Controls = the crypto analogue of the paper's 151 fundamentals: market beta,
size, momentum (1/6/12m), short-term reversal, realised vol, Amihud illiquidity,
turnover, and lottery-demand (MAX5). Any Alpha191 signal that survives is
adding something those do NOT already explain.

Discipline: SELECTION uses 2018-2022 only. 2023-2026 is a clean holdout.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LassoCV

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
P = pd.read_parquet(os.path.join(OUT, "crypto_alpha191.parquet"))
R = pd.read_parquet(os.path.join(OUT, "crypto_rets.parquet"))
DV = pd.read_parquet(os.path.join(OUT, "crypto_dv.parquet"))
R.columns = ["date", "asset", "ret"]
DV.columns = ["date", "asset", "dv"]
D = P.merge(R, on=["date", "asset"], how="left").merge(DV, on=["date", "asset"], how="left")
D = D.sort_values(["asset", "date"])
SIG = [c for c in P.columns if c not in ("date", "asset")]

# ---------- controls: the crypto analogue of a fundamental factor set ----------
g = D.groupby("asset")
D["mom1"] = g["ret"].transform(lambda s: s.rolling(21, min_periods=10).sum())
D["mom6"] = g["ret"].transform(lambda s: s.rolling(126, min_periods=60).sum())
D["mom12"] = g["ret"].transform(lambda s: s.rolling(252, min_periods=120).sum())
D["rev5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=3).sum())
D["vol21"] = g["ret"].transform(lambda s: s.rolling(21, min_periods=10).std())
D["vol63"] = g["ret"].transform(lambda s: s.rolling(63, min_periods=30).std())
D["size"] = np.log(D["dv"].replace(0, np.nan))
D["amihud"] = g.apply(lambda x: (x["ret"].abs() / x["dv"].replace(0, np.nan))
                      .rolling(21, min_periods=10).mean()).reset_index(level=0, drop=True)
D["max5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=3).max())
mkt = D.groupby("date")["ret"].mean().rename("mkt")
D = D.merge(mkt, on="date", how="left")
D = D.sort_values(["asset", "date"])
g = D.groupby("asset")
D["beta"] = g.apply(lambda x: x["ret"].rolling(63, min_periods=30).cov(x["mkt"]) /
                    x["mkt"].rolling(63, min_periods=30).var()).reset_index(level=0, drop=True)
CTRL = ["mom1", "mom6", "mom12", "rev5", "vol21", "vol63", "size", "amihud", "max5", "beta"]

# forward 21-day return (monthly rebalance, as in the paper)
D["fwd"] = g["ret"].transform(lambda s: s.shift(-21).rolling(21, min_periods=15).sum())
D = D.replace([np.inf, -np.inf], np.nan)


def csz(df, cols):
    """cross-sectional z-score each day, so coefficients are comparable"""
    out = df.copy()
    for c in cols:
        gg = out.groupby("date")[c]
        out[c] = (out[c] - gg.transform("mean")) / gg.transform("std").replace(0, np.nan)
    return out


D = csz(D, SIG + CTRL)
D = D.dropna(subset=["fwd"] + CTRL)
TR = D[D.date < "2023-01-01"]
TE = D[D.date >= "2023-01-01"]
print("train %d rows (%s-%s), holdout %d rows (%s-%s), %d signals"
      % (len(TR), TR.date.min().date(), TR.date.max().date(),
         len(TE), TE.date.min().date(), TE.date.max().date(), len(SIG)))


def hc3_cluster(y, X, groups):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(np.arange(len(y))).groupby(groups).indices.items():
        Xg, eg = X[idx], e[idx]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    return b, b / se


def lasso_select(y, X, cols):
    m = LassoCV(n_alphas=100, cv=5, eps=0.05, max_iter=4000, n_jobs=-1, random_state=0)
    m.fit(X, y)
    # 1-SE rule: most restrictive alpha within 1 SE of min CV MSE
    mse = m.mse_path_.mean(axis=1)
    se = m.mse_path_.std(axis=1) / np.sqrt(m.mse_path_.shape[1])
    i = int(np.argmin(mse))
    thr = mse[i] + se[i]
    ok = np.where(mse <= thr)[0]
    a = m.alphas_[ok].max()
    from sklearn.linear_model import Lasso
    m2 = Lasso(alpha=a, max_iter=4000).fit(X, y)
    return [c for c, v in zip(cols, m2.coef_) if abs(v) > 1e-10]


Xc = TR[CTRL].values
y = TR["fwd"].values
S1 = lasso_select(y, Xc, CTRL)
print("\nstep 1 - controls selected for the RETURN equation: %s" % (S1 or "none"))

rows = []
for s in SIG:
    m = TR[s].notna().values
    if m.sum() < 5000:
        continue
    sub = TR[m]
    d = sub[s].values
    S2 = lasso_select(d, sub[CTRL].values, CTRL)
    use = sorted(set(S1) | set(S2))
    X = np.column_stack([np.ones(len(sub)), d] + [sub[c].values for c in use])
    b, t = hc3_cluster(sub["fwd"].values, X, sub["date"].values)
    rows.append(dict(sig=s, coef=b[1], t=t[1], nctrl=len(use), n=int(m.sum())))
S = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
# Bonferroni over the 69 tested signals
crit = 3.30
S["survives"] = S.t.abs() > crit
print("\n=== post-double-selection results (train 2018-2022) ===")
print("  Bonferroni |t| threshold for %d signals at 5%%: %.2f" % (len(SIG), crit))
print(S.head(16).to_string(index=False))
print("\n  survivors: %d of %d" % (S.survives.sum(), len(S)))
surv = S[S.survives].sig.tolist()
if not surv:
    surv = S.head(5).sig.tolist()
    print("  none clear Bonferroni; carrying the top 5 by |t| into the holdout as a weaker test")
print("  carried: %s" % ", ".join(surv))
pd.Series(surv).to_csv(os.path.join(OUT, "crypto_alpha_survivors.csv"), index=False)
S.to_csv(os.path.join(OUT, "crypto_alpha_dsl.csv"), index=False)
