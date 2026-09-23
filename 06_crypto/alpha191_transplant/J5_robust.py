# -*- coding: utf-8 -*-
"""Robustness on the Alpha191 crypto transplant.

Threats:
 1. a70/a95/a97/a132/a150 are all volume/amount-scale - probably ONE effect
    dressed as five. Drop the cluster and see what survives.
 2. survivorship: the universe is tickers that exist TODAY. Dead coins are
    missing. Test sensitivity by restricting to names alive from the start.
 3. placebo: shuffle the signal cross-sectionally each day.
 4. is it just short-vol / market beta in disguise?
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(4)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
P = pd.read_parquet(os.path.join(OUT, "crypto_alpha191.parquet"))
R = pd.read_parquet(os.path.join(OUT, "crypto_rets.parquet"))
DV = pd.read_parquet(os.path.join(OUT, "crypto_dv.parquet"))
R.columns = ["date", "asset", "ret"]
DV.columns = ["date", "asset", "dv"]
SURV = pd.read_csv(os.path.join(OUT, "crypto_alpha_survivors.csv")).iloc[:, 0].tolist()
DSL = pd.read_csv(os.path.join(OUT, "crypto_alpha_dsl.csv"))
sgn = {r.sig: np.sign(r.coef) for _, r in DSL.iterrows()}
VOLCLUSTER = ["a70", "a95", "a97", "a132", "a150"]

D = P.merge(R, on=["date", "asset"]).merge(DV, on=["date", "asset"])
D = D[D.dv > 5e6]
RET = D.pivot_table(index="date", columns="asset", values="ret")


def cz(x):
    z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-3, 3).where(x.notna().sum(axis=1) >= 8)


def compose(sigs, universe=None, shuffle=False):
    comp = None
    for s in sigs:
        pv = D.pivot_table(index="date", columns="asset", values=s)
        if universe is not None:
            pv = pv.reindex(columns=[c for c in pv.columns if c in universe])
        z = cz(pv) * sgn.get(s, 1.0)
        if shuffle:
            v = z.values.copy()
            for i in range(len(v)):
                m = ~np.isnan(v[i])
                if m.sum() > 2:
                    idx = np.where(m)[0]
                    v[i, idx] = rng.permutation(v[i, idx])
            z = pd.DataFrame(v, index=z.index, columns=z.columns)
        comp = z if comp is None else comp.add(z, fill_value=0.0)
    return comp / max(len(sigs), 1)


def book(sig, hold=42, cost=15.0, uni=None):
    rr = RET if uni is None else RET.reindex(columns=[c for c in RET.columns if c in uni])
    z = cz(sig)
    z = z.reindex(columns=rr.columns)
    w = z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0) * 2.0
    W = (sum(w.shift(k + 1) for k in range(hold)) / hold).fillna(0.0)
    net = (W * rr).sum(axis=1) - W.diff().abs().sum(axis=1) * cost / 1e4
    net = net.loc["2023-01-01":"2026-08-31"]
    return net[net.notna()]


def stat(r, lab):
    if len(r) < 60:
        print("  %-40s (insufficient)" % lab)
        return np.nan
    e = (1 + r).cumprod()
    dd = (e / e.cummax() - 1).min()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    sh = r.mean() / r.std() * np.sqrt(365)
    print("  %-40s Sh %5.2f  t %5.2f  CAGR %7.2f%%  maxDD %7.2f%%"
          % (lab, sh, r.mean() / r.std() * np.sqrt(len(r)),
             ((1 + r).prod() ** (1 / yrs) - 1) * 100, dd * 100))
    return sh


print("=== 1. is it one volume effect wearing five hats? (holdout, 42d, 15bp) ===")
base = stat(book(compose(SURV)), "all 17 survivors")
nonvol = [s for s in SURV if s not in VOLCLUSTER]
stat(book(compose(nonvol)), "12 survivors, volume cluster DROPPED")
stat(book(compose(VOLCLUSTER)), "the 5 volume-scale signals ALONE")
pv = {s: cz(D.pivot_table(index="date", columns="asset", values=s)) for s in VOLCLUSTER}
cc = pd.DataFrame({s: v.stack() for s, v in pv.items()}).corr()
print("  pairwise corr within the volume cluster: min %.2f  mean %.2f"
      % (cc.values[np.triu_indices(len(cc), 1)].min(),
         cc.values[np.triu_indices(len(cc), 1)].mean()))

print("\n=== 2. survivorship: restrict to names alive from 2018 ===")
first = D.groupby("asset").date.min()
old = set(first[first <= pd.Timestamp("2018-06-30")].index)
print("  names alive at the start: %d of %d" % (len(old), D.asset.nunique()))
stat(book(compose(SURV, universe=old), uni=old), "  all survivors, 2018-alive universe only")

print("\n=== 3. placebo: shuffle signals cross-sectionally ===")
ps = []
for _ in range(15):
    ps.append(stat_ := book(compose(SURV, shuffle=True)).pipe(
        lambda r: r.mean() / r.std() * np.sqrt(365)))
print("  real Sh %.3f   placebo mean %.3f  sd %.3f  max %.3f"
      % (base, np.mean(ps), np.std(ps), np.max(ps)))

print("\n=== 4. is it market beta or short-vol in disguise? ===")
r = book(compose(SURV))
mkt = RET.mean(axis=1).reindex(r.index)
A = pd.DataFrame({"r": r, "m": mkt}).dropna()
X = np.column_stack([np.ones(len(A)), A.m.values, A.m.values ** 2])
b = np.linalg.lstsq(X, A.r.values, rcond=None)[0]
e = A.r.values - X @ b
se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * (e @ e) / (len(A) - 3)))
print("  alpha %+.5f (t %+.2f)   beta %+.3f (t %+.2f)   beta^2 %+.3f (t %+.2f)"
      % (b[0], b[0] / se[0], b[1], b[1] / se[1], b[2], b[2] / se[2]))
print("  corr with equal-weight crypto market: %+.3f" % A.r.corr(A.m))
