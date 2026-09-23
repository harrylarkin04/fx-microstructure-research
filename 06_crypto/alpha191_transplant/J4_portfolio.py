# -*- coding: utf-8 -*-
"""Holdout portfolio from the surviving Alpha191 signals.

Selection happened on 2018-2022 only (J3). Everything here is 2023-2026, never
touched. Market-neutral cross-sectional long-short, monthly rebalance as in the
paper, net of realistic crypto taker costs.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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

D = P.merge(R, on=["date", "asset"]).merge(DV, on=["date", "asset"])
D = D[D.dv > 5e6]
piv = {s: D.pivot_table(index="date", columns="asset", values=s) for s in SURV}
RET = D.pivot_table(index="date", columns="asset", values="ret")
cal = RET.index


def cz(x):
    z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-3, 3).where(x.notna().sum(axis=1) >= 8)


# equal-risk composite of survivors, each oriented by its train-period sign
comp = None
for s in SURV:
    z = cz(piv[s]) * sgn.get(s, 1.0)
    comp = z if comp is None else comp.add(z, fill_value=0.0)
comp = comp / len(SURV)

HOLD = 21          # monthly rebalance
COST_BP = {"lo": 5.0, "mid": 15.0, "hi": 40.0}   # per unit one-way turnover


def book(sig, hold, cost_bp, start, end):
    z = cz(sig)
    w = z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0) * 2.0
    W = (sum(w.shift(k + 1) for k in range(hold)) / hold).fillna(0.0)
    gross = (W * RET).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    net = gross - turn * cost_bp / 1e4
    net = net.loc[start:end]
    return net[net.notna()]


def stat(r, lab, P_=365):
    if len(r) < 60:
        print("  %-34s (insufficient)" % lab)
        return None
    e = (1 + r).cumprod()
    dd = (e / e.cummax() - 1).min()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(P_)
    t = r.mean() / r.std() * np.sqrt(len(r))
    print("  %-34s Sh %5.2f  t %5.2f  CAGR %7.2f%%  vol %5.1f%%  maxDD %7.2f%%  Calmar %5.2f"
          % (lab, sh, t, cagr * 100, r.std() * np.sqrt(P_) * 100, dd * 100,
             cagr / abs(dd)))
    return r


print("=== TRAIN 2018-2022 (where selection happened - expect it to look good) ===")
for k, c in COST_BP.items():
    stat(book(comp, HOLD, c, "2018-01-01", "2022-12-31"), "  composite, cost %s (%.0fbp)" % (k, c))

print("\n=== HOLDOUT 2023-2026 (never touched) ===")
res = {}
for k, c in COST_BP.items():
    r = stat(book(comp, HOLD, c, "2023-01-01", "2026-08-31"), "  composite, cost %s (%.0fbp)" % (k, c))
    res[k] = r

print("\n=== holdout: each survivor alone (mid cost) ===")
keep = []
for s in SURV:
    r = book(cz(piv[s]) * sgn.get(s, 1.0), HOLD, COST_BP["mid"], "2023-01-01", "2026-08-31")
    if len(r) > 60:
        sh = r.mean() / r.std() * np.sqrt(365)
        keep.append((s, sh))
keep.sort(key=lambda x: -x[1])
print("  " + "  ".join("%s %+.2f" % k for k in keep))
print("  positive out of sample: %d of %d" % (sum(1 for _, v in keep if v > 0), len(keep)))

print("\n=== holdout by rebalance horizon (mid cost) ===")
for h in [5, 10, 21, 42, 63]:
    stat(book(comp, h, COST_BP["mid"], "2023-01-01", "2026-08-31"), "  hold %dd" % h)

r = res["mid"]
BOOK = os.environ.get("BOOK_RETURNS_CSV")  # optional: daily returns of an existing book
if r is not None and BOOK:
    cry = pd.read_csv(BOOK, index_col=0,
                      parse_dates=True).iloc[:, 0].dropna()
    A = pd.DataFrame({"alpha191": r, "crypto_book": cry}).dropna()
    print("\n=== pairing with the digital-asset book (holdout overlap) ===")
    print("  correlation %+.4f   (n=%d days)" % (A.alpha191.corr(A.crypto_book), len(A)))
    for w in [0.0, 0.2, 0.3, 0.4, 0.5]:
        b = (1 - w) * A.crypto_book + w * A.alpha191
        k = A.crypto_book.std() / b.std()
        x = b * k
        e = (1 + x).cumprod()
        dd = (e / e.cummax() - 1).min()
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        cg = (1 + x).prod() ** (1 / yrs) - 1
        print("  %3.0f%% crypto / %3.0f%% alpha191 (%.2fx): Sh %5.2f  CAGR %7.2f%%  maxDD %7.2f%%  Calmar %5.2f"
              % ((1 - w) * 100, w * 100, k, x.mean() / x.std() * np.sqrt(365), cg * 100,
                 dd * 100, cg / abs(dd)))
    r.to_csv(os.path.join(OUT, "CRYPTO_ALPHA191_BOOK.csv"))
    print("\nsaved out/CRYPTO_ALPHA191_BOOK.csv")
