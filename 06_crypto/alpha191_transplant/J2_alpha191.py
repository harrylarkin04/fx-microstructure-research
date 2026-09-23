# -*- coding: utf-8 -*-
"""Alpha191 (GTJA) transplanted to the crypto cross-section.

Du, Walter & Ulrich (arXiv 2601.06499) take the Alpha191 library - built for the
retail-heavy Chinese A-share market - into the S&P 500 and find 17 survive
double-selection LASSO against 151 fundamentals. They argue these are UNIVERSAL
BEHAVIOURAL dynamics rather than market-specific quirks.

If that is right the effect should be LARGER where retail dominance exceeds the
A-share market. Crypto is the extreme case. This file computes the signals; J3
runs the double-selection and the portfolio.

Operators follow the GTJA definitions. Signals use daily OHLCV only.
"""
import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(BASE, "data", "crypto")
OUT = os.path.join(BASE, "out")

frames = {}
for fp in sorted(glob.glob(os.path.join(DD, "*.csv"))):
    t = os.path.basename(fp).replace(".csv", "")
    d = pd.read_csv(fp, parse_dates=["date"]).dropna(subset=["close"])
    d = d.drop_duplicates("date").set_index("date").sort_index()
    if len(d) >= 800:
        frames[t] = d
cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in frames.values()])))
cal = cal[(cal >= "2018-01-01") & (cal <= "2026-08-31")]


def panel(col):
    return pd.DataFrame({t: v[col].reindex(cal) for t, v in frames.items()})


O, H, L, C, V = panel("open"), panel("high"), panel("low"), panel("close"), panel("volume")
# liquidity screen: need real dollar volume
DV = (C * V)
liquid = DV.rolling(60, min_periods=30).median() > 5e6
C = C.where(liquid)
RET = np.log(C).diff()
RET = RET.mask(RET.abs() > 1.0)
VWAP = (H + L + C) / 3.0
AMT = DV

# ---------- GTJA operators ----------
def DELAY(x, n): return x.shift(n)
def DELTA(x, n): return x.diff(n)
def RANK(x): return x.rank(axis=1, pct=True)
def SUM(x, n): return x.rolling(n, min_periods=max(2, n // 2)).sum()
def MEAN(x, n): return x.rolling(n, min_periods=max(2, n // 2)).mean()
def STD(x, n): return x.rolling(n, min_periods=max(3, n // 2)).std()
def TSMIN(x, n): return x.rolling(n, min_periods=max(2, n // 2)).min()
def TSMAX(x, n): return x.rolling(n, min_periods=max(2, n // 2)).max()
def TSRANK(x, n): return x.rolling(n, min_periods=max(3, n // 2)).rank(pct=True)
def SMA(x, n, m): return x.ewm(alpha=float(m) / n, adjust=False, min_periods=n // 2).mean()
def CORR(x, y, n): return x.rolling(n, min_periods=max(3, n // 2)).corr(y)
def COV(x, y, n): return x.rolling(n, min_periods=max(3, n // 2)).cov(y)
def SIGN(x): return np.sign(x)
def MAXV(x, y): return x.where(x > y, y)
def COUNT(cond, n): return cond.astype(float).rolling(n, min_periods=max(2, n // 2)).sum()
def HIGHDAY(x, n): return x.rolling(n, min_periods=max(2, n // 2)).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True)
def LOWDAY(x, n): return x.rolling(n, min_periods=max(2, n // 2)).apply(lambda a: len(a) - 1 - int(np.argmin(a)), raw=True)
def DECAY(x, n):
    w = np.arange(1, n + 1, dtype=float); w /= w.sum()
    return x.rolling(n, min_periods=max(2, n // 2)).apply(lambda a: float(np.dot(a, w)), raw=True)

up = C > DELAY(C, 1)
dn = C < DELAY(C, 1)
HL = (H - L).replace(0, np.nan)

A = {}
A["a1"] = -CORR(RANK(DELTA(np.log(V.replace(0, np.nan)), 1)), RANK((C - O) / O), 6)
A["a2"] = -DELTA(((C - L) - (H - C)) / HL, 1)
A["a6"] = -RANK(SIGN(DELTA(O * 0.85 + H * 0.15, 4)))
A["a8"] = -RANK(DELTA(((H + L) / 2) * 0.2 + VWAP * 0.8, 4))
A["a9"] = SMA(((H + L) / 2 - (DELAY(H, 1) + DELAY(L, 1)) / 2) * HL / V.replace(0, np.nan), 7, 2)
A["a12"] = RANK(O - MEAN(VWAP, 10)) * (-RANK((C - VWAP).abs()))
A["a13"] = (H * L) ** 0.5 - VWAP
A["a14"] = C - DELAY(C, 5)
A["a15"] = O / DELAY(C, 1) - 1
A["a18"] = C / DELAY(C, 5)
A["a20"] = (C - DELAY(C, 6)) / DELAY(C, 6) * 100
A["a24"] = SMA(C - DELAY(C, 5), 5, 1)
A["a29"] = (C - DELAY(C, 6)) / DELAY(C, 6) * V
A["a31"] = (C - MEAN(C, 12)) / MEAN(C, 12) * 100
A["a32"] = -SUM(RANK(CORR(RANK(H), RANK(V), 3)), 3)
A["a34"] = MEAN(C, 12) / C
A["a40"] = SUM(V.where(up, 0), 26) / SUM(V.where(~up, 0), 26).replace(0, np.nan) * 100
A["a41"] = -RANK(TSMAX(DELTA(VWAP, 3), 5))
A["a42"] = -RANK(STD(H, 10)) * CORR(H, V, 10)
A["a43"] = SUM(V.where(up, 0) - V.where(dn, 0), 6)
A["a46"] = (MEAN(C, 3) + MEAN(C, 6) + MEAN(C, 12) + MEAN(C, 24)) / (4 * C)
A["a47"] = SMA((TSMAX(H, 6) - C) / (TSMAX(H, 6) - TSMIN(L, 6)).replace(0, np.nan) * 100, 9, 1)
A["a53"] = COUNT(up, 12) / 12 * 100
A["a57"] = SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)).replace(0, np.nan) * 100, 3, 1)
A["a58"] = COUNT(up, 20) / 20 * 100
A["a62"] = -CORR(H, RANK(V), 5)
A["a65"] = MEAN(C, 6) / C
A["a66"] = (C - MEAN(C, 6)) / MEAN(C, 6) * 100
A["a70"] = STD(AMT, 6)
A["a71"] = (C - MEAN(C, 24)) / MEAN(C, 24) * 100
A["a76"] = STD((C / DELAY(C, 1) - 1).abs() / V.replace(0, np.nan), 20) / \
    MEAN((C / DELAY(C, 1) - 1).abs() / V.replace(0, np.nan), 20).replace(0, np.nan)
A["a80"] = (V - DELAY(V, 5)) / DELAY(V, 5).replace(0, np.nan) * 100
A["a83"] = -RANK(COV(RANK(H), RANK(V), 5))
A["a84"] = SUM(V.where(up, 0) - V.where(dn, 0), 20)
A["a85"] = TSRANK(V / MEAN(V, 20).replace(0, np.nan), 20) * TSRANK(-DELTA(C, 7), 8)
A["a88"] = (C - DELAY(C, 20)) / DELAY(C, 20) * 100
A["a95"] = STD(AMT, 20)
A["a97"] = STD(V, 10)
A["a100"] = STD(V, 20)
A["a102"] = SMA(MAXV(V - DELAY(V, 1), 0 * V), 6, 1) / \
    SMA((V - DELAY(V, 1)).abs(), 6, 1).replace(0, np.nan) * 100
A["a103"] = (20 - LOWDAY(L, 20)) / 20 * 100
A["a106"] = C - DELAY(C, 20)
A["a109"] = SMA(HL, 10, 2) / SMA(SMA(HL, 10, 2), 10, 2).replace(0, np.nan)
A["a111"] = SMA(V * ((C - L) - (H - C)) / HL, 11, 2) - SMA(V * ((C - L) - (H - C)) / HL, 4, 2)
A["a117"] = TSRANK(V, 32) * (1 - TSRANK((C + H) - L, 16)) * (1 - TSRANK(RET, 32))
A["a118"] = SUM(H - O, 20) / SUM(O - L, 20).replace(0, np.nan) * 100
A["a126"] = (C + H + L) / 3
A["a129"] = SUM((C - DELAY(C, 1)).abs().where(dn, 0), 12)
A["a132"] = MEAN(AMT, 20)
A["a133"] = (20 - HIGHDAY(H, 20)) / 20 * 100 - (20 - LOWDAY(L, 20)) / 20 * 100
A["a134"] = (C - DELAY(C, 12)) / DELAY(C, 12) * V
A["a135"] = SMA(DELAY(C / DELAY(C, 20), 1), 20, 1)
A["a139"] = -CORR(O, V, 10)
A["a145"] = (MEAN(V, 9) - MEAN(V, 26)) / MEAN(V, 12).replace(0, np.nan) * 100
A["a150"] = (C + H + L) / 3 * V
A["a153"] = (MEAN(C, 3) + MEAN(C, 6) + MEAN(C, 12) + MEAN(C, 24)) / 4
A["a155"] = SMA(V, 13, 2) - SMA(V, 27, 2) - SMA(SMA(V, 13, 2) - SMA(V, 27, 2), 10, 2)
A["a158"] = ((H - SMA(C, 15, 2)) - (L - SMA(C, 15, 2))) / C
A["a161"] = MEAN(MAXV(MAXV(HL, (DELAY(C, 1) - H).abs()), (DELAY(C, 1) - L).abs()), 12)
A["a167"] = SUM((C - DELAY(C, 1)).where(up, 0), 12)
A["a168"] = -V / MEAN(V, 20).replace(0, np.nan)
A["a171"] = -((L - C) * (O ** 5)) / (((C - H) * (C ** 5)).replace(0, np.nan))
A["a175"] = MEAN(MAXV(MAXV(HL, (DELAY(C, 1) - H).abs()), (DELAY(C, 1) - L).abs()), 6)
A["a176"] = CORR(RANK((C - TSMIN(L, 12)) / (TSMAX(H, 12) - TSMIN(L, 12)).replace(0, np.nan)),
                 RANK(V), 6)
A["a177"] = (20 - HIGHDAY(H, 20)) / 20 * 100
A["a178"] = (C - DELAY(C, 1)) / DELAY(C, 1) * V
A["a188"] = (HL - SMA(HL, 11, 2)) / SMA(HL, 11, 2).replace(0, np.nan) * 100
A["a189"] = MEAN((C - MEAN(C, 6)).abs(), 6)
A["a191"] = CORR(MEAN(V, 20), L, 5) + (H + L) / 2 - C

keep = {}
for k, v in A.items():
    v = v.replace([np.inf, -np.inf], np.nan).where(liquid)
    cov = v.notna().sum(axis=1)
    if (cov >= 8).mean() > 0.5:
        keep[k] = v
print("computed %d Alpha191 signals with adequate cross-sectional coverage" % len(keep))
st = {k: v.stack() for k, v in keep.items()}
P = pd.DataFrame(st)
P.index.names = ["date", "asset"]
P = P.reset_index()
P.to_parquet(os.path.join(OUT, "crypto_alpha191.parquet"))
RET.stack().rename("ret").reset_index().rename(
    columns={"level_0": "date", "level_1": "asset"}).to_parquet(
    os.path.join(OUT, "crypto_rets.parquet"))
DV.stack().rename("dv").reset_index().rename(
    columns={"level_0": "date", "level_1": "asset"}).to_parquet(
    os.path.join(OUT, "crypto_dv.parquet"))
print("panel: %d rows, %d assets, %s -> %s"
      % (len(P), P.asset.nunique(), P.date.min().date(), P.date.max().date()))
print("median assets per day: %.0f" % P.groupby("date").size().median())
