"""
Signal library at event resolution, one row per touch change.

Every signal here is economically motivated and meant to stand on its own in a
univariate regression.  Nothing is a transform hunted for by search; the point
is to have a small set of interpretable predictors that each survive OLS, so
that an ensemble on top of them is combining real effects rather than mining
noise.

Targets are SKIP-ONE forward mid changes: y_h = mid(t+h+1) - mid(t+1).  Starting
one event late removes the immediate quote flip, so a signal cannot score by
predicting bid-ask bounce.
"""
import os

import numpy as np
import polars as pl

L1 = os.environ.get("CBOE_L1_DIR", "data/cboe_l1")
SYMS = ["EURUSD", "GBPUSD", "USDJPY",
        "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURGBP", "EURCHF", "EURJPY",
        "EURAUD", "EURCAD", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "AUDNZD",
        "AUDCAD", "AUDCHF", "CADCHF"]
DAYS = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
        "2026-07-27", "2026-07-28", "2026-07-29"]
HORIZONS = (5, 20, 100)

NAMES = ["imb1", "imb5", "micro", "nimb", "ofi5", "ofi20", "mom5", "mom50",
         "spread_z", "thin", "intensity", "depth_asym", "vol_z", "wipe_asym"]


def pip_of(s):
    return 0.01 if s.endswith("JPY") else 0.0001


def _roll_mean(x, w):
    """Causal rolling mean over w events (expanding until full)."""
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - w + 1)
    return (cs[i + 1] - cs[lo]) / (i + 1 - lo)


def build(sym, day):
    p = os.path.join(L1, f"{sym}_{day}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).filter(pl.col("spread_pip") > 0)
    n = len(d)
    if n < 20000:
        return None
    pip = pip_of(sym)
    ts = d["ts_ms"].to_numpy().astype(np.float64)
    bid = d["bid"].to_numpy()
    ask = d["ask"].to_numpy()
    bs = d["bid_sz"].to_numpy()
    az = d["ask_sz"].to_numpy()
    bd = d["bid_d5"].to_numpy()
    ad = d["ask_d5"].to_numpy()
    bn = d["bid_n"].to_numpy().astype(np.float64)
    an = d["ask_n"].to_numpy().astype(np.float64)
    mid = (bid + ask) / 2.0
    spr = (ask - bid) / pip

    X = {}
    # 1-2  static book pressure: more size bid than offered -> price drifts up
    X["imb1"] = (bs - az) / (bs + az)
    X["imb5"] = (bd - ad) / (bd + ad)
    # 3    size-weighted fair price, as a fraction of the spread
    X["micro"] = ((bid * az + ask * bs) / (bs + az) - mid) / (ask - bid)
    # 4    many small orders vs few large ones at the touch
    X["nimb"] = (bn - an) / (bn + an)

    # 5-6  order flow imbalance (Cont/Kukanov/Stoikov): the signed size added to
    #      the bid minus the signed size added to the ask, event by event
    db = np.zeros(n)
    da = np.zeros(n)
    up_b = bid[1:] > bid[:-1]
    dn_b = bid[1:] < bid[:-1]
    sm_b = ~up_b & ~dn_b
    db[1:] = np.where(up_b, bs[1:], np.where(dn_b, -bs[:-1], 0.0)) \
        + np.where(sm_b, bs[1:] - bs[:-1], 0.0)
    dn_a = ask[1:] < ask[:-1]
    up_a = ask[1:] > ask[:-1]
    sm_a = ~up_a & ~dn_a
    da[1:] = np.where(dn_a, az[1:], np.where(up_a, -az[:-1], 0.0)) \
        + np.where(sm_a, az[1:] - az[:-1], 0.0)
    ofi = db - da
    sc = np.maximum(_roll_mean(np.abs(ofi), 2000), 1.0)
    for w in (5, 20):
        cs = np.concatenate([[0.0], np.cumsum(ofi)])
        r = np.zeros(n)
        i = np.arange(n)
        lo = np.maximum(0, i - w + 1)
        r = (cs[i + 1] - cs[lo]) / (w * sc)
        X[f"ofi{w}"] = r

    # 7-8  short-horizon momentum, expected to REVERT after a burst
    for w in (5, 50):
        m = np.full(n, np.nan)
        m[w:] = (mid[w:] - mid[:-w]) / pip
        X[f"mom{w}"] = m / np.maximum(_roll_mean(np.abs(np.nan_to_num(m)), 2000), 1e-6)

    # 9    spread against its own recent level: wide = uncertain / risky
    X["spread_z"] = spr / np.maximum(_roll_mean(spr, 2000), 1e-6) - 1.0
    # 10   thin touch is fragile
    tot = bs + az
    X["thin"] = tot / np.maximum(_roll_mean(tot, 2000), 1.0) - 1.0
    # 11   event intensity: a busy tape is a moving tape
    dt = np.maximum(np.diff(ts, prepend=ts[0]), 0.0)
    X["intensity"] = -np.log1p(dt) + np.log1p(_roll_mean(dt, 2000))
    # 12   depth behind the touch, bid side vs ask side
    rb = bd / np.maximum(bs, 1.0)
    ra = ad / np.maximum(az, 1.0)
    X["depth_asym"] = np.log(np.maximum(rb, 1e-6)) - np.log(np.maximum(ra, 1e-6))
    # 13   realised vol regime
    dm = np.abs(np.diff(mid, prepend=mid[0])) / pip
    X["vol_z"] = dm / np.maximum(_roll_mean(dm, 2000), 1e-6) - 1.0
    # 14   which side keeps getting wiped out
    wb = (bid < np.concatenate([[bid[0]], bid[:-1]])).astype(np.float64)
    wa = (ask > np.concatenate([[ask[0]], ask[:-1]])).astype(np.float64)
    X["wipe_asym"] = _roll_mean(wa, 20) - _roll_mean(wb, 20)

    Xm = np.column_stack([np.nan_to_num(X[k], nan=0.0, posinf=0.0, neginf=0.0)
                          for k in NAMES]).astype(np.float32)
    np.clip(Xm, -10, 10, out=Xm)

    Y = {}
    for h in HORIZONS:
        y = np.full(n, np.nan)
        y[:-h - 1] = (mid[h + 1:] - mid[1:-h]) / pip
        Y[h] = y.astype(np.float32)

    return dict(X=Xm, Y=Y, spread=spr.astype(np.float32),
                mid=mid, bid=bid, ask=ask, ts=ts,
                bid_sz=bs, ask_sz=az, pip=pip, n=n)


if __name__ == "__main__":
    f = build("EURUSD", "2026-07-23")
    print("rows", f["n"])
    for i, k in enumerate(NAMES):
        x = f["X"][:, i]
        print(f"  {k:12s} sd={x.std():.4f}  "
              f"IC(h=20)={np.corrcoef(x[:-101], f['Y'][20][:-101])[0,1]:+.4f}")
