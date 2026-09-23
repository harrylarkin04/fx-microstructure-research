"""
Pushing the taker at the long horizon, where the perfect-foresight bound does
not rule it out.

Two levers, neither of which has been tried:

  1. FEATURES AT THE RIGHT TIMESCALE.  Every feature in the ensemble is a 5-20
     event signal - touch imbalance, 5-event order flow, micro-price.  At a
     300-event horizon those are noise by construction.  This adds accumulated
     flow, deep-book pressure, vol regime and cross-pair factor structure
     measured over hundreds of events instead of a handful.

  2. VOLATILITY CONDITIONING.  Required IC = cost / sigma(h).  The cost is
     roughly the spread and is fairly stable; sigma is not.  If sigma rises
     faster than the spread in volatile periods, the IC needed to break even
     FALLS, and the bar may be reachable somewhere even if it is not on
     average.  That is a conditioning question, not a forecasting one, and it
     has never been measured here.
"""
import os
import sys

import numpy as np
import polars as pl

import trirv as T

L1 = os.environ.get("CBOE_L1_DIR", "data/cboe_l1")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
        "2026-07-27", "2026-07-28", "2026-07-29"]
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
HS = (300, 1000, 3000)


def roll(x, w):
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def build(sym, day):
    p = os.path.join(L1, f"{sym}_{day}.parquet")
    if not os.path.exists(p):
        return None
    d = pl.read_parquet(p).filter(pl.col("spread_pip") > 0)
    n = len(d)
    if n < 100_000:
        return None
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    bs, az = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
    bd, ad = d["bid_d5"].to_numpy(), d["ask_d5"].to_numpy()
    ts = d["ts_ms"].to_numpy().astype(float)
    mid = (bid + ask) / 2
    pip = 0.01 if sym.endswith("JPY") else 0.0001
    spr = (ask - bid) / mid * 1e4
    lm = np.log(mid)

    db = np.zeros(n); da = np.zeros(n)
    ub, dn = bid[1:] > bid[:-1], bid[1:] < bid[:-1]
    db[1:] = (np.where(ub, bs[1:], np.where(dn, -bs[:-1], 0.0))
              + np.where(~ub & ~dn, bs[1:] - bs[:-1], 0.0))
    da_, ua = ask[1:] < ask[:-1], ask[1:] > ask[:-1]
    da[1:] = (np.where(da_, az[1:], np.where(ua, -az[:-1], 0.0))
              + np.where(~da_ & ~ua, az[1:] - az[:-1], 0.0))
    ofi = db - da
    sc = np.maximum(roll(np.abs(ofi), 5000), 1.0)

    X, names = [], []

    def add(v, nm):
        X.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        names.append(nm)

    # accumulated flow over hundreds of events, not five
    for w in (50, 300, 1000, 3000):
        add(roll(ofi, w) / sc, f"ofi{w}")
    # momentum / reversion at the horizon's own scale
    for w in (300, 1000, 3000):
        m = np.full(n, np.nan)
        m[w:] = (lm[w:] - lm[:-w]) * 1e4
        add(m / np.maximum(roll(np.abs(np.nan_to_num(m)), 5000), 1e-6),
            f"mom{w}")
    # book pressure, touch and deep
    add((bs - az) / (bs + az), "imb1")
    add((bd - ad) / (bd + ad), "imb5")
    # persistent deep-book pressure
    add(roll((bd - ad) / (bd + ad), 1000), "imb5_1k")
    # vol regime and liquidity regime
    dm = np.abs(np.diff(lm, prepend=lm[0])) * 1e4
    vol = roll(dm, 1000)
    add(vol / np.maximum(roll(dm, 20000), 1e-9) - 1.0, "vol_reg")
    add(spr / np.maximum(roll(spr, 20000), 1e-9) - 1.0, "spr_reg")
    dt = np.maximum(np.diff(ts, prepend=ts[0]), 0.0)
    add(np.log1p(roll(dt, 1000)) - np.log1p(roll(dt, 20000)), "intensity")
    add(roll((bd + ad) / 2, 1000) / np.maximum(roll((bd + ad) / 2, 20000), 1.0)
        - 1.0, "depth_reg")

    Xm = np.column_stack(X)
    np.clip(Xm, -10, 10, out=Xm)
    return dict(X=Xm, names=names, lm=lm, mid=mid, bid=bid, ask=ask,
                spread=spr, vol=vol, pip=pip, n=n)


def main():
    cache = {}
    for sym in MAJORS:
        for day in DAYS:
            b = build(sym, day)
            if b is not None:
                cache[(sym, day)] = b
        print("built", sym, flush=True)
    names = cache[next(iter(cache))]["names"]
    print(f"{len(names)} features: {names}\n")

    rows = []
    for h in HS:
        for i, day in enumerate(DAYS):
            if i < 2:
                continue
            for sym in MAJORS:
                tr = [cache[(sym, d)] for d in DAYS[:i] if (sym, d) in cache]
                te = cache.get((sym, day))
                if te is None or len(tr) < 2:
                    continue
                Xs, ys = [], []
                for b in tr:
                    y = np.full(b["n"], np.nan)
                    y[:-h - 1] = (b["lm"][h + 1:] - b["lm"][1:-h]) * 1e4
                    m = np.isfinite(y)
                    Xs.append(b["X"][m][::10]); ys.append(y[m][::10])
                X = np.vstack(Xs); y = np.concatenate(ys)
                Z = np.column_stack([np.ones(len(X)), X])
                w = np.linalg.solve(Z.T @ Z + 1e-2 * len(X) * np.eye(Z.shape[1]),
                                    Z.T @ y)
                yt = np.full(te["n"], np.nan)
                yt[:-h - 1] = (te["lm"][h + 1:] - te["lm"][1:-h]) * 1e4
                pred = np.column_stack([np.ones(te["n"]), te["X"]]) @ w
                m = np.isfinite(yt)
                if m.sum() < 20000:
                    continue
                # vol regime terciles, defined on the TRAILING window only
                v = te["vol"]
                lo, hi = np.nanquantile(v[m], [1 / 3, 2 / 3])
                for lab, sel in (("low", m & (v <= lo)),
                                 ("mid", m & (v > lo) & (v <= hi)),
                                 ("high", m & (v > hi)),
                                 ("all", m)):
                    if sel.sum() < 5000:
                        continue
                    a, b_ = pred[sel], yt[sel]
                    if a.std() == 0 or b_.std() == 0:
                        continue
                    rows.append(dict(
                        h=h, sym=sym, day=day, regime=lab, n=int(sel.sum()),
                        ic=float(np.corrcoef(a, b_)[0, 1]),
                        sigma=float(b_.std()),
                        spread=float(np.nanmean(te["spread"][sel]))))
    d = pl.DataFrame(rows)
    d.write_csv(os.path.join(OUT, "longhorizon.csv"))

    w = pl.col("n")
    g = (d.group_by(["h", "regime"]).agg(
        w.sum().alias("n"),
        pl.col("ic").mean().alias("ic"),
        ((pl.col("sigma") * w).sum() / w.sum()).alias("sigma"),
        ((pl.col("spread") * w).sum() / w.sum()).alias("spread"))
        .sort(["h", "regime"]))
    print("=== IC and the REQUIRED IC, by horizon and volatility regime ===")
    print("   cost is taken as the round trip actually paid ~ 2.2x the quoted")
    print("   spread at these caps (measured earlier: 0.10 quoted -> 0.288 paid)")
    print(f"{'h':>6}{'regime':>8}{'n':>10}{'IC':>8}{'sigma':>8}{'spread':>8}"
          f"{'cost':>8}{'IC needed':>11}{'gap':>8}")
    for r in g.iter_rows(named=True):
        cost = 2.2 * r["spread"]
        need = cost / r["sigma"]
        print(f"{r['h']:>6}{r['regime']:>8}{r['n']:>10,}{r['ic']:>8.4f}"
              f"{r['sigma']:>8.4f}{r['spread']:>8.4f}{cost:>8.4f}"
              f"{need:>11.4f}{need/max(r['ic'],1e-9):>7.1f}x")


if __name__ == "__main__":
    main()
