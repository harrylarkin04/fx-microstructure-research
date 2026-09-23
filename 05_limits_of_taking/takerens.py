"""
The strongest taker ensemble I can build, traded only when the spread is small.

Two things are new versus the earlier attempt:

  1. CROSS-PAIR FEATURES.  The earlier ensemble (IC 0.174) used single-pair book
     features only.  The whole-graph fair-value residual and the USD-factor
     order flow were built later and never folded in.  Both are orthogonal to
     anything a single book can see.

  2. THE REQUIRED-IC FRAME.  Rather than asking "does it work", compute for
     every horizon and spread cap the IC that WOULD be needed to cover the
     realised round-trip cost, and set it against the IC actually achieved.
     Required IC = cost / sigma(h), so it falls as sqrt(h) while achieved IC
     also falls with h - the question is purely which falls faster, and that is
     one table rather than an argument.

Execution is priced honestly throughout: the decision uses the book at t, the
fills use the book at t+latency, so conditioning on a tiny spread does not
grant a tiny spread.
"""
import os
import sys

import numpy as np
import polars as pl

import feats as F
import trirv as T

if hasattr(sys.stdout, "reconfigure"):          # polars tables on a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

L1 = F.L1
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
        "2026-07-27", "2026-07-28", "2026-07-29"]
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
HS = (5, 20, 50, 100, 300)
KEEP = ["imb1", "imb5", "micro", "nimb", "ofi5", "ofi20", "mom5", "depth_asym"]


def graph_resid(day, grid):
    """Whole-graph fair-value residual for every pair, on a 1s grid."""
    syms = sorted({f.split("_")[0] for f in os.listdir(L1)
                   if f.endswith(f"_{day}.parquet")})
    syms = [s for s in syms if len(s) == 6 and s[:3] in T.CCYS and s[3:] in T.CCYS]
    lp, sp = [], []
    keep = []
    for s in syms:
        d = pl.read_parquet(os.path.join(L1, f"{s}_{day}.parquet"),
                            columns=["ts_ms", "bid", "ask"]).filter(
            pl.col("ask") > pl.col("bid"))
        if len(d) < 20000:
            continue
        ts = d["ts_ms"].to_numpy()
        m = ((d["bid"] + d["ask"]) / 2).to_numpy()
        j = np.searchsorted(ts, grid, side="right") - 1
        v = np.where(j >= 0, np.log(m[np.maximum(j, 0)]), np.nan)
        sd = np.where(j >= 0, ((d["ask"] - d["bid"]).to_numpy()
                               / m)[np.maximum(j, 0)] * 1e4, np.nan)
        lp.append(v)
        sp.append(sd)
        keep.append(s)
    LP = np.vstack(lp)
    SP = np.vstack(sp)
    ccy = [c for c in T.CCYS if c != "USD"]
    col = {c: i for i, c in enumerate(ccy)}
    A = np.zeros((len(keep), len(ccy)))
    for i, s in enumerate(keep):
        if s[:3] in col:
            A[i, col[s[:3]]] += 1
        if s[3:] in col:
            A[i, col[s[3:]]] -= 1
    w = 1.0 / np.maximum(np.nanmedian(SP, axis=1), 0.05)
    W = np.diag(w / w.sum() * len(w))
    M = np.linalg.solve(A.T @ W @ A + 1e-9 * np.eye(A.shape[1]), A.T @ W)
    Y = np.nan_to_num(LP)
    R = (LP - A @ (M @ Y)) * 1e4
    R[:, ~np.isfinite(LP).all(axis=0)] = np.nan
    return keep, R


def usd_flow(day, grid):
    """USD-factor order flow: dollar buying pressure across the majors."""
    sgn = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
           "USDJPY": +1, "USDCHF": +1, "USDCAD": +1}
    acc = []
    for s, g in sgn.items():
        p = os.path.join(L1, f"{s}_{day}.parquet")
        if not os.path.exists(p):
            continue
        d = pl.read_parquet(p, columns=["ts_ms", "bid", "ask", "bid_sz",
                                        "ask_sz"]).filter(
            pl.col("ask") > pl.col("bid"))
        ts = d["ts_ms"].to_numpy()
        bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
        bs, az = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
        n = len(bid)
        db = np.zeros(n); da = np.zeros(n)
        ub, dn = bid[1:] > bid[:-1], bid[1:] < bid[:-1]
        db[1:] = (np.where(ub, bs[1:], np.where(dn, -bs[:-1], 0.0))
                  + np.where(~ub & ~dn, bs[1:] - bs[:-1], 0.0))
        da_, ua = ask[1:] < ask[:-1], ask[1:] > ask[:-1]
        da[1:] = (np.where(da_, az[1:], np.where(ua, -az[:-1], 0.0))
                  + np.where(~da_ & ~ua, az[1:] - az[:-1], 0.0))
        c = np.cumsum(db - da)
        j = np.searchsorted(ts, grid, side="right") - 1
        v = np.where(j >= 0, c[np.maximum(j, 0)], np.nan)
        dv = np.full(len(grid), np.nan)
        dv[1:] = np.diff(v)
        s_ = np.nanstd(dv)
        acc.append(g * dv / (s_ if s_ > 0 else 1.0))
    return np.nanmean(np.vstack(acc), axis=0)


def build(sym, day):
    f = F.build(sym, day)
    if f is None:
        return None
    ts = f["ts"]
    lo, hi = int(ts[0]), int(ts[-1])
    grid = np.arange(lo, hi, 1000)
    keep, R = graph_resid(day, grid)
    uf = usd_flow(day, grid)
    if sym not in keep:
        return None
    r = R[keep.index(sym)]
    # trailing z of the residual, then forward-fill onto the event clock
    c1 = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(r))])
    c2 = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(r) ** 2)])
    i = np.arange(len(r)); loi = np.maximum(0, i - 1800)
    cnt = i + 1 - loi
    mu = (c1[i + 1] - c1[loi]) / cnt
    sd = np.sqrt(np.maximum((c2[i + 1] - c2[loi]) / cnt - mu * mu, 1e-18))
    z = (r - mu) / sd
    ufz = uf / (np.nanstd(uf) if np.nanstd(uf) > 0 else 1.0)
    j = np.searchsorted(grid, ts, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    X = f["X"][:, [F.NAMES.index(k) for k in KEEP]].astype(np.float64)
    X = np.column_stack([X,
                         np.where(ok, np.nan_to_num(z[j]), 0.0),
                         np.where(ok, np.nan_to_num(ufz[j]), 0.0)])
    return dict(X=X, mid=f["mid"], bid=f["bid"], ask=f["ask"], ts=ts,
                spread=f["spread"], pip=f["pip"], n=f["n"])


def main(lat_ms=1.0):
    names = KEEP + ["graph_z", "usd_flow"]
    cache = {}
    for sym in MAJORS:
        for day in DAYS:
            b = build(sym, day)
            if b is not None:
                cache[(sym, day)] = b
        print("built", sym, flush=True)

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
                    lm = np.log(b["mid"])
                    y = np.full(b["n"], np.nan)
                    y[:-h - 1] = (lm[h + 1:] - lm[1:-h]) * 1e4
                    m = np.isfinite(y) & np.isfinite(b["X"]).all(1)
                    Xs.append(b["X"][m][::5]); ys.append(y[m][::5])
                X = np.vstack(Xs); y = np.concatenate(ys)
                Z = np.column_stack([np.ones(len(X)), X])
                w = np.linalg.solve(Z.T @ Z + 1e-2 * len(X) * np.eye(Z.shape[1]),
                                    Z.T @ y)
                lm = np.log(te["mid"])
                yt = np.full(te["n"], np.nan)
                yt[:-h - 1] = (lm[h + 1:] - lm[1:-h]) * 1e4
                pred = np.column_stack([np.ones(te["n"]), te["X"]]) @ w
                m = np.isfinite(yt) & np.isfinite(pred)
                if m.sum() < 5000:
                    continue
                ic = float(np.corrcoef(pred[m], yt[m])[0, 1])
                sig = float(np.std(yt[m]))
                # honest execution: decide at t, fill at t+latency
                arr = np.searchsorted(te["ts"], te["ts"] + lat_ms, side="right")
                arr = np.minimum(np.maximum(arr, np.arange(te["n"]) + 1),
                                 te["n"] - 1)
                for cap in (0.15, 0.25, 0.35, 99.0):
                    sel = m & (te["spread"] <= cap)
                    idx = np.flatnonzero(sel)
                    if idx.size < 200:
                        continue
                    kk, last = [], -(10 ** 9)
                    for q in idx:
                        if q - last >= h:
                            kk.append(q); last = q
                    kk = np.array(kk)
                    sd_ = np.sign(pred[kk])
                    ein = arr[kk]
                    xout = np.minimum(arr[np.minimum(kk + h, te["n"] - 1)],
                                      te["n"] - 1)
                    g = np.where(xout > ein)
                    ein, xout, sd_ = ein[g], xout[g], sd_[g]
                    if ein.size < 50:
                        continue
                    buy = np.where(sd_ > 0, te["ask"][ein], te["bid"][ein])
                    sell = np.where(sd_ > 0, te["bid"][xout], te["ask"][xout])
                    net = sd_ * (sell - buy) / te["pip"]
                    midmv = sd_ * (te["mid"][xout] - te["mid"][ein]) / te["pip"]
                    rows.append(dict(h=h, day=day, sym=sym, cap=cap,
                                     n=int(ein.size), ic=ic, sigma=sig,
                                     mid_pip=float(midmv.mean()),
                                     cost=float(midmv.mean() - net.mean()),
                                     net=float(net.mean()),
                                     entry_spr=float(te["spread"][kk[g]].mean())))
    d = pl.DataFrame(rows)
    d.write_csv(os.path.join(OUT, f"takerens_lat{lat_ms}.csv"))

    print(f"\n=== ENSEMBLE IC by horizon ({len(names)} features incl. "
          f"graph residual + USD flow) ===")
    a = (d.group_by("h").agg(pl.col("ic").mean(), pl.col("sigma").mean(),
                             pl.len()).sort("h"))
    with pl.Config(tbl_width_chars=120, float_precision=4):
        print(a)

    print(f"\n=== REQUIRED IC vs ACHIEVED, latency {lat_ms}ms ===")
    print("   required IC = realised round-trip cost / sigma(h)")
    print(f"{'h':>5}{'cap':>7}{'n':>9}{'IC':>8}{'sigma':>8}{'cost':>8}"
          f"{'net':>9}{'IC needed':>11}{'shortfall':>11}")
    w = pl.col("n")
    g = (d.group_by(["h", "cap"]).agg(
        w.sum().alias("n"),
        pl.col("ic").mean().alias("ic"),
        pl.col("sigma").mean().alias("sigma"),
        ((pl.col("cost") * w).sum() / w.sum()).alias("cost"),
        ((pl.col("net") * w).sum() / w.sum()).alias("net"),
        ((pl.col("entry_spr") * w).sum() / w.sum()).alias("entry"))
        .sort(["h", "cap"]))
    for r in g.iter_rows(named=True):
        need = r["cost"] / r["sigma"] if r["sigma"] > 0 else np.nan
        print(f"{r['h']:>5}{r['cap']:>7.2f}{r['n']:>9,}{r['ic']:>8.4f}"
              f"{r['sigma']:>8.4f}{r['cost']:>8.4f}{r['net']:>+9.4f}"
              f"{need:>11.4f}{need / max(r['ic'], 1e-9):>10.1f}x")

    # the bound: what a PERFECT direction call earns, at the tightest cap.
    # E|move| = sigma * sqrt(2/pi) for a normal, so this is the ceiling on
    # gross P&L per trade, against the cost the same trades actually paid.
    print("\n=== PERFECT FORESIGHT at cap 0.15 (pips) ===")
    print("   E|move| = sigma * sqrt(2/pi); net = E|move| - realised cost")
    print(f"{'h':>5}{'E|move|':>10}{'cost':>8}{'net':>9}")
    for r in g.filter(pl.col("cap") == 0.15).iter_rows(named=True):
        mv = r["sigma"] * np.sqrt(2 / np.pi)
        print(f"{r['h']:>5}{mv:>10.3f}{r['cost']:>8.3f}{mv - r['cost']:>+9.3f}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
