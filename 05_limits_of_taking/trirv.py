"""
The options RV idea, translated to spot FX.

Their trade: fair-value an illiquid instrument using information borrowed from
liquid proxies, then trade the gap and wait for convergence.  In spot FX the
borrowing is not statistical, it is exact:

    synthetic(EURJPY) = EURUSD x USDJPY

So for every cross we can build a fair value out of its two liquid legs and ask
whether the QUOTED cross reverts toward it.  That is a statistical RV trade
with a holding period, which is a different question from the instantaneous
triangular arbitrage tested earlier (that one has to clear three spreads at
once; this one only has to clear the cross's own spread, because the prediction
is about the cross alone).

THE TRAP THIS MUST SURVIVE.  If the cross quote is simply STALE, then the gap
closes mechanically and predicting it is worth nothing: you cannot trade at a
price that is already gone.  Everything below is therefore run twice, once with
no freshness requirement and once demanding the cross was quoted within the
last `fresh_ms`, and with execution priced at the touch one latency hop later.
"""
import itertools
import os
import sys

import numpy as np
import polars as pl

L1 = os.environ.get("CBOE_L1_DIR", "data/cboe_l1")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
        "2026-07-27", "2026-07-28", "2026-07-29"]

CCYS = ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CHF", "CAD",
        "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "MXN", "ZAR",
        "SGD", "CNH", "ILS", "THB", "HKD"]
HORIZONS = (1, 5, 15, 60, 300)


def pip_of(s):
    return 0.01 if s.endswith("JPY") else 0.0001


def have(sym, day):
    return os.path.exists(os.path.join(L1, f"{sym}_{day}.parquet"))


def triangles(day):
    """(cross, legA, legB, sgnA, sgnB) with log(cross) = sA*log(A)+sB*log(B)."""
    out = []
    for a, b, c in itertools.combinations(CCYS, 3):
        # the cross is the pair NOT involving USD if there is one
        legs = {}
        ok = True
        for x, y in ((a, b), (b, c), (a, c)):
            if have(x + y, day):
                legs[(x, y)] = (x + y, +1)
            elif have(y + x, day):
                legs[(x, y)] = (y + x, -1)
            else:
                ok = False
                break
        if not ok:
            continue
        for (x, y) in list(legs):
            # express pair (x,y) via the other two
            others = [k for k in legs if k != (x, y)]
            (p, q), (r, s) = others
            shared = ({p, q} & {r, s})
            if len(shared) != 1:
                continue
            m = shared.pop()
            cross_sym, cs = legs[(x, y)]
            A, sa = legs[(p, q)]
            B, sb = legs[(r, s)]
            out.append((cross_sym, cs, A, sa, B, sb, (p, q), (r, s), m, (x, y)))
    return out


def grid(sym, day, g):
    d = pl.read_parquet(os.path.join(L1, f"{sym}_{day}.parquet"),
                        columns=["ts_ms", "bid", "ask"]).filter(
        pl.col("ask") > pl.col("bid"))
    ts = d["ts_ms"].to_numpy()
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    j = np.searchsorted(ts, g, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    age = np.where(ok, g - ts[j], 1 << 30)
    return (np.where(ok, bid[j], np.nan), np.where(ok, ask[j], np.nan),
            age.astype(np.float64))


def trail_mean_sd(x, w):
    v = np.nan_to_num(x)
    c = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - w + 1)
    cnt = (i + 1 - lo)
    m = (c[i + 1] - c[lo]) / cnt
    s2 = (c2[i + 1] - c2[lo]) / cnt - m * m
    return m, np.sqrt(np.maximum(s2, 1e-18))


def run(day, fresh_ms, lat_s=1, win=1800):
    p0 = pl.read_parquet(os.path.join(L1, f"EURUSD_{day}.parquet"),
                         columns=["ts_ms"])["ts_ms"].to_numpy()
    g = np.arange(p0[0], p0[-1], 1000)
    cache = {}

    def get(sym):
        if sym not in cache:
            cache[sym] = grid(sym, day, g)
        return cache[sym]

    rows = []
    seen = set()
    for (cross, cs, A, sa, B, sb, ka, kb, shared, kx) in triangles(day):
        key = (cross, A, B)
        if key in seen or cross in (A, B):
            continue
        seen.add(key)
        try:
            cb, ca, cage = get(cross)
            ab, aa, aage = get(A)
            bb, ba, bage = get(B)
        except Exception:                                    # noqa: BLE001
            continue
        cm = np.log((cb + ca) / 2)
        am = np.log((ab + aa) / 2)
        bm = np.log((bb + ba) / 2)
        synth = sa * am + sb * bm
        # the constant offset absorbs the direction convention; the DEVIATION
        # from its own trailing level is what we trade
        basis = (cm - synth) * 1e4
        if not np.isfinite(basis).sum() > 20000:
            continue
        mu, sd = trail_mean_sd(basis, win)
        dev = basis - mu
        z = dev / np.where(sd > 1e-9, sd, np.nan)

        pip = pip_of(cross)
        spr = (ca - cb) / pip
        spr_bp = (ca - cb) / ((ca + cb) / 2) * 1e4
        fresh = (cage <= fresh_ms) & (aage <= fresh_ms) & (bage <= fresh_ms)

        for h in HORIZONS:
            f = np.full(len(cm), np.nan)
            f[:-h] = (cm[h:] - cm[:-h]) * 1e4
            # execution one latency step later, so we never trade on the
            # very print that defined the signal
            fl = np.full(len(cm), np.nan)
            e = lat_s
            fl[:-h - e] = (cm[h + e:] - cm[e:-h]) * 1e4
            m = np.isfinite(z) & np.isfinite(fl) & fresh
            idx = np.flatnonzero(m)[::h]
            if len(idx) < 200:
                continue
            zz, ff = z[idx], fl[idx]
            if zz.std() == 0 or ff.std() == 0:
                continue
            ic = float(np.corrcoef(zz, ff)[0, 1])
            # economics: take the top decile of |z| and see what it captures
            k = max(int(len(idx) * 0.10), 50)
            sel = np.argsort(-np.abs(zz))[:k]
            cap = float(np.mean(-np.sign(zz[sel]) * ff[sel]))
            rows.append(dict(day=day, cross=cross, legs=f"{A}/{B}", h=h,
                             n=len(idx), ic=ic,
                             basis_sd_bp=float(np.nanmedian(sd)),
                             spread_bp=float(np.nanmedian(spr_bp)),
                             capture_bp=cap,
                             cover=cap / float(np.nanmedian(spr_bp))))
    return rows


if __name__ == "__main__":
    fresh = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    allr = []
    for day in DAYS:
        r = run(day, fresh_ms=fresh)
        allr += r
        print(f"{day}: {len(r)} cells", flush=True)
    df = pl.DataFrame(allr)
    df.write_csv(os.path.join(OUT, f"trirv_f{fresh}.csv"))

    print(f"\n=== CROSS REVERTS TO ITS SYNTHETIC?  fresh<{fresh}ms ===")
    print("   ic  = corr(deviation z, forward cross return); NEGATIVE = reverts")
    print("   capture_bp = bp captured in the top decile of |z|")
    print("   cover = capture / the cross's own spread; needs > 1.0 to pay")
    a = (df.group_by("h").agg(
        pl.col("ic").mean().alias("ic"),
        pl.col("capture_bp").mean().alias("capture_bp"),
        pl.col("spread_bp").median().alias("spread_bp"),
        pl.col("cover").mean().alias("cover"),
        pl.len().alias("cells")).sort("h"))
    with pl.Config(tbl_width_chars=150, float_precision=4):
        print(a)

    print("\n=== BEST CROSSES AT h=60 (by cover) ===")
    b = (df.filter(pl.col("h") == 60).group_by(["cross", "legs"]).agg(
        pl.col("ic").mean(), pl.col("capture_bp").mean(),
        pl.col("spread_bp").median(), pl.col("cover").mean(),
        pl.len().alias("days")).sort("cover", descending=True).head(15))
    with pl.Config(tbl_rows=20, tbl_width_chars=160, float_precision=4):
        print(b)
