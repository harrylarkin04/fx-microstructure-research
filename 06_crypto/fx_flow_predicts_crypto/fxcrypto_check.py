"""
Three checks on the FX -> crypto result, and then the only question that
matters: how many basis points is it worth against the fee?

 1 LAG PROFILE.  Does the signal LEAD crypto, and does the correlation peak at
   zero lag?  A peak away from zero means the New York -> UTC conversion is
   wrong, which would invent the whole effect.

 2 SKIP TEST.  The crypto price on the grid is the last TRADE, so it can be
   stale.  If the signal only predicts the very next instant it is measuring
   crypto catching up to information already public in FX, which is not
   tradeable.  Re-run predicting [t+skip, t+skip+h] instead of [t, t+h].

 3 PLACEBO.  Shift the FX series by a whole day.  Any surviving IC is spurious.

 4 ECONOMICS.  Convert the IC into expected basis points per trade and set it
   against the crypto round-trip fee.
"""
import os
import sys

import numpy as np
import polars as pl

import fxcrypto as F


def series(day):
    d = F.build_day(day)
    if d is None:
        return None
    return d


def ic_at(sig, fwd, step):
    m = np.isfinite(sig) & np.isfinite(fwd)
    idx = np.flatnonzero(m)[::step]
    if len(idx) < 30:
        return np.nan
    a, b = sig[idx], fwd[idx]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    days = F.DAYS
    data = {}
    for day in days:
        d = series(day)
        if d is not None:
            data[day] = d
    print(f"days loaded: {len(data)}\n")

    H = 10
    NAMES = ["usd_ofi", "usd_ret", "risk_ret"]

    # ---------------- 1. lag profile (also validates the clock) -------------
    print("=== 1. LAG PROFILE, BTCUSDT, h=10s ===")
    print("   FX shifted by L seconds; L<0 = FX leads crypto")
    print("   lag(s)  " + "  ".join(f"{n:>9}" for n in NAMES))
    for L in (-120, -60, -30, -10, 0, 10, 30, 60, 120):
        out = []
        for name in NAMES:
            ics = []
            for day, d in data.items():
                px = d["BTCUSDT"]
                if px is None:
                    continue
                lp = np.log(px)
                fwd = np.full(len(lp), np.nan)
                fwd[:-H] = lp[H:] - lp[:-H]
                s = F.roll_sum(d[name], H)
                sig = np.full(len(lp), np.nan)
                sig[H - 1:] = s
                sig = np.roll(sig, L)
                if L > 0:
                    sig[:L] = np.nan
                elif L < 0:
                    sig[L:] = np.nan
                ics.append(ic_at(sig, fwd, H))
            out.append(np.nanmean(ics))
        print(f"   {L:>6}  " + "  ".join(f"{v:>+9.4f}" for v in out))

    # ---------------- 2. skip test -----------------------------------------
    print("\n=== 2. SKIP TEST, BTCUSDT, h=10s (is it just crypto catching up?) ===")
    print("   skip(s) " + "  ".join(f"{n:>9}" for n in NAMES))
    for skip in (0, 1, 2, 5, 10):
        out = []
        for name in NAMES:
            ics = []
            for day, d in data.items():
                px = d["BTCUSDT"]
                if px is None:
                    continue
                lp = np.log(px)
                fwd = np.full(len(lp), np.nan)
                end = len(lp) - H - skip
                fwd[:end] = lp[skip + H:skip + H + end] - lp[skip:skip + end]
                s = F.roll_sum(d[name], H)
                sig = np.full(len(lp), np.nan)
                sig[H - 1:] = s
                ics.append(ic_at(sig, fwd, H))
            out.append(np.nanmean(ics))
        print(f"   {skip:>6}  " + "  ".join(f"{v:>+9.4f}" for v in out))

    # ---------------- 3. placebo -------------------------------------------
    print("\n=== 3. PLACEBO: FX from the WRONG day ===")
    dl = list(data)
    for name in NAMES:
        real, fake = [], []
        for i, day in enumerate(dl):
            d = data[day]
            other = data[dl[(i + 1) % len(dl)]]
            px = d["BTCUSDT"]
            if px is None:
                continue
            lp = np.log(px)
            fwd = np.full(len(lp), np.nan)
            fwd[:-H] = lp[H:] - lp[:-H]
            for src, acc in ((d, real), (other, fake)):
                s = F.roll_sum(src[name], H)
                sig = np.full(len(lp), np.nan)
                k = min(len(sig) - (H - 1), len(s))
                sig[H - 1:H - 1 + k] = s[:k]
                acc.append(ic_at(sig, fwd, H))
        print(f"   {name:>9}: real {np.nanmean(real):+.4f}   "
              f"placebo {np.nanmean(fake):+.4f}")

    # ---------------- 4. economics -----------------------------------------
    print("\n=== 4. ECONOMICS: what is the signal worth, in basis points? ===")
    for sym in ("BTCUSDT", "ETHUSDT"):
        vols, ics = [], []
        for day, d in data.items():
            px = d[sym]
            if px is None:
                continue
            lp = np.log(px)
            fwd = np.full(len(lp), np.nan)
            fwd[:-H] = lp[H:] - lp[:-H]
            vols.append(np.nanstd(fwd) * 1e4)
            s = F.roll_sum(d["usd_ofi"], H)
            sig = np.full(len(lp), np.nan)
            sig[H - 1:] = s
            ics.append(ic_at(sig, fwd, H))
        vol = float(np.nanmean(vols))
        ic = abs(float(np.nanmean(ics)))
        print(f"   {sym}: 10s move sd {vol:.2f} bp,  |IC| {ic:.4f}")
        for k in (1, 2, 3):
            print(f"      expected move at {k}-sigma signal: "
                  f"{k * ic * vol:.4f} bp")
        print(f"      round-trip taker fee, top tier:   3.20 bp")
        print(f"      best case ({3 * ic * vol:.4f} bp) is "
              f"{3.20 / (3 * ic * vol):.0f}x too small\n")


if __name__ == "__main__":
    main()
