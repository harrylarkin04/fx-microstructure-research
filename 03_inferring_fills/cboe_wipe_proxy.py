"""
Corrected adverse-selection measurement.

adverse2.py compared adverse selection measured CONDITIONAL on a level wipe
against the UNCONDITIONAL time-weighted half-spread. Wipes cluster when spreads
are tight and the market is moving, so that comparison is biased.

This version records the half-spread AT THE WIPE INSTANT, giving a like-for-like
comparison: what the maker would actually have captured on that fill, against
what the market then did to them.

Sign convention (same as adverse2.py):
  ask wiped -> maker sold at ask(t);  adverse = (mid(t+h) - ask(t)) / pip
  bid wiped -> maker bought at bid(t); adverse = (bid(t) - mid(t+h)) / pip
Positive adverse = market moved against the maker.

Caveat retained: wipes proxy aggressive prints; cancels dilute toward zero, so
adverse remains a LOWER bound.
"""
import glob, os, json
import numpy as np
import polars as pl

FEAT_DIR = os.environ.get("CBOE_FEATURES_DIR", "data/cboe_features")
DEST = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(DEST, exist_ok=True)
HORIZONS_MS = [200, 1000, 5000, 30000]
SYMS = ["EURUSD", "USDJPY", "USDCHF", "EURCHF", "GBPUSD", "AUDUSD", "USDCAD",
        "EURJPY", "NZDUSD", "EURGBP", "GBPJPY", "AUDJPY", "CHFJPY", "EURAUD"]


def measure(sym):
    pip = 0.01 if sym[3:] == "JPY" else 0.0001
    files = sorted(glob.glob(os.path.join(FEAT_DIR, f"{sym}_*.parquet")))
    if not files:
        return None

    adv = {h: [] for h in HORIZONS_MS}   # adverse per wipe event
    cond_hs = []                          # half-spread AT the wipe instant
    uncond_hs, uncond_w = [], []          # time-weighted, unconditional (old method)
    ndays = 0

    for f in files:
        d = pl.read_parquet(f, columns=["ts", "bid_i", "ask_i",
                                        "c_wipe_b", "c_wipe_a"])
        ts = d["ts"].to_numpy()
        if len(ts) < 200:
            continue
        bid = d["bid_i"].to_numpy() / 1e5
        ask = d["ask_i"].to_numpy() / 1e5
        mid = (bid + ask) / 2
        sp = (ask - bid) / pip                      # spread in pips
        wb = np.diff(d["c_wipe_b"].to_numpy(), prepend=0)
        wa = np.diff(d["c_wipe_a"].to_numpy(), prepend=0)

        good = np.isfinite(sp) & (sp > 0) & (sp < 50)

        # unconditional time-weighted half-spread (reproduces adverse2.py)
        dwell = np.diff(ts, append=ts[-1]) / 1e6
        m = good & (dwell >= 0)
        if m.sum() < 100:
            continue
        uncond_hs.append(float(np.average(sp[m] / 2, weights=dwell[m])))
        uncond_w.append(float(dwell[m].sum()))
        ndays += 1

        # wipe events, both sides
        sel_a = (wa > 0) & (wb == 0) & good
        sel_b = (wb > 0) & (wa == 0) & good
        cond_hs.append(np.concatenate([sp[sel_a] / 2, sp[sel_b] / 2]))

        for h in HORIZONS_MS:
            j = np.clip(np.searchsorted(ts, ts + h * 1000, "right") - 1,
                        0, len(ts) - 1)
            fm = mid[j]
            a = (fm[sel_a] - ask[sel_a]) / pip
            b = (bid[sel_b] - fm[sel_b]) / pip
            adv[h].append(np.concatenate([a, b]))

    if not uncond_w:
        return None

    ch = np.concatenate(cond_hs)
    ch = ch[np.isfinite(ch)]
    res = {
        "symbol": sym,
        "days": ndays,
        "n_wipes": int(len(ch)),
        "hs_uncond": float(np.average(uncond_hs, weights=uncond_w)),
        "hs_cond": float(np.mean(ch)),
        "hs_cond_med": float(np.median(ch)),
    }
    for h in HORIZONS_MS:
        x = np.concatenate(adv[h])
        x = x[np.isfinite(x)]
        res[f"adv_{h}"] = float(np.mean(x))
        res[f"se_{h}"] = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        res[f"n_{h}"] = int(len(x))
    return res


if __name__ == "__main__":
    rows = []
    for sym in SYMS:
        r = measure(sym)
        if r:
            rows.append(r)
            print("  %-7s days=%2d wipes=%9d  hs_uncond=%.4f hs_cond=%.4f  adv1s=%.4f"
                  % (sym, r["days"], r["n_wipes"], r["hs_uncond"], r["hs_cond"],
                     r["adv_1000"]), flush=True)
    with open(os.path.join(DEST, "adverse_conditional.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote adverse_conditional.json (%d symbols)" % len(rows))
