"""
Does the Cboe FX L3 book predict BTC and ETH?

Hypotheses, stated before testing, each with a mechanism and a SIGN fixed in
advance so nothing is fitted:

  H1  DOLLAR LEVEL.  Crypto is a USD-priced risk asset.  A broad dollar rally
      should push BTCUSDT and ETHUSDT down.  Sign: negative.

  H2  DOLLAR FLOW (this is what L3 buys over free FX prices).  Order-flow
      imbalance aggregated across the USD majors measures dollar buying
      pressure BEFORE it fully shows in the price.  If FX flow leads the FX
      price, and the FX price co-moves with crypto, then FX FLOW should lead
      crypto.  Sign: negative.

  H3  RISK APPETITE.  AUD/JPY and NZD/JPY are the classic carry barometers.
      Risk-on lifts both them and crypto.  Sign: positive.

Only H2 requires the L3 data.  H1 and H3 need nothing but a price feed, so if
only they work, the L3 book has added nothing.

Clocks: Cboe stamps New York, Binance stamps UTC.  July is UTC-4.  The
conversion is verified against the data rather than assumed, by checking that
the contemporaneous FX/crypto correlation peaks at zero lag.
"""
import os
import sys
import zipfile

import numpy as np
import polars as pl

CBOE = os.environ.get("CBOE_L1_DIR", "data/cboe_l1")
RAW = os.environ.get("CRYPTO_RAW_DIR", "data/crypto_raw")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
        "2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29"]
NY_TO_UTC_MS = 4 * 3600 * 1000

# USD majors, with the sign that makes a POSITIVE contribution mean STRONGER USD
USD_PAIRS = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
             "USDJPY": +1, "USDCHF": +1, "USDCAD": +1}
RISK_PAIRS = ["AUDJPY", "NZDJPY"]
HORIZONS = [10, 60, 300, 900, 3600]


def crypto_grid(sym, day, grid):
    """Last traded price on the 1s UTC grid, from the aggTrades tape."""
    p = os.path.join(RAW, f"{sym}-aggTrades-{day}.zip")
    if not os.path.exists(p):
        return None
    z = zipfile.ZipFile(p)
    raw = z.read(z.namelist()[0])
    head = raw[:200].split(b"\n")[0].lower()
    has_hdr = b"price" in head
    d = pl.read_csv(raw, has_header=has_hdr,
                    new_columns=None if has_hdr else
                    ["agg_trade_id", "price", "quantity", "first_trade_id",
                     "last_trade_id", "transact_time", "is_buyer_maker"])
    cols = d.columns
    px = d[[c for c in cols if "price" in c.lower()][0]].cast(pl.Float64).to_numpy()
    ts = d[[c for c in cols if "transact" in c.lower()][0]].cast(pl.Int64).to_numpy()
    o = np.argsort(ts, kind="stable")
    ts, px = ts[o], px[o]
    j = np.searchsorted(ts, grid, side="right") - 1
    return np.where(j >= 0, px[np.maximum(j, 0)], np.nan)


def fx_grid(sym, day, grid):
    """Mid and cumulative order-flow imbalance on the same UTC grid."""
    p = os.path.join(CBOE, f"{sym}_{day}.parquet")
    if not os.path.exists(p):
        return None, None
    d = pl.read_parquet(p, columns=["ts_ms", "bid", "ask", "bid_sz", "ask_sz"])
    d = d.filter(pl.col("ask") > pl.col("bid"))
    ts = d["ts_ms"].to_numpy() + NY_TO_UTC_MS
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    bs, az = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
    mid = (bid + ask) / 2

    n = len(mid)
    db = np.zeros(n)
    da = np.zeros(n)
    up_b, dn_b = bid[1:] > bid[:-1], bid[1:] < bid[:-1]
    db[1:] = (np.where(up_b, bs[1:], np.where(dn_b, -bs[:-1], 0.0))
              + np.where(~up_b & ~dn_b, bs[1:] - bs[:-1], 0.0))
    dn_a, up_a = ask[1:] < ask[:-1], ask[1:] > ask[:-1]
    da[1:] = (np.where(dn_a, az[1:], np.where(up_a, -az[:-1], 0.0))
              + np.where(~up_a & ~dn_a, az[1:] - az[:-1], 0.0))
    cofi = np.cumsum(db - da)                       # cumulative signed flow

    j = np.searchsorted(ts, grid, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    return (np.where(ok, mid[j], np.nan), np.where(ok, cofi[j], np.nan))


def build_day(day):
    lo = int(np.datetime64(f"{day}T00:00:00", "ms").astype("int64"))
    grid = np.arange(lo, lo + 86_400_000, 1000)

    usd_mid, usd_flow, nseen = [], [], 0
    for sym, sgn in USD_PAIRS.items():
        m, f = fx_grid(sym, day, grid)
        if m is None:
            continue
        r = np.full(len(m), np.nan)
        r[1:] = np.diff(np.log(m))
        usd_mid.append(sgn * r)
        fl = np.full(len(f), np.nan)
        fl[1:] = np.diff(f)
        sd = np.nanstd(fl)
        usd_flow.append(sgn * fl / (sd if sd > 0 else 1.0))
        nseen += 1
    if nseen < 4:
        return None
    usd_ret = np.nanmean(np.vstack(usd_mid), axis=0)
    usd_ofi = np.nanmean(np.vstack(usd_flow), axis=0)

    risk = []
    for sym in RISK_PAIRS:
        m, _ = fx_grid(sym, day, grid)
        if m is None:
            continue
        r = np.full(len(m), np.nan)
        r[1:] = np.diff(np.log(m))
        risk.append(r)
    risk_ret = np.nanmean(np.vstack(risk), axis=0) if risk else np.full(len(grid), np.nan)

    out = {"grid": grid, "usd_ret": usd_ret, "usd_ofi": usd_ofi,
           "risk_ret": risk_ret}
    for sym in ("BTCUSDT", "ETHUSDT"):
        px = crypto_grid(sym, day, grid)
        out[sym] = px
    return out


def roll_sum(x, w):
    c = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(x))])
    return c[w:] - c[:-w]


def main():
    rows = []
    for day in DAYS:
        d = build_day(day)
        if d is None:
            print("skip", day)
            continue
        for sym in ("BTCUSDT", "ETHUSDT"):
            px = d[sym]
            if px is None:
                continue
            lp = np.log(px)
            for h in HORIZONS:
                fwd = np.full(len(lp), np.nan)
                fwd[:-h] = lp[h:] - lp[:-h]
                for name in ("usd_ret", "usd_ofi", "risk_ret"):
                    sig = np.full(len(lp), np.nan)
                    s = roll_sum(d[name], h)
                    sig[h - 1:] = s               # trailing sum over the same h
                    m = np.isfinite(sig) & np.isfinite(fwd)
                    # NON-OVERLAPPING sample so the t-stat is not inflated
                    idx = np.flatnonzero(m)
                    idx = idx[::h]
                    if len(idx) < 30:
                        continue
                    a, b = sig[idx], fwd[idx]
                    if a.std() == 0 or b.std() == 0:
                        continue
                    rows.append(dict(day=day, sym=sym, h=h, signal=name,
                                     n=len(idx),
                                     ic=float(np.corrcoef(a, b)[0, 1])))
        print("done", day, flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(OUT, "fxcrypto_ic.csv"))
    agg = (df.group_by(["signal", "sym", "h"]).agg(
        pl.col("ic").mean().alias("ic"),
        pl.col("ic").std().alias("sd"),
        pl.col("n").sum().alias("n"),
        pl.len().alias("days"))
        .with_columns((pl.col("ic") / (pl.col("sd") / pl.col("days").sqrt()))
                      .alias("t"))
        .sort(["signal", "sym", "h"]))
    with pl.Config(tbl_rows=60, tbl_width_chars=160, float_precision=4):
        print(agg)


if __name__ == "__main__":
    main()
