"""
Final consolidated result, with the accounting corrected.

The correction that matters: making on one venue and hedging on the other does
NOT leave you flat.  It leaves you long one book and short the other, which is
an OPEN POSITION IN THE BASIS.  Booking (hedge_price - fill_price) as profit at
that instant banks an unrealised basis position as if it were realised.  Since
the coin-margined and USDT-margined perps sit ~9bp apart on BTC, that single
accounting choice is worth about 9bp per trade in whichever direction the fill
happened to come.

So the honest measure is the BASIS-NEUTRAL one: what the quote earned after
removing the basis exposure it opened.
"""
import os

import numpy as np
import polars as pl

import load
import xbt

OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2023-09-12", "2023-11-15", "2024-01-17",
        "2024-02-14", "2024-03-11", "2024-03-20"]
BOOKS = [("BTC", "BTCUSDT", "BTCUSD_PERP", 0.25),
         ("ETH", "ETHUSDT", "ETHUSD_PERP", 4.0)]
FEES = {"VIP0 (retail)": (2.0, 5.0), "VIP5": (1.4, 3.4), "VIP9": (0.9, 2.35),
        "market maker (MM3)": (-0.5, 1.7), "zero fee": (0.0, 0.0)}


def have(day):
    return all(os.path.exists(os.path.join(load.RAW, f"{s}-{k}-{day}.zip"))
               for s in ("BTCUSDT", "BTCUSD_PERP", "ETHUSDT", "ETHUSD_PERP")
               for k in ("bookTicker", "aggTrades"))


rows = []
for day in [d for d in DAYS if have(d)]:
    for asset, mk, hd, clip in BOOKS:
        r = xbt.run(mk, hd, day, clip, lat_ms=5)
        if r is None or len(r["fill"]) == 0:
            continue
        s, f, h, mm, hm = r["side"], r["fill"], r["hedge"], r["mmid"], r["hmid"]
        coin = load.to_base_units(mk, r["qty"], f)
        notl = coin * f
        gross = s * (h - f) / f * 1e4
        B = (hm - mm) / mm * 1e4
        neutral = gross - s * B
        t = load.trades(mk, day)
        venue_usd = float((load.to_base_units(mk, t[2], t[1]) * t[1]).sum())
        rows.append(dict(
            day=day, asset=asset, fills=len(f),
            notional_usd=float(notl.sum()),
            participation_pct=float(notl.sum() / venue_usd * 100),
            gross_bp=float(np.average(gross, weights=notl)),
            basis_bp=float(np.average(B, weights=notl)),
            neutral_bp=float(np.average(neutral, weights=notl)),
            buy_leg_bp=float(np.average(gross[s > 0], weights=notl[s > 0])),
            sell_leg_bp=float(np.average(gross[s < 0], weights=notl[s < 0])),
            pct_buys=float(notl[s > 0].sum() / notl.sum() * 100)))
        print(f"{day} {asset}: gross {rows[-1]['gross_bp']:+.4f}  "
              f"neutral {rows[-1]['neutral_bp']:+.4f}  "
              f"participation {rows[-1]['participation_pct']:.2f}%", flush=True)

df = pl.DataFrame(rows)
df.write_csv(os.path.join(OUT, "final.csv"))
w = pl.col("notional_usd")
nd = df["day"].n_unique()

print("\n=== PER ROUND TRIP, bp of notional (notional-weighted) ===")
agg = (df.group_by("asset").agg(
    pl.col("fills").sum(),
    (w.sum() / 1e9 / nd).alias("our_bn_per_day"),
    pl.col("participation_pct").mean().alias("pct_of_venue"),
    ((pl.col("buy_leg_bp") * w).sum() / w.sum()).alias("buy_leg"),
    ((pl.col("sell_leg_bp") * w).sum() / w.sum()).alias("sell_leg"),
    ((pl.col("gross_bp") * w).sum() / w.sum()).alias("gross"),
    ((pl.col("basis_bp") * w).sum() / w.sum()).alias("basis"),
    ((pl.col("neutral_bp") * w).sum() / w.sum()).alias("BASIS_NEUTRAL"))
    .sort("asset"))
with pl.Config(tbl_rows=10, tbl_width_chars=190, float_precision=4):
    print(agg)

print("\n=== NET AFTER FEES, on the BASIS-NEUTRAL edge ===")
tot_notional = float(df["notional_usd"].sum())
neut = float((df["neutral_bp"] * df["notional_usd"]).sum() / tot_notional)
for name, (mf, tf) in FEES.items():
    net = neut - (mf + tf)
    print(f"  {name:22s} edge {neut:+.4f} - fees {mf+tf:5.2f} = {net:+8.4f} bp"
          f"   ${net/1e4*tot_notional/nd:>14,.0f} /day")
print(f"\n  (traded notional ${tot_notional/nd/1e9:.2f} bn/day across {nd} days)")
