"""Run the cross-venue maker/hedge backtest and report it honestly."""
import os
import sys

import numpy as np
import polars as pl

import load
import xbt

OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
DAYS = ["2023-09-12", "2023-11-15", "2024-01-17",
        "2024-02-14", "2024-03-11", "2024-03-20"]

# (asset, maker venue, hedge venue, clip in the maker venue's own units)
BOOKS = [("BTC", "BTCUSD_PERP", "BTCUSDT", 100.0),     # 100 contracts = $10k
         ("BTC", "BTCUSDT", "BTCUSD_PERP", 0.25),      # 0.25 BTC ~ $10k
         ("ETH", "ETHUSD_PERP", "ETHUSDT", 1000.0),    # 1000 contracts = $10k
         ("ETH", "ETHUSDT", "ETHUSD_PERP", 4.0)]       # 4 ETH ~ $10k

# Binance futures fee tiers, basis points of notional.
# (maker, taker); negative maker = rebate.
FEES = {"VIP0 (retail)": (2.0, 5.0),
        "VIP5": (1.4, 3.4),
        "VIP9": (0.9, 2.35),
        "market maker (MM3)": (-0.5, 1.7),
        "zero fee (hypothetical)": (0.0, 0.0)}


def have(day):
    return all(os.path.exists(os.path.join(
        load.RAW, f"{s}-{k}-{day}.zip"))
        for s in ("BTCUSDT", "BTCUSD_PERP", "ETHUSDT", "ETHUSD_PERP")
        for k in ("bookTicker", "aggTrades"))


def main(lat_ms=5):
    os.makedirs(OUT, exist_ok=True)
    days = [d for d in DAYS if have(d)]
    print(f"days with complete data: {days}\n")
    rows = []
    for day in days:
        for asset, mk, hd, clip in BOOKS:
            r = xbt.run(mk, hd, day, clip, lat_ms=lat_ms)
            if r is None:
                continue
            base = xbt.pnl(r, 0.0, 0.0)
            if base["n"] == 0:
                continue
            row = dict(day=day, asset=asset, maker=mk, hedge=hd,
                       fills=base["n"], notional_usd=base["notional"],
                       gross_bp=base["gross_bp"],
                       half_earned_bp=base["half_earned_bp"],
                       basis_drift_bp=base["basis_drift_bp"],
                       half_paid_bp=base["half_paid_bp"])
            for name, (mf, tf) in FEES.items():
                p = xbt.pnl(r, mf, tf)
                row[f"net_bp[{name}]"] = p["net_bp"]
                row[f"usd[{name}]"] = p["net_usd"]
            rows.append(row)
            print(f"{day} {asset} make {mk:12s} hedge {hd:12s} "
                  f"fills {base['n']:>7,}  gross {base['gross_bp']:+.4f}bp",
                  flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(OUT, f"xbt_lat{lat_ms}.csv"))

    print("\n=== ROUND-TRIP DECOMPOSITION (bp of notional, notional-weighted) ===")
    w = pl.col("notional_usd")
    agg = (df.group_by(["asset", "maker", "hedge"]).agg(
        pl.col("fills").sum(),
        (w.sum() / 1e9).alias("notional_bn"),
        ((pl.col("half_earned_bp") * w).sum() / w.sum()).alias("half_earned"),
        ((pl.col("basis_drift_bp") * w).sum() / w.sum()).alias("basis_drift"),
        ((pl.col("half_paid_bp") * w).sum() / w.sum()).alias("half_paid"),
        ((pl.col("gross_bp") * w).sum() / w.sum()).alias("gross_bp"))
        .sort("gross_bp", descending=True))
    with pl.Config(tbl_rows=20, tbl_width_chars=190, float_precision=4):
        print(agg)

    print("\n=== NET AFTER FEES ($/day, all four books combined) ===")
    nd = len(days)
    for name in FEES:
        tot = float(df[f"usd[{name}]"].sum())
        bp = float((df[f"net_bp[{name}]"] * df["notional_usd"]).sum()
                   / df["notional_usd"].sum())
        print(f"  {name:26s} {bp:+8.4f} bp   ${tot/nd:>14,.0f} /day")
    print(f"\n  traded notional: ${float(df['notional_usd'].sum())/nd/1e9:.2f} bn/day"
          f"   fills: {int(df['fills'].sum())/nd:,.0f}/day")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
