"""Parse the Binance daily zips into numpy arrays."""
import io
import os
import zipfile

import numpy as np
import polars as pl

RAW = os.environ.get("CRYPTO_RAW_DIR", "data/crypto_raw")
# contract multipliers for the coin-margined perps (USD notional per contract)
CM_MULT = {"BTCUSD_PERP": 100.0, "ETHUSD_PERP": 10.0}


def _open(sym, kind, day):
    p = os.path.join(RAW, f"{sym}-{kind}-{day}.zip")
    if not os.path.exists(p):
        return None
    z = zipfile.ZipFile(p)
    return z.read(z.namelist()[0])


def book(sym, day):
    """-> ts_ms, bid, bid_qty, ask, ask_qty  (sorted, sane rows only)."""
    raw = _open(sym, "bookTicker", day)
    if raw is None:
        return None
    d = pl.read_csv(io.BytesIO(raw), schema_overrides={
        "update_id": pl.Int64, "best_bid_price": pl.Float64,
        "best_bid_qty": pl.Float64, "best_ask_price": pl.Float64,
        "best_ask_qty": pl.Float64, "transaction_time": pl.Int64,
        "event_time": pl.Int64})
    d = (d.filter((pl.col("best_ask_price") > pl.col("best_bid_price"))
                  & (pl.col("best_bid_price") > 0)
                  & (pl.col("best_bid_qty") > 0) & (pl.col("best_ask_qty") > 0))
          .sort("transaction_time"))
    return (d["transaction_time"].to_numpy(),
            d["best_bid_price"].to_numpy(), d["best_bid_qty"].to_numpy(),
            d["best_ask_price"].to_numpy(), d["best_ask_qty"].to_numpy())


def trades(sym, day):
    """
    -> ts_ms, price, qty, aggressor  where aggressor = -1 for a market SELL
    and +1 for a market BUY.

    Binance flags `is_buyer_maker`: when it is true the resting order was a
    BID, so the aggressor was a seller.  This is the field that makes a passive
    fill decidable rather than assumed.
    """
    raw = _open(sym, "aggTrades", day)
    if raw is None:
        return None
    d = pl.read_csv(io.BytesIO(raw), schema_overrides={
        "agg_trade_id": pl.Int64, "price": pl.Float64, "quantity": pl.Float64,
        "first_trade_id": pl.Int64, "last_trade_id": pl.Int64,
        "transact_time": pl.Int64, "is_buyer_maker": pl.Boolean})
    d = d.filter(pl.col("price") > 0).sort("transact_time")
    agg = np.where(d["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    return (d["transact_time"].to_numpy(), d["price"].to_numpy(),
            d["quantity"].to_numpy(), agg)


def to_base_units(sym, qty, price):
    """CM quantities are CONTRACTS of fixed USD notional; convert to coin."""
    m = CM_MULT.get(sym)
    return qty * m / price if m else qty
