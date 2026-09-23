"""
Download Binance public market data for the cross-venue test.

Venues (both are real, separately-matched books on the same underlying):
  UM  = USD-margined perpetual   (BTCUSDT, ETHUSDT)      - the deep/tight leg
  CM  = coin-margined perpetual  (BTCUSD_PERP, ...)      - the wider leg

For each we take:
  bookTicker  best bid/ask with sizes, event-time stamped  -> where to quote
  aggTrades   every aggressive trade with the MAKER SIDE   -> who got filled

That second file is the point of doing this in crypto: on Cboe there was no
trade tape, so a passive fill could only be bracketed.  Here the tape is public
and signed, so a maker fill is DETERMINED, not assumed.

Days are deliberately spread across regimes (quiet autumn 2023, the Jan 2024
ETF period, the March 2024 all-time-high and its selloff) rather than taken as
one consecutive block, so the result cannot be a single-regime artefact.
"""
import os
import sys
import urllib.request
import concurrent.futures as cf

BASE = "https://data.binance.vision/data"
RAW = os.environ.get("CRYPTO_RAW_DIR", "data/crypto_raw")
DAYS = ["2023-09-12", "2023-11-15", "2024-01-17",
        "2024-02-14", "2024-03-11", "2024-03-20"]

# (market_path, symbol)
VENUES = [("futures/um", "BTCUSDT"), ("futures/cm", "BTCUSD_PERP"),
          ("futures/um", "ETHUSDT"), ("futures/cm", "ETHUSD_PERP")]
KINDS = ["bookTicker", "aggTrades"]


def url_for(market, sym, kind, day):
    return f"{BASE}/{market}/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"


def local_for(sym, kind, day):
    return os.path.join(RAW, f"{sym}-{kind}-{day}.zip")


def get(job):
    market, sym, kind, day = job
    dst = local_for(sym, kind, day)
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return f"cached {os.path.basename(dst)}"
    u = url_for(market, sym, kind, day)
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(u, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dst)
        return f"ok     {os.path.basename(dst)}  {os.path.getsize(dst)/1e6:.0f} MB"
    except Exception as e:                                       # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"FAIL   {os.path.basename(dst)}  {type(e).__name__}: {e}"


def main():
    os.makedirs(RAW, exist_ok=True)
    jobs = [(m, s, k, d) for d in DAYS for (m, s) in VENUES for k in KINDS]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(get, jobs):
            print(r, flush=True)
    tot = sum(os.path.getsize(os.path.join(RAW, f))
              for f in os.listdir(RAW) if f.endswith(".zip"))
    print(f"\ntotal {tot/1e9:.2f} GB in {RAW}")


if __name__ == "__main__":
    main()
