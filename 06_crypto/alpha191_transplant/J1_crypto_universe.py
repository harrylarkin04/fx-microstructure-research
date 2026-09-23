# -*- coding: utf-8 -*-
"""Daily OHLCV for a crypto cross-section, for the Alpha191 transplant.

Thesis: the Alpha191 (GTJA) library was engineered for the retail-dominated
Chinese A-share market. Du, Walter & Ulrich (arXiv 2601.06499, May-2026) show
17 of them survive double-selection LASSO against 151 fundamentals in the S&P
500 and call them "universal behavioural dynamics". If that is right, they
should be STRONGER where retail dominance is more extreme than A-shares - and
crypto is the most retail-driven liquid market there is. Nobody has run that
transplant with valid post-selection inference.

Yahoo chart API, explicit period1/period2 (range=max silently coerces to monthly).
"""
import os
import time
import json
import urllib.request
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(BASE, "data", "crypto")
os.makedirs(DD, exist_ok=True)

TICK = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD",
        "DOT-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
        "ETC-USD", "FIL-USD", "TRX-USD", "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD",
        "HBAR-USD", "EOS-USD", "AAVE-USD", "MKR-USD", "XTZ-USD", "THETA-USD", "AXS-USD",
        "SAND-USD", "MANA-USD", "GRT-USD", "CRV-USD", "COMP-USD", "SNX-USD", "ZEC-USD",
        "DASH-USD", "WAVES-USD", "CHZ-USD", "ENJ-USD", "BAT-USD", "QNT-USD", "KSM-USD",
        "ZIL-USD", "1INCH-USD", "YFI-USD", "UNI-USD", "SUSHI-USD", "RUNE-USD"]
P1 = int(pd.Timestamp("2017-01-01").timestamp())
P2 = int(pd.Timestamp("2026-09-01").timestamp())
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(t):
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s"
           "?period1=%d&period2=%d&interval=1d" % (t, P1, P2))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read().decode())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    d = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s").normalize(),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"]})
    return d.dropna(subset=["close"]).drop_duplicates("date")


ok, bad = [], []
for t in TICK:
    fp = os.path.join(DD, t.replace("-", "") + ".csv")
    if os.path.exists(fp):
        ok.append(t)
        continue
    try:
        d = fetch(t)
        if len(d) > 400:
            d.to_csv(fp, index=False)
            ok.append(t)
        else:
            bad.append((t, "short %d" % len(d)))
    except Exception as e:
        bad.append((t, str(e)[:40]))
    time.sleep(0.35)

print("downloaded %d, failed %d" % (len(ok), len(bad)))
if bad:
    print("  failed: " + ", ".join("%s(%s)" % b for b in bad[:8]))
rows = []
for t in ok:
    fp = os.path.join(DD, t.replace("-", "") + ".csv")
    d = pd.read_csv(fp, parse_dates=["date"])
    rows.append((t, d.date.min().date(), d.date.max().date(), len(d),
                 float(d.volume.tail(250).median() * d.close.tail(250).median() / 1e6)))
S = pd.DataFrame(rows, columns=["tick", "from", "to", "n", "adv_musd"]).sort_values(
    "adv_musd", ascending=False)
print("\ntop by median daily $ volume (last 250d, $mm):")
print(S.head(20).to_string(index=False))
print("\nuniverse with >=$5mm/day and >=800 obs: %d"
      % ((S.adv_musd >= 5) & (S.n >= 800)).sum())
