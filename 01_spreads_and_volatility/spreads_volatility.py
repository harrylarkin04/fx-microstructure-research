"""
Is the 13-second constant a CONSTANT, or is it 1/(trade arrival rate)?

Claim:      half_spread = sigma * sqrt(T),  T ~ 13s, universal.
Rival:      Wyart/Bouchaud -- spread ~ volatility PER TRADE, i.e. T = 1/lambda,
            in which case "13 seconds" is just the mean time between trades and
            the constant is an artefact of the pairs sampled.

Per pair-day:
    RV      = sum of squared mid changes between consecutive quote updates
              (gaps > 300s dropped, so stale crosses cannot inflate it)
    active  = summed accepted gap time
    sigma   = sqrt(RV / active)              price units per sqrt(second)
    hs      = time-weighted mean (ask-bid)/2 price units
    lambda  = deals / active                 trades per second
    T_impl  = (hs / sigma)^2                 seconds

Decisive regression:   log T_impl  ~  a + b * log(1/lambda)
    b ~ 1  -> T is just the trade interval        (Wyart; nothing new)
    b ~ 0  -> T is invariant to trade rate        (a genuine constant)
"""
import glob, os, json
import numpy as np
import polars as pl

SRC = os.environ.get("EBS_L1_DIR", "data/EBS_SPOTFX_L1")
SP = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(SP, exist_ok=True)
COLS = ["date", "time", "pair", "typ", "bid", "ask", "bidvol", "askvol"]
MAX_GAP = 300.0        # seconds; staleness cap
MIN_DEALS = 100
MIN_QUOTES = 500


def day(path):
    d = pl.read_csv(path, has_header=False, new_columns=COLS,
                    infer_schema_length=0)
    d = d.with_columns([
        pl.col("bid").str.strip_chars().cast(pl.Float64, strict=False),
        pl.col("ask").str.strip_chars().cast(pl.Float64, strict=False),
        (pl.col("date") + " " + pl.col("time"))
            .str.to_datetime(format="%m/%d/%y %H:%M:%S", strict=False)
            .dt.timestamp("ms").alias("ts"),
    ]).drop_nulls("ts")

    out = []
    for pair, g in d.group_by("pair"):
        pair = pair[0] if isinstance(pair, tuple) else pair
        if "SM" in pair or "/" not in pair:
            continue
        q = g.filter((pl.col("typ") == "P") & pl.col("bid").is_not_null()
                     & pl.col("ask").is_not_null()).sort("ts")
        if q.height < MIN_QUOTES:
            continue
        ndeals = g.filter(pl.col("typ") == "D").height
        if ndeals < MIN_DEALS:
            continue

        ts = q["ts"].to_numpy() / 1000.0
        bid = q["bid"].to_numpy(); ask = q["ask"].to_numpy()
        mid = (bid + ask) / 2.0
        sp = ask - bid
        ok = np.isfinite(mid) & np.isfinite(sp) & (sp > 0) & (mid > 0)
        ts, mid, sp = ts[ok], mid[ok], sp[ok]
        if len(ts) < MIN_QUOTES:
            continue

        gap = np.diff(ts)
        good = (gap > 0) & (gap <= MAX_GAP)
        if good.sum() < MIN_QUOTES // 2:
            continue
        dm = np.diff(mid)[good]
        active = float(gap[good].sum())
        if active <= 0:
            continue

        rv = float(np.sum(dm ** 2))
        sigma = np.sqrt(rv / active)                       # price per sqrt(s)
        hs = float(np.average(sp[:-1][good], weights=gap[good]) / 2.0)
        lam = ndeals / active
        if sigma <= 0 or hs <= 0 or lam <= 0:
            continue
        out.append({"pair": pair, "day": os.path.basename(path)[12:20],
                    "hs": hs, "sigma": sigma, "lam": lam,
                    "T": (hs / sigma) ** 2, "mid": float(np.mean(mid)),
                    "nq": int(len(ts)), "nd": int(ndeals)})
    return out


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    n, k = X.shape
    s2 = r @ r / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


if __name__ == "__main__":
    rows = []
    files = sorted(glob.glob(os.path.join(SRC, "**", "*.csv.gz"), recursive=True))
    for i, f in enumerate(files):
        try:
            rows += day(f)
        except Exception as e:
            print("  skip", os.path.basename(f), e, flush=True)
        if (i + 1) % 20 == 0:
            print("  %d/%d files, %d pair-days" % (i + 1, len(files), len(rows)), flush=True)
    json.dump(rows, open(os.path.join(SP, "spreads_volatility.json"), "w"))
    print("\npair-days: %d   pairs: %d" % (len(rows), len({r['pair'] for r in rows})))

    T = np.array([r["T"] for r in rows])
    lam = np.array([r["lam"] for r in rows])
    m = np.isfinite(T) & np.isfinite(lam) & (T > 0) & (lam > 0)
    T, lam = T[m], lam[m]
    print("\nimplied T (seconds):  median %.2f   IQR %.2f-%.2f   mean %.2f"
          % (np.median(T), *np.percentile(T, [25, 75]), T.mean()))
    print("   dispersion across pair-days: 95th/5th percentile = %.1fx"
          % (np.percentile(T, 95) / np.percentile(T, 5)))
    print("trade interval 1/lambda (s): median %.2f   IQR %.2f-%.2f"
          % (np.median(1 / lam), *np.percentile(1 / lam, [25, 75])))

    y = np.log(T); x = np.log(1 / lam)
    b, se, r2 = ols(y, np.c_[np.ones(len(x)), x])
    print("\n=== DECISIVE TEST:  log T  ~  a + b log(1/lambda) ===")
    print("   b = %+.4f  (se %.4f, t vs 0 = %+.1f, t vs 1 = %+.1f)   R2 = %.3f  n = %d"
          % (b[1], se[1], b[1] / se[1], (b[1] - 1) / se[1], r2, len(x)))
    print("   b ~ 1 -> Wyart (T is the trade interval).  b ~ 0 -> genuine constant.")

    print("\n=== per-pair implied T (median across days) ===")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], []).append((r["T"], 1 / r["lam"]))
    tab = sorted(((p, np.median([a for a, _ in v]), np.median([b for _, b in v]), len(v))
                  for p, v in pairs.items() if len(v) >= 20),
                 key=lambda z: -z[3])
    print("%-10s %10s %14s %6s" % ("pair", "T (s)", "1/lambda (s)", "days"))
    for p, t, l, n in tab[:20]:
        print("%-10s %10.2f %14.2f %6d" % (p, t, l, n))
    tt = np.array([t for _, t, _, _ in tab])
    print("\ncross-pair T: median %.2f, IQR %.2f-%.2f, ratio max/min %.1fx  (%d pairs)"
          % (np.median(tt), *np.percentile(tt, [25, 75]), tt.max() / tt.min(), len(tt)))

    # ---- within-pair test: does the implied horizon move with trade rate? ----
    hs = np.array([r["hs"] for r in rows]); sg = np.array([r["sigma"] for r in rows])
    lam_all = np.array([r["lam"] for r in rows]); pr = np.array([r["pair"] for r in rows])
    Tall = np.array([r["T"] for r in rows])
    k = np.isfinite(hs) & np.isfinite(sg) & (hs > 0) & (sg > 0) & (lam_all > 0) & (Tall > 0)
    hs, sg, lam_all, pr, Tall = hs[k], sg[k], lam_all[k], pr[k], Tall[k]
    b, se, r2 = ols(np.log(hs), np.c_[np.ones(len(hs)), np.log(sg)])
    print("\n=== half-spread vs volatility ===")
    print("   log hs = %.3f + %.4f log sigma   (se %.4f)   R2 = %.3f   n = %d" % (b[0], b[1], se[1], r2, len(hs)))
    ps = sorted(set(pr)); Dm = np.zeros((len(pr), len(ps)))
    for j, p in enumerate(ps):
        Dm[pr == p, j] = 1
    b, se, r2 = ols(np.log(Tall), np.c_[np.log(1 / lam_all), Dm])
    print("\n=== within-pair (pair fixed effects): log T ~ b log(1/lambda) ===")
    print("   b = %+.4f  (se %.4f)  t vs 0 = %+.1f   t vs 1 = %+.1f   n = %d, %d pairs"
          % (b[0], se[0], b[0] / se[0], (b[0] - 1) / se[0], len(Tall), len(ps)))
