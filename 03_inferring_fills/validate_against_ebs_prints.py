"""
Does the WIPE PROXY reproduce adverse selection measured on REAL PRINTS?

The Cboe measurement infers a maker fill from a level disappearing, because that
feed carries no trades. If levels disappear because makers CANCEL ahead of a
move, the proxy overstates adverse selection and the Cboe result is inflated.

EBS gives both on the same tape, so the proxy can be tested directly:

  METHOD A (ground truth): real deal prints with aggressor side.
      aggressor buys  -> a maker sold at the ask
      aggressor sells -> a maker bought at the bid
  METHOD B (the proxy) : a touch level disappearing between consecutive ticks.
      ask moves up   -> treat as "maker sold at the old ask"
      bid moves down -> treat as "maker bought at the old bid"

Both use the same tick series, the same horizons and the same sign convention,
so the ratio adverse/half-spread is directly comparable.

Guards (July recordings have session gaps -- see freshness note):
  * reference quote must be < 200ms old
  * the quote used at t+h must not be stale by more than max(200ms, h/2)
"""
import glob, os, json
import numpy as np
import polars as pl

SP = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(SP, exist_ok=True)
EBS = os.environ.get("EBS_RECORDINGS_DIR", "data/ebs_recordings")
HOR_US = [200_000, 1_000_000, 5_000_000, 30_000_000]     # 200ms, 1s, 5s, 30s
REF_FRESH_US = 200_000
MIN_EVENTS = 2000


def load(pattern, cols):
    frames = []
    for f in sorted(glob.glob(os.path.join(EBS, pattern))):
        try:
            d = pl.read_csv(f, columns=cols, infer_schema_length=2000)
        except Exception:
            continue
        if d.height:
            frames.append(d)
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed")


def to_us(s):
    # ISO-8601 with up to 100ns precision -> microseconds since epoch
    return (s.str.replace(r"(\.\d{6})\d+Z$", r"${1}Z")
             .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
             .dt.timestamp("us"))


def markout(ts, mid, t_ref, h):
    """last mid at or before t_ref+h, with a staleness guard"""
    tol = max(200_000, h // 2)
    j = np.clip(np.searchsorted(ts, t_ref + h, "right") - 1, 0, len(ts) - 1)
    ok = (t_ref + h - ts[j]) <= tol
    return mid[j], ok


def run():
    print("loading ticks...", flush=True)
    T = load("ticks-*.csv", ["timestamp_utc", "symbol", "bid", "ask"])
    print("loading deals...", flush=True)
    D = load("deals-*.csv", ["timestamp_utc", "symbol", "side"])
    T = T.with_columns(to_us(pl.col("timestamp_utc")).alias("tus")).drop_nulls("tus")
    D = D.with_columns(to_us(pl.col("timestamp_utc")).alias("tus")).drop_nulls("tus")
    T = T.unique(subset=["tus", "symbol", "bid", "ask"], maintain_order=True).sort("tus", maintain_order=True)
    D = D.sort("tus", maintain_order=True)
    print("ticks %d  deals %d  symbols %d" % (T.height, D.height, T["symbol"].n_unique()), flush=True)

    out = []
    for sym in sorted(T["symbol"].unique().to_list()):
        pip = 0.01 if sym.endswith("JPY") else 0.0001
        t = T.filter(pl.col("symbol") == sym)
        d = D.filter(pl.col("symbol") == sym)
        if t.height < 5000 or d.height < MIN_EVENTS:
            continue
        ts = t["tus"].to_numpy()
        bid = t["bid"].to_numpy().astype(float)
        ask = t["ask"].to_numpy().astype(float)
        mid = (bid + ask) / 2.0
        sp = (ask - bid) / pip
        valid = np.isfinite(sp) & (sp > 0) & (sp < 50)

        rec = {"symbol": sym, "n_ticks": int(t.height), "n_deals": int(d.height)}

        # ---------- METHOD A: real prints ----------
        dts = d["tus"].to_numpy()
        side = d["side"].str.to_lowercase().to_numpy()
        i = np.searchsorted(ts, dts, "left") - 1           # strictly before the print
        keep = (i >= 0) & ((dts - ts[np.clip(i, 0, len(ts) - 1)]) <= REF_FRESH_US)
        i = np.clip(i, 0, len(ts) - 1)
        keep &= valid[i]
        buy = (side == "buy")
        for h in HOR_US:
            fm, ok = markout(ts, mid, ts[i], h)
            k = keep & ok
            adv = np.where(buy, (fm - ask[i]) / pip, (bid[i] - fm) / pip)
            a = adv[k]
            hs = sp[i][k] / 2.0
            if len(a) >= MIN_EVENTS:
                rec[f"A_adv_{h}"] = float(np.mean(a))
                rec[f"A_hs_{h}"] = float(np.mean(hs))
                rec[f"A_n_{h}"] = int(len(a))
                rec[f"A_se_{h}"] = float(np.std(a, ddof=1) / np.sqrt(len(a)))

        # ---------- METHOD B: wipe proxy ----------
        up = np.zeros(len(ts), bool); up[1:] = ask[1:] > ask[:-1]     # ask level gone
        dn = np.zeros(len(ts), bool); dn[1:] = bid[1:] < bid[:-1]     # bid level gone
        prev = np.arange(len(ts)) - 1
        selA = up & (prev >= 0); selB = dn & (prev >= 0)
        pA = prev[selA]; pB = prev[selB]
        pA = pA[valid[pA]]; pB = pB[valid[pB]]
        for h in HOR_US:
            fmA, okA = markout(ts, mid, ts[pA], h)
            fmB, okB = markout(ts, mid, ts[pB], h)
            a = np.concatenate([((fmA - ask[pA]) / pip)[okA],
                                ((bid[pB] - fmB) / pip)[okB]])
            hs = np.concatenate([(sp[pA] / 2.0)[okA], (sp[pB] / 2.0)[okB]])
            if len(a) >= MIN_EVENTS:
                rec[f"B_adv_{h}"] = float(np.mean(a))
                rec[f"B_hs_{h}"] = float(np.mean(hs))
                rec[f"B_n_{h}"] = int(len(a))

        if f"A_adv_{HOR_US[1]}" in rec and f"B_adv_{HOR_US[1]}" in rec:
            out.append(rec)
            print("  %-8s A: adv %+.4f hs %.4f ratio %.3f (n=%d)   B: adv %+.4f hs %.4f ratio %.3f (n=%d)"
                  % (sym,
                     rec["A_adv_1000000"], rec["A_hs_1000000"],
                     rec["A_adv_1000000"] / rec["A_hs_1000000"], rec["A_n_1000000"],
                     rec["B_adv_1000000"], rec["B_hs_1000000"],
                     rec["B_adv_1000000"] / rec["B_hs_1000000"], rec["B_n_1000000"]),
                  flush=True)

    with open(os.path.join(SP, "validate_wipe.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote validate_wipe.json (%d symbols)" % len(out))


if __name__ == "__main__":
    run()
