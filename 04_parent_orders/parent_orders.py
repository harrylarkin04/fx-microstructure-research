"""
Parent-order reconstruction from anonymous EBS child prints.

EBS publishes individual deal prints ("children") but never the parent order
behind them. Large orders are worked as a sequence of same-side prints, so a
parent can be reconstructed by clustering same-side prints that arrive close
together. The question: does the adverse selection a passive maker suffers
depend on where in that sequence it was filled?

Data: EBS Level-1 vendor tape (1-second stamps, quotes P and deals D).

Deal direction (vendor-tape fix). Inferring direction from which size field is
populated is wrong for ~19% of rows: some rows aggregate two prints, and some
prices match neither prevailing quote. Kept here: rows with exactly one side
populated AND whose price equals the prevailing quote on that side.
  ask field populated, price == prevailing ask  -> buyer-initiated
  bid field populated, price == prevailing bid  -> seller-initiated

Parents: same-side prints, a new parent whenever the gap exceeds GAP seconds.
Child number = position of the print within its parent (1 = first / isolated).

Maker markout at +H seconds, in basis points (x100 = USD per 1M notional):
  buyer-initiated at p  (maker sold at p):   (p - mid[t+H]) / p
  seller-initiated at p (maker bought at p): (mid[t+H] - p) / p
Positive = the maker made money on the fill over H seconds.
"""
import glob, os, json
import numpy as np
import polars as pl

SRC = os.environ.get("EBS_L1_DIR", "data/EBS_SPOTFX_L1")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUT, exist_ok=True)
COLS = ["date", "time", "pair", "typ", "bid", "ask", "bidvol", "askvol"]
GAP = 5            # seconds between same-side prints that still count as one parent
HORIZONS = [5, 30]  # markout horizons, seconds
BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
           (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**9, "21+")]
MIN_DEALS_DAY = 200


def load_day(path):
    d = pl.read_csv(path, has_header=False, new_columns=COLS, infer_schema_length=0)
    return d.with_columns([
        pl.col("bid").str.strip_chars().cast(pl.Float64, strict=False),
        pl.col("ask").str.strip_chars().cast(pl.Float64, strict=False),
        pl.col("bidvol").str.strip_chars().cast(pl.Float64, strict=False),
        pl.col("askvol").str.strip_chars().cast(pl.Float64, strict=False),
        (pl.col("date") + " " + pl.col("time"))
            .str.to_datetime(format="%m/%d/%y %H:%M:%S", strict=False)
            .dt.timestamp("ms").alias("ts"),
    ]).drop_nulls("ts")


def child_numbers(t):
    """t sorted seconds -> 1-based position within same-side cluster (gap > GAP starts new)."""
    n = np.ones(len(t), dtype=np.int64)
    for i in range(1, len(t)):
        n[i] = n[i - 1] + 1 if t[i] - t[i - 1] <= GAP else 1
    return n


def one_pair(g):
    q = g.filter((pl.col("typ") == "P") & pl.col("bid").is_not_null() & pl.col("ask").is_not_null()
                 & (pl.col("ask") > pl.col("bid"))).sort("ts")
    if q.height < 500:
        return None
    qt = q["ts"].to_numpy() // 1000
    qb, qa = q["bid"].to_numpy(), q["ask"].to_numpy()
    qm = (qb + qa) / 2

    d = g.filter(pl.col("typ") == "D").sort("ts")
    if d.height < MIN_DEALS_DAY:
        return None
    t = d["ts"].to_numpy() // 1000
    pb, pa = d["bid"].to_numpy(), d["ask"].to_numpy()
    vb, va = d["bidvol"].to_numpy(), d["askvol"].to_numpy()
    has_b = np.isfinite(pb) & (np.nan_to_num(vb) > 0)
    has_a = np.isfinite(pa) & (np.nan_to_num(va) > 0)

    i = np.searchsorted(qt, t, "left") - 1                # prevailing quote strictly before the print second
    ok = i >= 0
    i = np.clip(i, 0, None)
    tol = 1e-9
    buy = ok & has_a & ~has_b & (np.abs(pa - qa[i]) < tol)
    sell = ok & has_b & ~has_a & (np.abs(pb - qb[i]) < tol)
    kept_share = float((buy | sell).sum() / max(1, len(t)))

    rows = []
    for side_mask, price, vol, sgn in ((buy, pa, va, +1), (sell, pb, vb, -1)):
        idx = np.where(side_mask)[0]
        if len(idx) < 2:
            continue
        tt = t[idx]
        cn = child_numbers(tt)
        p = price[idx]
        rec = {"child": cn, "size": vol[idx]}
        for h in HORIZONS:
            j = np.searchsorted(qt, tt + h, "right") - 1
            m = qm[np.clip(j, 0, None)]
            # sgn=+1: aggressor bought, maker sold -> maker gains if mid falls
            rec[f"mk{h}"] = sgn * (p - m) / p * 1e4
        rows.append(rec)
    if not rows:
        return None
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out["kept_share"] = kept_share
    return out


def main():
    files = sorted(glob.glob(os.path.join(SRC, "**", "*.csv.gz"), recursive=True))
    per_pair = {}
    kept = []
    for n_f, f in enumerate(files):
        try:
            D = load_day(f)
        except Exception as e:
            print("skip", os.path.basename(f), e, flush=True); continue
        for pair, g in D.group_by("pair"):
            pair = pair[0] if isinstance(pair, tuple) else pair
            if "SM" in pair or "/" not in pair:
                continue
            r = one_pair(g)
            if r is None:
                continue
            kept.append(r.pop("kept_share"))
            per_pair.setdefault(pair, []).append(r)
        if (n_f + 1) % 20 == 0:
            print("  %d/%d files" % (n_f + 1, len(files)), flush=True)

    res = {"gap_s": GAP, "horizons_s": HORIZONS, "direction_kept_share_median": float(np.median(kept)),
           "pairs": {}, "pooled": {}}
    pooled = {k: [] for k in ["child", "size"] + [f"mk{h}" for h in HORIZONS]}
    for pair, lst in per_pair.items():
        cat = {k: np.concatenate([x[k] for x in lst]) for k in lst[0]}
        if len(cat["child"]) < 5000:
            continue
        for k in pooled:
            pooled[k].append(cat[k])
        tab = {}
        for lo, hi, lab in BUCKETS:
            m = (cat["child"] >= lo) & (cat["child"] <= hi)
            if m.sum() >= 50:
                tab[lab] = {"n": int(m.sum()), "mk5_bp": float(np.mean(cat["mk5"][m]))}
        res["pairs"][pair] = {"n": int(len(cat["child"])), "buckets": tab}

    P = {k: np.concatenate(v) for k, v in pooled.items()}
    res["n_fills"] = int(len(P["child"]))
    for h in HORIZONS:
        tab = {}
        for lo, hi, lab in BUCKETS:
            m = (P["child"] >= lo) & (P["child"] <= hi)
            x = P[f"mk{h}"][m]
            tab[lab] = {"n": int(m.sum()), "mean_bp": float(np.mean(x)),
                        "se_bp": float(np.std(x, ddof=1) / np.sqrt(len(x))),
                        "median_size_m": float(np.median(P["size"][m]) / 1e6)}
        res["pooled"][f"h{h}"] = tab
    # size control: only the most common clip size (1M)
    one = P["size"] == 1e6
    tab = {}
    for lo, hi, lab in BUCKETS:
        m = one & (P["child"] >= lo) & (P["child"] <= hi)
        if m.sum() >= 100:
            tab[lab] = {"n": int(m.sum()), "mean_bp": float(np.mean(P["mk5"][m]))}
    res["size_controlled_1m_h5"] = tab
    # monotonicity per pair (Spearman of bucket order vs markout)
    mono = []
    for pair, v in res["pairs"].items():
        b = [v["buckets"][lab]["mk5_bp"] for _, _, lab in BUCKETS if lab in v["buckets"]]
        if len(b) >= 4:
            r = np.corrcoef(np.argsort(np.argsort(b)), np.arange(len(b)))[0, 1]
            mono.append((pair, float(r)))
    res["per_pair_rank_corr"] = dict(mono)

    json.dump(res, open(os.path.join(OUT, "parent_orders.json"), "w"), indent=1)
    print("\nfills kept: %s  pairs: %d  direction-kept share (median day): %.1f%%"
          % (f"{res['n_fills']:,}", len(res["pairs"]), 100 * res["direction_kept_share_median"]))
    print("\nMAKER MARKOUT BY CHILD NUMBER (bp; x100 = USD per 1M)")
    for h in HORIZONS:
        print("  +%ds: " % h + "  ".join("%s:%+.3f(n=%d)" % (k, v["mean_bp"], v["n"])
                                           for k, v in res["pooled"][f"h{h}"].items()))
    print("  size-controlled (1M prints, +5s): " + "  ".join(
        "%s:%+.3f" % (k, v["mean_bp"]) for k, v in res["size_controlled_1m_h5"].items()))
    print("  median print size by bucket (M): " + "  ".join(
        "%s:%.1f" % (k, v["median_size_m"]) for k, v in res["pooled"]["h5"].items()))
    print("  per-pair rank corr (child bucket vs markout): " + "  ".join(
        "%s:%+.2f" % (k, v) for k, v in res["per_pair_rank_corr"].items()))


if __name__ == "__main__":
    main()
