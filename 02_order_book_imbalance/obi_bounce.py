"""
Order-book imbalance on Cboe FX: how much of its predictive power is bounce?

Section 0 is the headline. Sections 1-3 split by touch freshness.

Flicker test, corrected.

v1 bucketed on min(age_b, age_a), which is 0 whenever EITHER side is fresh and
collapsed half the sample. Ages are in SECONDS (median ~45ms, p99 ~20s).

Kept separate here:
  1. IC of obi1 within each (bid fresh/stale) x (ask fresh/stale) cell
  2. a freshness-imbalance signal of its own: which side is the fresher quote
  3. everything also residualised on bwd200, since v1 showed 86% of obi1's raw
     IC is recent-return bounce
"""
import glob, os
import numpy as np
import polars as pl

FEAT = os.environ.get("CBOE_FEATURES_DIR", "data/cboe_features")
FWD = ["fwd50", "fwd200", "fwd1000", "fwd5000"]
CUT = 0.05          # seconds; ~median touch age


def ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2000:
        return np.nan, 0
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan, 0
    return float(np.corrcoef(x, y)[0, 1]), int(m.sum())


def resid_ic(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 2000:
        return np.nan, 0
    X = np.c_[np.ones(m.sum()), z[m]]
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    return ic(rx, ry)


cols = ["obi1", "age_b", "age_a", "b1", "a1", "bwd200"] + FWD
D = pl.concat([pl.read_parquet(f, columns=cols)
               for f in sorted(glob.glob(os.path.join(FEAT, "*.parquet")))],
              how="vertical_relaxed")
print("pooled rows: %s\n" % f"{D.height:,}", flush=True)

obi = D["obi1"].to_numpy()
ab, aa = D["age_b"].to_numpy(), D["age_a"].to_numpy()
bwd = D["bwd200"].to_numpy()
fresh_b, fresh_a = ab <= CUT, aa <= CUT


print("=== 0. POOLED IC of obi1: raw | residualised on the preceding 200ms return ===")
for k in FWD:
    y = D[k].to_numpy()
    r1, n = ic(obi, y)
    r2, _ = resid_ic(obi, y, bwd)
    print("   %-8s raw %+.4f   bounce-residualised %+.4f   (n=%s)" % (k, r1, r2, f"{n:,}"))
print()

cells = [("both fresh", fresh_b & fresh_a),
         ("bid fresh / ask stale", fresh_b & ~fresh_a),
         ("bid stale / ask fresh", ~fresh_b & fresh_a),
         ("both stale", ~fresh_b & ~fresh_a)]

print("=== 1. IC of obi1 by touch freshness (raw | bounce-residualised) ===")
print("%-24s %9s %s" % ("cell", "share", "  ".join("%18s" % k for k in FWD)))
for name, m in cells:
    parts = []
    for k in FWD:
        y = D[k].to_numpy()
        r1, _ = ic(obi[m], y[m])
        r2, _ = resid_ic(obi[m], y[m], bwd[m])
        parts.append("%8.4f|%8.4f" % (r1, r2))
    print("%-24s %8.1f%% %s" % (name, 100 * m.mean(), "  ".join(parts)))

print("\n=== 2. freshness imbalance as its own signal ===")
eps = 1e-6
fi = (np.log1p(aa) - np.log1p(ab)) / (np.log1p(aa) + np.log1p(ab) + eps)
print("%-24s %s" % ("", "  ".join("%18s" % k for k in FWD)))
parts = []
for k in FWD:
    y = D[k].to_numpy()
    r1, n = ic(fi, y)
    r2, _ = resid_ic(fi, y, bwd)
    parts.append("%8.4f|%8.4f" % (r1, r2))
print("%-24s %s" % ("freshness imbalance", "  ".join(parts)))

print("\n=== 3. is freshness imbalance independent of obi1? ===")
r, n = ic(fi, obi)
print("   corr(freshness imbalance, obi1) = %+.4f  (n=%s)" % (r, f"{n:,}"))
print("\n   combined: IC of obi1 residualised on freshness imbalance, and vice versa")
for k in FWD:
    y = D[k].to_numpy()
    a, _ = resid_ic(obi, y, fi)
    b, _ = resid_ic(fi, y, obi)
    print("   %-8s obi1|fi %+.4f    fi|obi1 %+.4f" % (k, a, b))
