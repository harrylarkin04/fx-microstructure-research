# 02 · Order-book imbalance

**Question.** Top-of-book imbalance, (bid size − ask size) / (bid size + ask size), is a standard short-horizon predictor of price. How much of its predictive power is information about the next move, and how much is the previous move reversing?

**Data.** Cboe FX order-book features for 14 pairs over 8 sessions (21–30 July 2026), 31.4 million observations, with forward mid returns at 50 ms, 200 ms, 1 s and 5 s.

**Method.** The correlation (IC) between imbalance and the forward return, first raw and then after removing the part of both that is explained by the mid return over the previous 200 ms. The same comparison is repeated split by how recently each side of the touch was refreshed.

**Results.**

| Horizon | Raw IC | After removing the previous return |
|---|---|---|
| 50 ms | +0.0140 | +0.0020 |
| 1 s | +0.0145 | +0.0049 |

About 86% of the raw IC at 50 ms is bounce.

Split by touch freshness, the raw IC is positive when both sides are equally fresh and negative when one side is fresh and the other stale. That looks like two populations cancelling each other out. After removing the previous return every cell is positive: a fresh bid against a stale ask is mostly a record of a price that has just moved.

**Caveats.** The size of the remaining IC depends on whether the adjustment is made across the whole sample (+0.002) or within each freshness cell (+0.008 to +0.017). The headline uses the whole-sample figure. Eight sessions.

**Run.** `python obi_bounce.py`
