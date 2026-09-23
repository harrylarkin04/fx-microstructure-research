# 03 · Inferring fills from the order book

**Question.** Cboe FX's order-by-order feed publishes no trades. A common workaround is to treat a price level disappearing as a fill. Is that a valid way to measure what a market maker earns?

**Data.**

- Cboe FX order-book features, 14 pairs, 21–30 July 2026: 15.5 million level disappearances.
- EBS top-of-book and Time & Sales recordings over the same period, where real trade prints with the aggressor side exist alongside the quotes.

**Method.** On EBS, spread capture is measured two ways from the same tape: at real prints, and at level disappearances treated as fills. Capture is minus the markout divided by the half-spread: 1 means the maker keeps the whole half-spread, 0 means adverse selection takes all of it, and a negative value means the maker loses more than the spread. Every reference quote must be less than 200 ms old, and the quote used for the markout must not be stale.

In both scripts a negative markout means the market moved in the maker's favour.

**Results.** EUR/USD, spread capture:

| | 200 ms | 1 s | 5 s | 30 s |
|---|---|---|---|---|
| Real prints | 0.21 | 0.16 | 0.19 | 0.24 |
| Disappearances as fills | 0.64 | 0.58 | 0.60 | 0.60 |
| Prints / disappearances used | 6,465 / 88,404 | 4,770 / 60,021 | 6,046 / 81,370 | 6,341 / 87,612 |

The proxy overstates capture by 3.7× at 1 second. On USD/JPY it reverses the sign: real prints show the maker losing nearly a full half-spread at 1 second (−0.96), while the proxy shows it keeping about half (+0.48). Most disappearances are cancels, and a cancel carries no adverse move.

Applied to Cboe (`cboe_wipe_proxy.py`), the proxy implies makers capture about 110% of the half-spread on all 14 pairs. That implausible result is what prompted the check.

**Caveats.** Only EUR/USD and USD/JPY have enough recorded EBS prints, over eight sessions. Cancel behaviour on Cboe may differ from EBS.

**Run.** `python cboe_wipe_proxy.py`, then `python validate_against_ebs_prints.py`
