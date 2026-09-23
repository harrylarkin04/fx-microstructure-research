# 01 · Spreads and volatility

**Question.** Does the FX half-spread scale with volatility? If it equals σ√T, then T = (half-spread / σ)² is the horizon over which a market maker is being compensated. Is T a fixed constant, or is it just the average time between trades, as a volatility-per-trade model (Wyart et al., 2008) would predict?

**Data.** EBS Level-1 vendor tape, 1 March to 31 May 2026. 17 currency pairs with at least 100 deals a day, 612 pair-days.

**Method.** For each pair and day:

- volatility from squared mid changes between consecutive quote updates, dropping gaps over 300 seconds so stale quotes on thin pairs don't depress it;
- the half-spread, weighted by how long each quote was live;
- the trade rate λ, in deals per active second;
- the implied horizon T = (half-spread / σ)².

The decisive test regresses log T on log(1/λ). A slope near 1 means T is the trade interval; a slope near 0 means T does not depend on how often the pair trades.

**Results.**

- log(half-spread) = 0.993 + 0.954 · log(σ), R² = 0.940, n = 612.
- The median implied horizon is 18.6 seconds (interquartile range 11–31 s). Across pair-days the 95th percentile is 27× the 5th, so it is not a universal constant.
- Pooled across pairs the slope is 0.13, which looks as if the horizon ignores the trade rate. With pair fixed effects the slope is **0.65 (s.e. 0.08)**: within a pair, the horizon lengthens as trading thins, at about two-thirds the rate the volatility-per-trade model predicts. The pooled figure reflects differences between pairs rather than the trade-rate effect.

**Caveats.** Three months, one venue.

**Run.** `python spreads_volatility.py`
