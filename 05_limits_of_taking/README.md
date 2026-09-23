# 05 · Limits of taking

**Question.** How good does a short-horizon direction forecast need to be before crossing the spread pays?

**Data.** Cboe FX top-of-book, seven major pairs, seven sessions in July 2026.

**Method.** A walk-forward ridge ensemble of single-pair book features, plus two cross-pair features: a fair-value residual across the whole currency graph (`trirv.py`) and dollar order flow across the majors. Execution is priced conservatively: the decision uses the book at time t and the fill uses the book at t + 1 ms. For each horizon and spread cap the script compares the information coefficient achieved with the one needed to cover the realised round-trip cost, and computes what a perfect direction call would earn.

**Results.** Achieved IC: 0.173 at 5 events, 0.183 at 20, 0.131 at 50, 0.096 at 100 and 0.057 at 300.

Trading only when the spread is tight halves the realised cost at 20 events (0.648 → 0.288 pip), but the tight spread at entry does not last until the exit.

Perfect foresight at the 0.15-pip spread cap, in pips:

| Horizon (events) | 5 | 20 | 50 | 100 | 300 |
|---|---|---|---|---|---|
| Expected size of move | 0.133 | 0.183 | 0.265 | 0.359 | 0.599 |
| Realised round-trip cost | 0.256 | 0.288 | 0.309 | 0.322 | 0.348 |
| Net for a perfect call | −0.122 | −0.105 | −0.044 | +0.037 | +0.251 |

Up to 50 events a perfect call loses money, so the IC required to profit exceeds 1 and no forecast can close the gap. `longhorizon.py` looks at the longer horizons where the bound does not rule taking out.

**Caveats.** Seven sessions. The perfect-foresight figures assume normally distributed moves.

**Run.** `python takerens.py`, `python longhorizon.py`
