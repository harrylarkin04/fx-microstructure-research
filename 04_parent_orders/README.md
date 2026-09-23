# 04 · Parent-order reconstruction

**Question.** EBS publishes individual trade prints but not the orders behind them. Large orders are usually worked as a run of same-side prints. Does the adverse selection a passive maker suffers depend on where in that run it was filled?

**Data.** EBS Level-1 vendor tape, 1 March to 31 May 2026. Six pairs, 403,132 prints after filtering.

**Method.**

- **Direction.** In this tape, inferring the aggressor side from which size field is filled is wrong for about a fifth of rows: some rows combine two prints, and some prices match neither side of the quote. Kept: rows with one side filled and a price equal to that side of the quote prevailing in the previous second (64% of rows).
- **Parents.** Same-side prints less than 5 seconds apart are treated as one parent order. A print's child number is its position in that run; 1 is the first or only print.
- **Markout.** The passive side's gain over 5 and 30 seconds, in basis points (multiply by 100 for USD per million).

**Results.** Maker markout at 5 seconds by child number:

| Child number | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21+ |
|---|---|---|---|---|---|---|---|
| All prints (bp) | +0.013 | +0.029 | +0.040 | +0.053 | +0.085 | +0.081 | +0.184 |
| 1M prints only (bp) | +0.029 | +0.064 | +0.087 | +0.109 | +0.148 | +0.176 | +0.220 |
| Prints | 299,276 | 57,433 | 20,319 | 14,693 | 8,294 | 2,444 | 673 |

Isolated prints are the most toxic fills and the later prints of a long run the least toxic. The pattern holds at a fixed print size, and in five of six pairs (rank correlation +0.89 EUR/USD, +0.96 USD/CNH, +0.96 USD/CHF, +0.40 EUR/CHF, +0.30 AUD/USD, −0.04 USD/JPY). By 30 seconds it has largely faded, except in the longest runs.

A likely reading is that a long run of same-side prints is someone working a large order with little sensitivity to price, while a single print is more often informed.

**Caveats.**

- The result depends on classifying direction correctly. Using the quote from the same second instead of the previous one reverses the sign, because a print that consumed a level is then compared with the book after it traded. The previous-second quote is the right reference, and it keeps the share of rows the documented correction predicts.
- The tape is stamped to the second and prints within a second are netted, so child numbers count seconds with a print rather than individual prints.
- Markouts are before fees and before the cost of exiting the position.

**Run.** `python parent_orders.py`
