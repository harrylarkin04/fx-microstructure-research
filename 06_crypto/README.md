# 06 · Crypto

Three studies.

## Perpetual market making across venues — `perps_maker_hedge/`

**Question.** Does it pay to make on one BTC or ETH perpetual book and hedge on another?

**Data.** Binance USD-margined and coin-margined perpetuals (BTCUSDT and BTCUSD_PERP, ETHUSDT and ETHUSD_PERP): top-of-book and aggregated trades from data.binance.vision, six days spread across market regimes between September 2023 and March 2024. The trade tape records the aggressor side, so maker fills are observed rather than assumed.

**Results.** The headline profit is a basis position booked as profit. Making on one book and hedging on the other leaves you long one and short the other, so booking the hedge price minus the fill price as profit banks an open position in the basis between them. That basis is large next to anything the quote earns and it swings with the funding cycle: across the twelve asset-days it runs from −9.7 to +9.6 bp, averaging +1.8 bp with a standard deviation of 6.0.

It shows up as two large numbers that nearly cancel. On 17 January 2024 in BTC, with the basis at −9.7 bp, passive buys book −9.44 bp and passive sells +9.89 bp; the reported profit is the residual, and its sign is set by which side of the book happened to fill more.

With the basis removed, the edge is +0.032 bp per fill, positive on 12 of 12 asset-days. Both books trade at a one-tick spread (half-spread 0.019 bp), so even the best fee tier, at 1.20 bp for the round trip, leaves −1.17 bp. In crypto perpetuals the fee schedule is the binding constraint.

**Run.** `fetch.py` downloads the data, `run.py` runs the backtest, `final.py` gives the basis-neutral result.

## FX dollar flow and crypto — `fx_flow_predicts_crypto/`

**Question.** Does order flow across the USD majors on Cboe FX predict BTC and ETH?

**Data.** Cboe FX order book and Binance perpetual trades, 20–29 July 2026.

**Method.** Three hypotheses, each with its sign fixed before testing: the dollar's price move (negative), dollar order-flow imbalance across the majors (negative, and the only one that needs order-book data), and risk appetite via AUD/JPY and NZD/JPY (positive). Non-overlapping samples.

**Results.** All three have the predicted sign at 10–60 seconds. At 10 seconds:

| Signal | BTC | ETH |
|---|---|---|
| Dollar order flow | −0.025 (t −7.2) | −0.020 (t −4.4) |
| Dollar price move | −0.021 (t −3.5) | −0.016 (t −2.1) |
| Risk appetite | +0.025 (t +3.6) | +0.016 (t +2.3) |

Order flow beats the price-only version of the same idea. A placebo using FX data from the wrong day gives +0.003. Nothing remains by 300 seconds, and the effect is far too small to pay a taker fee.

Shifting the FX series back by 10 seconds gives an IC of −0.18, but only because the windows then overlap: it measures co-movement, not prediction.

**Run.** `python fxcrypto.py`, `python fxcrypto_check.py`

## Alpha191 on the crypto cross-section — `alpha191_transplant/`

**Question.** Du, Walter and Ulrich (arXiv 2601.06499) find that part of the Alpha191 factor library, built for Chinese A-shares, survives in the S&P 500, and argue that it captures behavioural effects. If so, it should also work where retail trading is even more dominant. Does it survive in crypto?

**Data.** Daily prices and volumes for 45 liquid crypto assets (median 38 a day), 2018–2026, downloaded by `J1_crypto_universe.py`.

**Method.** The 69 Alpha191 signals that can be computed from daily data, tested with post-double-selection LASSO (Belloni, Chernozhukov and Hansen) against crypto controls: beta, size, momentum, short-term reversal, volatility, illiquidity and lottery demand. Selection uses 2018–2022 only; 2023–2026 is held out.

**Results.** 17 of 69 signals survive a Bonferroni threshold (|t| > 3.30) in the selection period. On the held-out period, a long-short portfolio with a 42-day hold and 15 bp costs has a Sharpe ratio of 0.97 (t 1.86) and a maximum drawdown of −23.8%. Shuffling the signals across assets gives a mean Sharpe of −0.85 and a best of 0.08. Eleven of the 17 are individually positive out of sample.

**Caveats.**

- A t-statistic of 1.86 over three years is suggestive rather than conclusive, and the universe has survivorship bias.
- The portfolio is not market-neutral as it stands: its beta on an equal-weighted crypto market is −0.17 (t −25) and its correlation with that market is −0.58, so part of the return is a short-beta tilt over a period when the market fell. Regressing it on the market and the market squared leaves an alpha with a t of 1.4.
- Five of the 17 survivors are volume-scale signals correlated 0.90 with each other. Dropping that cluster takes the held-out Sharpe from 0.97 to 0.40, so the result leans on one effect more than the count of survivors suggests.

**Run.** `J1` to `J5` in order. `J4` can also test diversification against an existing book if `BOOK_RETURNS_CSV` points to a CSV of its daily returns.
