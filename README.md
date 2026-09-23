# FX market microstructure research

Harry Larkin · harry@larkin.me.uk

Research on the economics of liquidity provision in FX and crypto, using EBS and Cboe FX tick and order-book data and public crypto exchange data. Each folder is one study, with its code and a README covering the question, data, method, results and caveats.

| # | Study | Result |
|---|---|---|
| 01 | [Spreads and volatility](01_spreads_and_volatility) | The half-spread scales one-for-one with volatility across 17 FX pairs (elasticity 0.95, R² 0.94). Within a pair, the implied holding horizon lengthens as trading thins, at about two-thirds the rate a volatility-per-trade model predicts. |
| 02 | [Order-book imbalance](02_order_book_imbalance) | About 86% of the apparent predictive power of top-of-book imbalance on Cboe FX is the previous return reversing. |
| 03 | [Inferring fills from the order book](03_inferring_fills) | On a venue without trade messages, treating a disappearing price level as a fill overstates maker spread capture by 3.7× on EUR/USD and reverses its sign on USD/JPY. |
| 04 | [Parent-order reconstruction](04_parent_orders) | Grouping anonymous EBS prints into the orders behind them, the maker's markout rises with a print's position in its run: isolated prints are the most toxic, the later prints of long runs the least. |
| 05 | [Limits of taking](05_limits_of_taking) | At horizons up to 50 events, a perfect direction call loses money after realised costs at every spread cap tested. |
| 06 | [Crypto](06_crypto) | Apparent cross-venue profit in BTC/ETH perpetual market making is a basis position booked as profit; FX dollar order flow predicts BTC and ETH at 10 seconds; part of the Alpha191 factor library survives out of sample on the crypto cross-section, with a short-beta tilt that the headline Sharpe flatters. |

## Publication

Larkin, H. & Larkin, A. (2026). *When, Not Which Way: Move-Occurrence Predictability, Directional Efficiency, and Regime-Adaptive Clustering in Foreign-Exchange Tick Data.* SSRN 6919443. <https://ssrn.com/abstract=6919443>

## Data

Market data is not included. EBS and Cboe FX data are licensed from the exchanges and cannot be redistributed, and the full set runs to several gigabytes. The scripts read their inputs from environment variables:

| Variable | Contents | Used by |
|---|---|---|
| `EBS_L1_DIR` | EBS Level-1 vendor tape: one gzipped CSV per day, columns `date,time,pair,P/D,bid,ask,bidvol,askvol`, 1-second stamps | 01, 04 |
| `EBS_RECORDINGS_DIR` | EBS top-of-book and Time & Sales recordings (`ticks-*.csv`, `deals-*.csv`) | 03 |
| `CBOE_FEATURES_DIR` | Per-pair, per-day parquet files built from the Cboe FX order-by-order feed | 02, 03 |
| `CBOE_L1_DIR` | Cboe FX top-of-book parquet | 05, 06 |
| `CRYPTO_RAW_DIR` | Binance public data from data.binance.vision (`06_crypto/perps_maker_hedge/fetch.py` downloads it) | 06 |
| `OUT_DIR` | Where results are written; defaults to `out/` inside each study folder | all |

The Alpha191 study downloads its own daily data.

## Setup

```
pip install -r requirements.txt
```

## Method

Most of the work in these studies goes into checking whether a result is real. The checks that recur are placebos (random pairing, shuffled signals, feeds made deliberately stale), removing the previous return before measuring prediction, held-out or walk-forward samples, and validating a proxy against ground truth where it exists. Several of the results here came out of those checks rather than out of the original hypothesis.
