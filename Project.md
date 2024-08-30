# How and When are High-Frequency Crypto Returns Predictable?

[TOC]

## Introduction

This is a toy project based on and containing some code implementations of parts from the  work by Aït-Sahalia et al. (2022)[^1].

The main purpose of this project is to study the predictability of ultra high-frequency crypto returns (in an ideal offline research environment). This project is not intended to be a practical trading strategy, but rather a simple demonstration for research purpose. It is also a practice of processing the high-frequency order book and trades data of cryptocurrencies.

We also discussed the possible assumptions and limitations, as well as why a real world agent can barely benefit from such kinds of predictions and then monetize such research works. The demonstration may give out some simple insights and ideas for further research.

## Data

For simplicity and reproducibility here we use the [Binance Public Market Data](https://data.binance.vision/). Other options including [tardis](https://tardis.dev) and self-collected data. Data organizing and preprocessing are done in `./data/data_process.ipynb`.

### Ticker

The processed `ticker` dataframe contains the following attributes:

- `ts`: the `event_time` of the ticker
- `bp`: the best bid price
- `bv`: the best bid volume
- `ap`: the best ask price
- `av`: the best ask volume
- `mp`: the mid price calculated as `(bp + ap)/2`

The `ticker` dataframe is aggregated by `ts`, and we only keep the last record for multiple records with duplicated `ts`.

The `ticker` dataframe is then converted to `numpy.rec.recarray` for better performance in the feature calculation part.
### Trades

The processed `trades` dataframe contains the following attributes:

- `ts`: timestamp of the exchange (`transact_time`)
- `p`: the price of the trade
- `v`: the volume of the trade. If the trade is a buy, the volume is positive; if the trade is a sell, the volume is negative.
- `mp`: the latest mid price right before the current `trades:ts`, included for price impact calculation

The `trades` dataframe actually contains the `aggTrade` data from Binance (see [API Docs](https://binance-docs.github.io/apidocs/spot/en/#compressed-aggregate-trades-list)). 

> aggTrades: Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity aggregated.

The `is_buyer_maker` attribute of the original data is used to cast the volume to positive or negative.

The `trades` dataframe is then converted to `numpy.rec.recarray` for better performance.

## Feature, Target and Model

The implementations of feature operators are in the `./variables/features.py` file. The features are calculated based on the `ticker` and `trades` data. Each feature calculation function takes slices of the `ticker` and `trades` data as input and returns a single float value as the feature value.

We have implamented nearly all the features(predictors) mentioned in the original work[^1], except fot the *Turnover*. The *AutoCov* feature is modified since we donot aggregate the `trades` to unique timestamps.

For *time clocks*, both the *natural calendar clock* and the *trading(transaction) clock* are implemented, ***but not the volume clock***. See section 2 of [^1].

After finishing feature calculations, one may utilize those kaggle tricks for model fine-tuning, which is beyond the scope of this project.

Note: codes are not carefully written and checked. You may use other HPC platforms such as polars, hadoop/spark or rapids.

## Discussion: Assumptions and Limitations

This part mainly discussed why the prediction results from the research environment can barely monetize and be realized in real world. 

### Data and Latency

We use the `event_time` and assume that agents immediately receive the events from the server (with zero latency). Work by Aït-Sahalia et al. may also be using such kind of timestamps as they stated

> (Each transaction record) contains a timestamp of the transaction and its associated price, size and trading direction

Moreover, they have analyzed how latency affects the predictability in the section 6 of their work[^1]

Most of the time network latency is of a much larger order of magnitude than internal(local) processing latency.

### Feature

Features are not ’strong’ enough, or at least failed to capture or describe some crucial dynamics/properties of the market.

Another issue is that different order book events (trade, limit order placement and cancellation) may arrive with non-stationary and inconsistent intensity over time, containing and reflecting different levels of information. One needs to deal with the calendar clock and event clock properly.

### Target

We focus on the return prediction task here. One main issue is that the mid price is not a good estimator for ‘fair price’ given the public information. Neither the mean price of transactions.

Given arrays of `trades, ticker`, if use the terminal mid price as the fair price:

```python
n = len(ticker)
weight = np.zeros(n)
weight[-1] = 1
fair_px = weight @ ticker['mid_price']
```

Similarly, if using the close price(last price):

```python
n = len(trades)
weight = np.zeros(n)
weight[-1] = 1
fair_px = weight @ trades['price']
```

and if use the mean traded price:

```python
n = len(trades)
weight = np.ones(n) / n
fair_px = weight @ trades['price']
```

In general, one may consider a general function of fair price as

```
fair_px = function(trades, ticker, bookDepth, ...)
```

Market participants with different assumptions, information and utilities may come up with different *local fair price estimators*. See the micro-price by Stoikov[^2] for examples and a discussion of properties.

Another issue is that the mid price (or some other fair prices) is non-tradable. One **cannot trade at the mid price** even if there are counterparties with a willingness to deal. 

### Model

Targets, labels and loss functions all matter.

### Execution

Cornerstone.

## Conclusion

This project contains some code implementations of feature and target calculations of the work by Aït-Sahalia et al. (2022)[^1]. We further briefly discussed the possible assumptions and limitations, as well as why a real world agent can barely benefit from such kinds of predictions and then monetize. 

A useful and intuitive conclusion from Aït-Sahalia et al. (2022)[^1] is that, the short-term return predictability quickly vanishes within milliseconds. This short-term “predictability” exists just because it is difficult to realize and monetize, as the competitive arbitragers will try to profit from the short-term mis-pricing. This phenomenon can also be viewed as a realization of market effeciency, to some extent.

## Reference

[^1]: Aït-Sahalia, Y., Fan, J., Xue, L., & Zhou, Y. (2022). *How and When are High-Frequency Stock Returns Predictable?* (No. w30366). National Bureau of Economic Research. https://www.nber.org/system/files/working_papers/w30366/w30366.pdf

[^2]: Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance*, *18*(12), 1959-1966. https://doi.org/10.1080/14697688.2018.1489139

