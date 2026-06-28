# Adaptive Stock Momentum Quantitative Trading Strategy

This repository contains the code for a quantitative trading research system designed to backtest a stock-only momentum strategy using historical U.S. equity market data. The system uses a multi-factor momentum signal, adaptive portfolio construction, volatility targeting, transaction-cost modeling and out-of-sample validation to evaluate whether a rule-based stock allocation strategy can outperform a market benchmark.

## Project Overview

The goal of this system is to construct and evaluate a systematic stock momentum portfolio by learning from historical price behavior. It does not use machine learning in the form of neural networks, but it uses quantitative rules, statistical indicators and portfolio optimization techniques to rank stocks, allocate capital and manage risk.

The strategy is built around the idea that stocks with strong and persistent price momentum may continue to outperform over medium-term horizons. Instead of relying on one simple momentum measure, the system combines several signals such as 12-month momentum, intermediate momentum, residual momentum, relative strength, trend quality, 52-week-high behavior, downside risk and volatility-adjusted performance.

The system is designed strictly for research and backtesting. It does not perform live trading and it does not connect to a broker for order execution.

## Key Components

### 1. Data Preparation

**Data Fetching:**
Historical stock data is downloaded using `yfinance`. The system retrieves daily open, high, low, close, adjusted close and volume data for a broad stock universe. The benchmark asset, usually `SPY`, is also downloaded and used for market comparison, beta estimation and regime filtering.

**Stock Universe Definition:**
The tradable stock universe is defined in `configs/instruments.yml`. This file contains the list of stock tickers, company names, sectors and other metadata used by the system. The current system uses an expanded stock universe of 111 tradable symbols.

**Adjusted Price Handling:**
The strategy uses adjusted close prices where available. This is important because stock prices are affected by dividends, splits and corporate actions. Adjusted prices make the historical return series more realistic for long-term research.

**Data Alignment:**
All stock price series are aligned by date. This ensures that the strategy compares stocks on the same trading days. Missing symbols can be skipped using the `--allow-missing` option.

**Data Diagnostics:**
A diagnostic script is included to check the quality of the downloaded data. It reports missing data, start dates, end dates, flat prices, volume issues and other possible data problems before the backtest is trusted.

### 2. Signal Generation

The main signal generation logic is implemented inside the strategy module. The system calculates a composite momentum score for each stock on each trading day.

The signal is not based on one indicator only. Instead, it combines multiple components to create a more stable and diversified ranking model.

The major signal components include:

**12-1 Momentum:**
Measures stock performance over approximately the past 12 months while skipping the most recent month. This is used to capture medium-term momentum while avoiding very short-term reversal noise.

**Intermediate Momentum:**
Measures earlier momentum over the intermediate part of the lookback window. This helps identify stocks that have had persistent strength over a longer horizon.

**9-1 and 6-1 Momentum:**
These are shorter momentum measures that capture recent but not extremely short-term price strength.

**Residual Momentum:**
Measures stock-specific momentum after removing the portion of return explained by the market benchmark. This helps separate true stock-level strength from general market movement.

**Relative Strength vs Benchmark:**
Compares each stock’s performance against the benchmark, usually `SPY`. A stock receives a better score if it outperforms the broad market.

**Volatility-Adjusted Momentum:**
Rewards stocks that achieve strong returns with lower volatility. This helps the system avoid stocks that only look strong because they are extremely unstable.

**Information Ratio Signal:**
Measures the efficiency of a stock’s return relative to its risk. It helps identify stocks with better risk-adjusted momentum.

**Trend Strength:**
Uses moving averages, such as the 50-day and 200-day moving averages, to identify whether a stock is in a healthy upward trend.

**52-Week-High and Breakout Features:**
Detects whether a stock is close to its yearly high. Stocks near their 52-week highs can often show persistent momentum.

**New-High Persistence:**
Checks whether a stock repeatedly stays near its highs instead of only briefly touching them.

**Efficiency Ratio:**
Measures whether a stock’s price movement is smooth or chaotic. The strategy prefers smoother momentum paths.

**Frog-in-the-Pan Momentum:**
Captures gradual and persistent momentum. This rewards stocks that rise steadily instead of stocks that move sharply in one sudden jump.

**Volatility Contraction:**
Checks whether recent volatility is lower than longer-term volatility. This can indicate a calmer and more controlled trend.

**Drawdown Quality:**
Penalizes stocks that have fallen too sharply from recent highs.

**Downside Quality:**
Focuses on negative-return volatility. Stocks with large downside moves receive lower scores.

**Return Consistency:**
Rewards stocks that produce more stable positive returns rather than irregular and noisy price movements.

**Short-Term Reversal Control:**
A small reversal feature is included to reduce the risk of buying stocks immediately after an excessive short-term move.

### 3. Signal Normalization

Since each signal component is measured differently, the system standardizes them before combining them.

The strategy uses cross-sectional normalization, which means each stock is compared against the other stocks in the universe on the same date.

For example, if `NVDA` has much stronger 12-month momentum than most other stocks, it receives a high normalized score. If another stock has weaker momentum than the universe average, it receives a lower score.

Extreme values are clipped to prevent one abnormal signal from dominating the final portfolio.

### 4. Composite Alpha Signal

After calculating and normalizing all signal components, the system combines them into one final alpha score.

The final signal represents the strategy’s view of each stock:

- A high positive signal means the stock is attractive.
- A low or negative signal means the stock is weak or unattractive.
- A near-zero signal means the stock is neutral.

The strategy then uses this final score to decide which stocks should receive higher or lower weights in the portfolio.

### 5. Portfolio Construction

The portfolio construction process converts stock signals into actual portfolio weights.

The main portfolio mode is:

```text
adaptive_total_return_v5
```

This portfolio mode is designed as a total-return stock allocation strategy. It combines broad market participation with momentum and quality tilts.

The portfolio is divided into several sleeves:

**Momentum Core Sleeve:**
This sleeve allocates capital to the strongest-ranked stocks based on the composite signal. It represents the high-conviction part of the portfolio.

**Diversified Positive-Signal Sleeve:**
This sleeve allocates across a broader set of stocks with positive signals. It reduces the risk of relying too heavily on only a few top-ranked names.

**Broad Universe Sleeve:**
This sleeve gives the strategy broad participation across the stock universe. It is included because broad diversified exposure can be difficult to beat during strong equity market periods.

**Satellite Overlay Sleeve:**
This is a small residual long-short overlay. It attempts to add a small amount of stock-selection alpha without dominating the overall portfolio.

The final portfolio is therefore not a pure market-neutral alpha strategy. It is a risk-managed equity momentum allocation system.

### 6. Risk Management

Risk management is a major part of the system. The strategy is not only trying to maximize return; it is also trying to control volatility, drawdown, concentration and market exposure.

The main risk controls include:

**Volatility Targeting:**
The system scales portfolio exposure so that expected annualized volatility stays close to the configured volatility target.

**Gross Exposure Limit:**
Controls the total absolute exposure of the portfolio. This prevents the strategy from using excessive leverage.

**Net Exposure Limit:**
Controls the overall long or short direction of the portfolio.

**Maximum Stock Weight:**
Limits how much capital can be placed in a single stock.

**Sector Cap:**
Prevents the portfolio from becoming overly concentrated in one sector, such as Technology.

**Market Regime Filter:**
Uses the benchmark, usually `SPY`, to detect whether the market is in a healthy or risky environment. If the market is weak or volatility is high, the system can reduce exposure.

**Drawdown Brake:**
Reduces exposure when the strategy itself experiences a large drawdown. This helps prevent the portfolio from continuing at full risk during difficult periods.

**Turnover Aversion:**
Reduces unnecessary trading by smoothing portfolio changes over time.

### 7. Transaction Cost Modeling

The system includes a transaction-cost model to make the backtest more realistic.

Costs include:

**Commission Cost:**
A small cost charged when the portfolio trades.

**Spread Cost:**
Represents the difference between the bid and ask price.

**Slippage Cost:**
Represents the possibility of receiving a worse execution price than expected.

**Market Impact Cost:**
Represents the cost of trading larger positions.

Transaction costs are subtracted from gross returns to calculate net strategy performance.

### 8. Backtesting Framework

The backtest engine simulates the strategy through time using historical data.

The engine follows this process:

1. Load the stock universe and benchmark.
2. Load historical OHLCV data.
3. Calculate stock returns.
4. Generate daily alpha signals.
5. Convert signals into target portfolio weights.
6. Apply risk controls and exposure limits.
7. Shift weights to avoid lookahead bias.
8. Calculate portfolio returns.
9. Subtract transaction costs.
10. Generate performance reports.

The system avoids lookahead bias by ensuring that today’s portfolio return is based on weights that would have been known before today’s return occurred.

### 9. Strategy Evaluation

The system produces several performance reports after each backtest.

The main output files include:

```text
metrics.json
returns.csv
equity_curve.csv
weights.csv
signals.csv
costs_turnover.csv
monthly_returns.csv
tearsheet.png
```

**metrics.json:**
Contains the main performance statistics.

**returns.csv:**
Contains daily gross returns, net returns, costs and benchmark returns.

**equity_curve.csv:**
Shows how the strategy capital grows over time.

**weights.csv:**
Shows the daily portfolio weight assigned to each stock.

**signals.csv:**
Shows the daily signal score for each stock.

**costs_turnover.csv:**
Shows daily trading turnover and estimated transaction costs.

**monthly_returns.csv:**
Shows monthly strategy returns.

**tearsheet.png:**
Provides a visual summary of equity curve, drawdown, exposure and turnover.

### 10. Validation Framework

The project includes additional scripts to test whether the strategy is robust.

**Strategy Suite:**
Runs multiple strategy configurations and compares their performance.

**Out-of-Sample Configuration Selection:**
Splits the data into training, validation and testing periods. This helps evaluate whether a strategy configuration performs well on unseen data.

**Walk-Forward Validation:**
Tests the strategy over multiple time windows to check whether performance is stable across different market periods.

**Cost Stress Testing:**
Tests whether the strategy still performs reasonably when transaction costs are increased.

**Parameter Sensitivity Testing:**
Checks whether performance depends too heavily on one exact parameter setting.

These validation tools are included to reduce the risk of overfitting and to provide a more realistic assessment of the strategy.

## Technical Stack

Programming Language:

```text
Python
```

Main Libraries:

```text
Pandas
NumPy
PyYAML
yfinance
Matplotlib
```

Core Techniques:

````text
Quantitative Momentum Investing
Cross-Sectional Ranking
Residual Momentum
Relative Strength
Volatility Targeting
Risk-Based Position Sizing
Transaction Cost Modeling
Portfolio Rebalancing
Out-of-Sample Validation
Walk-Forward Testing
Benchmark Attribution


## Results

The strategy was tested on historical daily stock data from 2014 to 2025.

### Main Backtest Results

```text
Observations: 3017
Total Return: 6.110894
CAGR: 17.80%
Annualized Volatility: 16.69%
Sharpe Ratio: 1.07
Sortino Ratio: 1.36
Maximum Drawdown: -26.75%
Calmar Ratio: 0.67
Hit Rate: 55.19%
Best Day: 4.56%
Worst Day: -6.25%
Average Daily Cost: 0.000008
Annualized Cost Drag: 0.21%
Average Turnover: 2.69%
Average Gross Exposure: 117.70%
Maximum Gross Exposure: 132.00%
Average Net Exposure: 117.70%
Benchmark Beta: 0.83
Annualized Alpha vs Benchmark: 6.03%
Benchmark Correlation: 0.86
Tradable Symbols: 111
Average Active Names: 108.54
````

### Benchmark Comparison

```text
Strategy CAGR: 17.80%
Benchmark CAGR: 13.55%

Strategy Sharpe Ratio: 1.07
Benchmark Sharpe Ratio: 0.82

Strategy Maximum Drawdown: -26.75%
Benchmark Maximum Drawdown: -33.72%
```

The strategy outperformed the benchmark in terms of annualized return, Sharpe ratio and maximum drawdown. The result suggests that the system was able to improve risk-adjusted performance relative to the benchmark while maintaining strong broad equity participation.

### Equal-Weight Universe Comparison

```text
Strategy CAGR: 17.80%
Equal-Weight CAGR: 17.86%

Strategy Sharpe Ratio: 1.07
Equal-Weight Sharpe Ratio: 1.04

Strategy Volatility: 16.69%
Equal-Weight Volatility: 17.31%
```

The strategy performed similarly to the equal-weight stock universe in terms of CAGR, but achieved a slightly higher Sharpe ratio with slightly lower volatility. This suggests that the strategy added value mainly through risk management, volatility control and momentum-based tilting rather than by dramatically outperforming a diversified stock basket.

## Strategy Suite Results

Several strategy configurations were tested. The main configurations include:

```text
v5_default_adaptive_total_return
v5_balanced
v5_growth
v5_sharpe
v5_equal_weight_plus
long_only
market_neutral
```

The strongest full-period results were produced by the adaptive total-return configurations. The market-neutral configuration produced weak results, which indicates that the pure long-short alpha signal was not strong enough on its own in this universe.

This means the best interpretation of the system is:

```text
A risk-managed equity momentum allocation strategy
```

rather than:

```text
A pure market-neutral alpha strategy
```

## Out-of-Sample Results

The out-of-sample configuration selection process showed that the leading strategy configurations maintained positive performance in the testing period.

The main default configuration achieved:

```text
Test Sharpe Ratio: 1.06
Test CAGR: 16.66%
Test Maximum Drawdown: -16.58%
Full-Period Sharpe Ratio: 1.07
Full-Period CAGR: 17.80%
Full-Period Maximum Drawdown: -26.75%
```

The out-of-sample result is important because it suggests that the strategy did not only perform well over the full historical period, but also maintained reasonable performance in a later unseen test period.

## Interpretation of Results

The results from this project above demonstrates:

- Data acquisition and preparation
- Multi-factor stock signal generation
- Momentum-based ranking
- Risk-managed portfolio construction
- Volatility targeting
- Transaction-cost modeling
- Benchmark comparison
- Strategy comparison
- Out-of-sample validation
- Cost and robustness testing

The strategy produced better risk-adjusted results than the benchmark and slightly better Sharpe ratio than the equal-weight stock universe. However, the system should not be interpreted as a fully proven production trading strategy.

The performance is strongest when the strategy is allowed to maintain broad equity exposure. The market-neutral version was weak, which means the strategy’s main strength is total-return equity allocation rather than pure long-short alpha generation.

## Conclusion

This project implements a complete stock-only quantitative momentum research system. It starts from raw historical stock data, calculates multiple momentum and quality features, builds a composite alpha signal, converts the signal into a risk-managed portfolio and evaluates the strategy through a full backtesting and validation pipeline.

The strategy achieved strong historical performance relative to the benchmark, with a higher CAGR, higher Sharpe ratio and lower maximum drawdown than the benchmark. It also performed competitively against an equal-weight stock universe.

The system is best described as a risk-managed adaptive equity momentum allocation strategy. It is suitable as a quantitative research project and demonstrates important concepts used in systematic trading, including signal generation, portfolio construction, volatility targeting, cost modeling, benchmark comparison and out-of-sample validation.
