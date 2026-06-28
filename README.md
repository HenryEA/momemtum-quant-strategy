# Adaptive Stock Momentum Quantitative Trading Strategy

This repository contains the code for a stock-only quantitative trading research system designed to backtest an adaptive momentum strategy on historical U.S. equity market data. The system combines multi-factor signal generation, risk-managed portfolio construction, volatility targeting, transaction-cost modeling, benchmark comparison and out-of-sample validation.
For a more extensive explanation of the signal logic, portfolio construction and validation framework, see the [full documentation](docs/documentation.md).

## Project Overview

The goal of this project is to construct and evaluate a systematic stock momentum portfolio. The strategy does not use live trading or broker execution. It is built strictly for research, backtesting and performance analysis.

The system ranks stocks using a composite momentum signal, converts those signals into portfolio weights, applies risk controls and then simulates historical performance. The main idea is that stocks with strong, smooth and persistent price momentum may continue to outperform over medium-term horizons, especially when combined with proper risk management and portfolio diversification.

The strategy is best described as a **risk-managed adaptive equity momentum allocation system**. It is not a pure market-neutral strategy, but a total-return stock allocation strategy that combines broad equity participation with momentum and quality tilts.

## Key Components

### 1. Data Preparation

Historical stock data is downloaded using `yfinance`. The system retrieves daily open, high, low, close, adjusted close and volume data for a broad U.S. stock universe. The benchmark asset, usually `SPY`, is also downloaded and used for market comparison, beta estimation and market-regime filtering.

The tradable universe is defined in `configs/instruments.yml`, which contains stock tickers, company names, sectors and metadata. The current system uses an expanded universe of 111 tradable symbols.

Adjusted close prices are used where available to account for dividends, splits and other corporate actions. The data loader aligns all stock price series by date so that signals and returns are calculated consistently across the universe.

A diagnostic script is also included to check data quality before running the backtest. It reports missing data, date coverage, flat prices, volume issues and other potential problems.

### 2. Signal Generation

The strategy uses a composite alpha signal instead of relying on one indicator. Each stock receives a daily score based on several momentum, trend and quality features.

The main signal components include:

- **12-1 Momentum:** Measures performance over the past 12 months while skipping the most recent month.
- **Intermediate Momentum:** Captures earlier momentum over the medium-term lookback window.
- **9-1 and 6-1 Momentum:** Measures shorter medium-term price strength.
- **Residual Momentum:** Measures stock-specific strength after removing market-related movement.
- **Relative Strength vs SPY:** Rewards stocks outperforming the broad market.
- **Volatility-Adjusted Momentum:** Favors stocks with strong returns and lower volatility.
- **Information Ratio Signal:** Measures return efficiency relative to risk.
- **Trend Strength:** Uses moving averages such as 50-day and 200-day averages.
- **52-Week-High Features:** Rewards stocks trading near persistent yearly highs.
- **Efficiency Ratio:** Favors smoother and less chaotic price trends.
- **Frog-in-the-Pan Momentum:** Rewards gradual and persistent momentum.
- **Downside and Drawdown Quality:** Penalizes stocks with large downside moves.
- **Short-Term Reversal Control:** Reduces the risk of buying after excessive short-term moves.

Each signal component is normalized cross-sectionally, meaning every stock is compared against the other stocks in the universe on the same date. Extreme values are clipped to prevent one abnormal reading from dominating the portfolio.

### 3. Portfolio Construction

The portfolio construction engine converts stock signals into portfolio weights. The main portfolio mode is:

```text
adaptive_total_return_v5
```

This mode combines several portfolio sleeves:

- **Momentum Core Sleeve:** Allocates to the strongest-ranked stocks.
- **Diversified Positive-Signal Sleeve:** Allocates across a broader group of stocks with positive signals.
- **Broad Universe Sleeve:** Maintains diversified stock-market participation.
- **Satellite Overlay Sleeve:** Adds a small residual long-short component without dominating the portfolio.

This structure allows the strategy to participate in broad equity market strength while still tilting toward stocks with stronger momentum and quality characteristics.

### 4. Risk Management

Risk management is central to the strategy. The system includes:

- **Volatility Targeting:** Scales exposure toward a target annualized volatility.
- **Gross Exposure Limit:** Prevents excessive leverage.
- **Net Exposure Limit:** Controls overall long market exposure.
- **Maximum Stock Weight:** Prevents overconcentration in one stock.
- **Sector Caps:** Limits excessive exposure to one sector.
- **Market Regime Filter:** Uses SPY to reduce exposure during weak or high-volatility markets.
- **Drawdown Brake:** Reduces risk when the strategy experiences large losses.
- **Turnover Aversion:** Smooths portfolio changes to reduce unnecessary trading.

### 5. Transaction Cost Modeling

The backtest includes transaction costs to make the simulation more realistic. The cost model includes commission, bid-ask spread, slippage and market-impact assumptions.

Costs are deducted from gross returns to calculate net strategy performance.

### 6. Backtesting Framework

The backtest engine simulates the full strategy through time. It performs the following steps:

1. Load stock universe and benchmark.
2. Load historical OHLCV data.
3. Calculate daily stock returns.
4. Generate daily alpha signals.
5. Convert signals into target portfolio weights.
6. Apply portfolio constraints and risk controls.
7. Shift weights to avoid lookahead bias.
8. Calculate portfolio returns.
9. Subtract transaction costs.
10. Generate performance reports.

The system avoids lookahead bias by ensuring that today’s return is calculated using portfolio weights that would have been known before today’s return occurred.

### 7. Validation Framework

The project includes multiple validation tools:

- **Strategy Suite:** Compares different strategy configurations.
- **Out-of-Sample Selection:** Splits data into train, validation and test periods.
- **Walk-Forward Validation:** Tests performance across rolling time windows.
- **Cost Stress Testing:** Checks whether the strategy survives higher trading costs.
- **Parameter Sensitivity Testing:** Tests whether performance is stable across parameter changes.

These tools help reduce the risk of overfitting and provide a more realistic view of strategy robustness.

## Technical Stack

**Programming Language:** Python

**Main Libraries:**

```text
Pandas
NumPy
PyYAML
yfinance
Matplotlib
```

**Core Techniques:**

```text
Quantitative Momentum Investing
Cross-Sectional Ranking
Residual Momentum
Relative Strength
Volatility Targeting
Risk-Based Position Sizing
Transaction Cost Modeling
Portfolio Rebalancing
Benchmark Comparison
Out-of-Sample Validation
Walk-Forward Testing
```

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
Average Turnover: 2.69%
Average Gross Exposure: 117.70%
Maximum Gross Exposure: 132.00%
Benchmark Beta: 0.83
Annualized Alpha vs Benchmark: 6.03%
Tradable Symbols: 111
Average Active Names: 108.54
```

### Benchmark Comparison

```text
Strategy CAGR: 17.80%
Benchmark CAGR: 13.55%

Strategy Sharpe Ratio: 1.07
Benchmark Sharpe Ratio: 0.82

Strategy Maximum Drawdown: -26.75%
Benchmark Maximum Drawdown: -33.72%
```

The strategy outperformed the benchmark in annualized return, Sharpe ratio and maximum drawdown.

### Equal-Weight Universe Comparison

```text
Strategy CAGR: 17.80%
Equal-Weight CAGR: 17.86%

Strategy Sharpe Ratio: 1.07
Equal-Weight Sharpe Ratio: 1.04

Strategy Volatility: 16.69%
Equal-Weight Volatility: 17.31%
```

The strategy performed similarly to the equal-weight universe in terms of CAGR, but achieved a slightly higher Sharpe ratio with lower volatility. This suggests that the system added value mainly through risk management, volatility control and momentum-based tilting.

### Out-of-Sample Results

The main default configuration achieved:

```text
Test Sharpe Ratio: 1.06
Test CAGR: 16.66%
Test Maximum Drawdown: -16.58%
Full-Period Sharpe Ratio: 1.07
Full-Period CAGR: 17.80%
Full-Period Maximum Drawdown: -26.75%
```

The out-of-sample result suggests that the strategy maintained reasonable performance in a later unseen test period.

## Interpretation

The results demonstrate the full quantitative research pipeline:

- Data acquisition and preparation
- Multi-factor signal generation
- Momentum-based ranking
- Risk-managed portfolio construction
- Volatility targeting
- Transaction-cost modeling
- Benchmark comparison
- Strategy comparison
- Out-of-sample validation
- Cost and robustness testing

## Conclusion

This project implements a complete stock-only quantitative momentum research system. It starts from historical stock data, builds multiple momentum and quality signals, creates a composite alpha score, converts the score into a risk-managed portfolio and evaluates performance through a full backtesting and validation framework.

The strategy achieved strong historical performance relative to the benchmark, with higher CAGR, higher Sharpe ratio and lower maximum drawdown. It also performed competitively against the equal-weight stock universe.
