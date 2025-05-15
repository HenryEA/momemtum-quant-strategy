## Quantitative Momentum Trading Strategy for FX Markets
This repository implements a momentum-based quantitative trading strategy tailored for the foreign exchange (FX) markets. The strategy involves portfolio optimization, risk management through volatility targeting, backtesting and signal generation.

## Strategy Overview
The core of this trading system is a momentum-based alpha generation mechanism. It computes momentum signals by analyzing historical price data over specified look-back periods. These signals are then utilized to construct a long-short FX portfolio, aiming to capitalize on the persistence of currency trends.
## Key Components
### 1. Alpha Signal Generation
Momentum Calculation: The strategy calculates momentum scores based on historical returns over defined look-back periods (e.g., 3, 6, 12 months).

Signal Normalization: Momentum scores are standardized to ensure comparability across different currency pairs.

Ranking Mechanism: Currency pairs are ranked based on their momentum scores to identify potential long and short candidates.

### 2. Portfolio Construction
Position Sizing: Positions are sized based on normalized momentum scores, ensuring that stronger signals have a more significant impact on the portfolio.

Leverage Constraints: The strategy incorporates leverage limits to manage exposure and adhere to risk parameters.

### 3. Risk Management
Volatility Targeting: Portfolio volatility is monitored and adjusted to align with predefined risk targets, ensuring consistent risk exposure over time.

Drawdown Control: Mechanisms are in place to monitor and mitigate significant portfolio drawdowns, preserving capital during adverse market conditions.

### 4. Backtesting Framework
Historical Data Analysis: The system allows for backtesting over historical FX data to evaluate strategy performance.
GitHub

## Technical Stack
Programming Language: Python, Pandas, NumPy, OANDA Python API, OOP,  Software Design Principles

## Performance Evaluation
The strategy's performance was assessed using the following metrics:

Cumulative Returns: Total return over the backtesting period.
Annualized Volatility: Standard deviation of returns, annualized.
Sharpe Ratio: Risk-adjusted return metric.
Maximum Drawdown: Largest peak-to-trough decline.
Win Rate: Percentage of profitable trades.

## Notes
Additional considerations and possible improvements to the strategy could involve slippage, transaction costs, and real-time data handling.
