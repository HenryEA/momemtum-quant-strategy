import numpy as np
import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine
from quant_stock_momentum.data.loader import MarketBars


def test_engine_runs_smoke():
    idx = pd.bdate_range("2018-01-01", periods=700, tz="UTC")
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(
        rng.normal(0.0002, 0.015, size=(len(idx), 5)),
        index=idx,
        columns=["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
    )
    close = 100 * (1 + rets).cumprod()
    bars = MarketBars(open=close, high=close * 1.01, low=close * 0.99, close=close, volume=close * 0 + 1_000_000)
    cfg = {
        "backtest": {"initial_capital": 10000, "trading_days": 252},
        "strategy": {"formation_lookback": 126, "skip_recent_days": 21, "short_lookback": 42, "liquidity_filter": {"min_average_volume": 0}},
        "portfolio": {"target_vol": 0.10, "rebalance_frequency": "W-FRI", "market_regime_filter": {"enabled": True, "benchmark": "SPY", "sma": 100}},
        "execution": {},
    }
    result = BacktestEngine(cfg).run(bars)
    assert not result.equity_curve.empty
    assert "sharpe" in result.metrics
    assert set(result.weights.columns) == {"AAPL", "MSFT", "NVDA", "AMZN"}
