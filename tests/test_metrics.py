import pandas as pd

from quant_stock_momentum.backtest.metrics import cagr, drawdown, sharpe_ratio


def test_metrics_basic():
    idx = pd.bdate_range("2020-01-01", periods=252, tz="UTC")
    r = pd.Series(0.001, index=idx)
    eq = 100 * (1 + r).cumprod()
    assert cagr(eq) > 0
    assert sharpe_ratio(r) > 0
    assert drawdown(eq).min() == 0
