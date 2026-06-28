import numpy as np
import pandas as pd

from quant_stock_momentum.data.loader import MarketBars
from quant_stock_momentum.features.signals import StockMomentumSignalConfig, StockMomentumStrategy


def test_stock_momentum_selects_winners_and_losers():
    idx = pd.bdate_range("2020-01-01", periods=320, tz="UTC")
    close = pd.DataFrame(
        {
            "WIN": np.linspace(50, 150, len(idx)),
            "LOSE": np.linspace(150, 50, len(idx)),
            "MID": np.linspace(100, 105, len(idx)),
        },
        index=idx,
    )
    bars = MarketBars(open=close, high=close * 1.01, low=close * 0.99, close=close, volume=close * 0 + 1_000_000)
    cfg = StockMomentumSignalConfig(min_average_volume=0)
    sig = StockMomentumStrategy(cfg).generate(bars)
    assert sig["WIN"].iloc[-1] > 0
    assert sig["LOSE"].iloc[-1] < 0
