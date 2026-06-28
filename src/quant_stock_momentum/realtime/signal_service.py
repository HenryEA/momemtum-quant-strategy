from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


@dataclass(frozen=True)
class LatestSignal:
    ticker: str
    target_weight: float
    signal: float


class LatestSignalService:
    """Generates latest research signals from newest local stock data files.

    This is not a broker/execution service. It is intentionally file-based so it can
    be used for research and paper-trading review without placing orders.
    """

    def __init__(
        self,
        instruments_path: str | Path,
        strategy_path: str | Path,
        data_dir: str | Path,
        allow_missing: bool = False,
    ) -> None:
        self.instruments_path = instruments_path
        self.strategy_path = strategy_path
        self.data_dir = data_dir
        self.allow_missing = allow_missing

    def latest(self) -> pd.DataFrame:
        cfg = load_yaml(self.strategy_path)
        instruments = load_stock_universe(self.instruments_path, include_benchmark=True)
        sector_map = load_sector_map(self.instruments_path)
        use_adjusted = bool(cfg.get("backtest", {}).get("use_adjusted_close", True))
        allow_missing = bool(self.allow_missing or cfg.get("backtest", {}).get("allow_missing_data", False))
        bars = load_market_bars(self.data_dir, instruments, use_adjusted_close=use_adjusted, allow_missing=allow_missing)
        engine = BacktestEngine(cfg, sector_map=sector_map)
        result = engine.run(bars)
        weights = result.weights.iloc[-1].rename("target_weight")
        signals = result.signals.iloc[-1].rename("signal")
        out = pd.concat([weights, signals], axis=1).reset_index().rename(columns={"index": "ticker"})
        out["asof"] = result.weights.index[-1]
        return out.sort_values("target_weight", key=lambda s: s.abs(), ascending=False)
