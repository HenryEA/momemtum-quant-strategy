from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_stock_momentum.config import load_stock_universe
from quant_stock_momentum.data.loader import save_symbol_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic stock OHLCV data for an offline smoke test.")
    parser.add_argument("--out", default="data/raw/yfinance")
    parser.add_argument("--instruments", default="configs/instruments.yml")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-benchmark", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    n = args.years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today(tz="UTC").normalize(), periods=n)

    market = rng.normal(0.00025, 0.009, size=n)
    market_trend = pd.Series(market).rolling(80, min_periods=1).mean().to_numpy()
    sector_shocks: dict[str, np.ndarray] = {}

    for j, inst in enumerate(instruments):
        sector = inst.sector or "Benchmark"
        if sector not in sector_shocks:
            sector_shocks[sector] = rng.normal(0.00002, 0.004, size=n)
        beta = 1.0 if inst.ticker == "SPY" else rng.uniform(0.65, 1.35)
        idio = rng.normal(0.00003 + 0.00002 * (j % 5), rng.uniform(0.010, 0.026), size=n)
        persistent_alpha = np.roll(market_trend, 5) * rng.uniform(0.4, 1.2)
        ret = beta * market + 0.6 * sector_shocks[sector] + idio + persistent_alpha
        if inst.ticker == "SPY":
            ret = market + rng.normal(0, 0.002, size=n)
        ret[:5] = beta * market[:5] + idio[:5]
        close = 50 * np.exp(np.cumsum(ret)) * (1 + j * 0.03)
        open_ = close * (1 + rng.normal(0, 0.002, size=n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, size=n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, size=n)))
        volume = rng.integers(800_000, 20_000_000, size=n)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": inst.ticker,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "adjusted_close": close,
            }
        )
        save_symbol_csv(df, out_dir / f"{inst.ticker}.csv")
        print(f"Saved synthetic {inst.ticker} -> {out_dir / (inst.ticker + '.csv')}")


if __name__ == "__main__":
    main()
