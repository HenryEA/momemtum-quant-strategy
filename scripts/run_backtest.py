from __future__ import annotations

import argparse
from pathlib import Path

from quant_stock_momentum.backtest.engine import BacktestEngine, save_report
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars
from quant_stock_momentum.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock momentum backtest.")
    parser.add_argument("--data-dir", default="data/raw/yfinance")
    parser.add_argument("--instruments", default="configs/instruments.yml")
    parser.add_argument("--strategy", default="configs/strategy.yml")
    parser.add_argument("--out", default="data/reports/stock_momentum_backtest")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing/unusable symbols instead of failing.")
    parser.add_argument("--min-rows", type=int, default=252)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    cfg = load_yaml(args.strategy)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    use_adjusted = bool(cfg.get("backtest", {}).get("use_adjusted_close", True))
    allow_missing = bool(args.allow_missing or cfg.get("backtest", {}).get("allow_missing_data", False))
    bars = load_market_bars(
        args.data_dir,
        instruments,
        use_adjusted_close=use_adjusted,
        allow_missing=allow_missing,
        min_rows=args.min_rows,
    )
    engine = BacktestEngine(cfg, sector_map=sector_map)
    result = engine.run(bars)
    save_report(result, args.out)
    print("Backtest complete")
    print(f"Loaded symbols: {', '.join(bars.tickers)}")
    print(f"Report folder: {Path(args.out).resolve()}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
