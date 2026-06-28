from __future__ import annotations

import argparse

from quant_stock_momentum.config import load_stock_universe
from quant_stock_momentum.data.yfinance_client import YFinanceStockClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download adjusted stock data using yfinance.")
    parser.add_argument("--instruments", default="configs/instruments.yml")
    parser.add_argument("--data-dir", default="data/raw/yfinance")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--include-benchmark", action="store_true")
    parser.add_argument("--auto-adjust", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruments = load_stock_universe(args.instruments, include_benchmark=args.include_benchmark)
    saved = YFinanceStockClient().download(
        instruments,
        args.data_dir,
        period=args.period,
        start=args.start,
        end=args.end,
        auto_adjust=args.auto_adjust,
        skip_failed=args.skip_failed,
    )
    for path in saved:
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
