from __future__ import annotations

import argparse
from pathlib import Path

from quant_stock_momentum.realtime.signal_service import LatestSignalService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest stock momentum research signals from local data.")
    parser.add_argument("--data-dir", default="data/raw/yfinance")
    parser.add_argument("--instruments", default="configs/instruments.yml")
    parser.add_argument("--strategy", default="configs/strategy.yml")
    parser.add_argument("--out", default="data/reports/latest_stock_signals.csv")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LatestSignalService(args.instruments, args.strategy, args.data_dir, allow_missing=args.allow_missing)
    signals = service.latest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out, index=False)
    print(signals.to_string(index=False))
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
