from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quant_stock_momentum.config import load_stock_universe
from quant_stock_momentum.data.dukascopy_stock_client import DukascopyStockClient, DukascopyStockRequest
from quant_stock_momentum.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Dukascopy stock CFD OHLCV data for configured stocks.")
    parser.add_argument("--instruments", default="configs/instruments.yml")
    parser.add_argument("--data-dir", default="data/raw/dukascopy")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", default="d1", choices=["d1", "h1", "m30", "m15", "m5", "m1"])
    parser.add_argument("--include-benchmark", action="store_true", help="Also download benchmark from configs/instruments.yml")
    parser.add_argument("--skip-failed", action="store_true", help="Continue downloading if one symbol fails.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    instruments = load_stock_universe(args.instruments, include_benchmark=args.include_benchmark)
    client = DukascopyStockClient()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    failures: list[str] = []
    for inst in instruments:
        print(f"Downloading {inst.ticker} from Dukascopy symbol {inst.dukascopy_symbol}...")
        req = DukascopyStockRequest(
            ticker=inst.ticker,
            dukascopy_symbol=inst.dukascopy_symbol or "",
            start=start,
            end=end,
            timeframe=args.timeframe,
        )
        try:
            out = client.download_to_csv(req, data_dir)
            print(f"Saved -> {out}")
        except Exception as exc:
            msg = f"{inst.ticker}: {exc}"
            failures.append(msg)
            print(f"FAILED -> {msg}")
            if not args.skip_failed:
                raise
    if failures:
        print("\nDownload completed with failures:")
        for item in failures:
            print(f"- {item}")


if __name__ == "__main__":
    main()
