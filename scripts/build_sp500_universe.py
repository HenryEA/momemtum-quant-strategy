from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from quant_stock_momentum.config import default_dukascopy_symbol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an expanded instruments.yml from the current S&P 500 Wikipedia table.")
    p.add_argument("--out", default="configs/instruments_sp500.yml")
    p.add_argument("--max-symbols", type=int, default=120, help="Limit universe size for quicker downloads/backtests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    # Keep the first N by table order. For production, replace this with point-in-time constituents.
    df = df.head(args.max_symbols)
    stocks = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).replace(".", "-").upper()
        stocks.append(
            {
                "ticker": ticker,
                "name": str(row["name"]),
                "sector": str(row["sector"]),
                "industry": str(row["industry"]),
                "dukascopy_symbol": default_dukascopy_symbol(ticker),
            }
        )
    cfg = {
        "stocks": stocks,
        "benchmark": {
            "ticker": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "sector": "Benchmark",
            "dukascopy_symbol": "spyususd",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Saved -> {out.resolve()}")
    print("Warning: this is not a point-in-time universe. Use CRSP/Compustat/Norgate for professional research.")


if __name__ == "__main__":
    main()
