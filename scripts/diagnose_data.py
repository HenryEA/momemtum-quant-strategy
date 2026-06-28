from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_stock_momentum.config import load_stock_universe
from quant_stock_momentum.data.loader import read_symbol_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose stock OHLCV data quality before backtesting.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--include-benchmark", action="store_true", default=True)
    p.add_argument("--out", default="data/reports/data_quality.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    rows = []
    for inst in load_stock_universe(args.instruments, include_benchmark=True):
        candidates = [data_dir / f"{inst.ticker}.csv"]
        if inst.dukascopy_symbol:
            candidates.append(data_dir / f"{inst.dukascopy_symbol}.csv")
        path = next((p for p in candidates if p.exists()), candidates[0])
        try:
            df = read_symbol_csv(path, inst.ticker, use_adjusted_close=False)
            ret = df["close"].pct_change(fill_method=None)
            rows.append(
                {
                    "ticker": inst.ticker,
                    "sector": inst.sector or "Unknown",
                    "path": str(path),
                    "exists": True,
                    "rows": len(df),
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                    "zero_volume_fraction": float((df["volume"] <= 0).mean()),
                    "flat_close_fraction": float((ret.fillna(0).abs() == 0).mean()),
                    "nan_close_fraction": float(df["close"].isna().mean()),
                    "min_close": float(df["close"].min()),
                    "max_close": float(df["close"].max()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "ticker": inst.ticker,
                    "sector": inst.sector or "Unknown",
                    "path": str(path),
                    "exists": path.exists(),
                    "rows": 0,
                    "error": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
