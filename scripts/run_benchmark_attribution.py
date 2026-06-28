from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create benchmark/equal-weight attribution diagnostics for a completed strategy run.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--strategy", default="configs/strategy.yml")
    p.add_argument("--out", default="data/reports/benchmark_attribution.csv")
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.strategy)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    allow_missing = bool(args.allow_missing or cfg.get("backtest", {}).get("allow_missing_data", False))
    bars = load_market_bars(args.data_dir, instruments, use_adjusted_close=True, allow_missing=allow_missing)
    res = BacktestEngine(cfg, sector_map=sector_map).run(bars)

    returns = res.returns.copy()
    strategy = returns["net_return"].rename("strategy")
    bench_cols = [c for c in returns.columns if c.endswith("_return") and c not in {"gross_return", "cost_return", "net_return"}]
    benchmark = returns[bench_cols[0]].rename("benchmark") if bench_cols else pd.Series(0.0, index=strategy.index, name="benchmark")
    equal_weight = returns["equal_weight_stock_return"].rename("equal_weight")
    active_vs_benchmark = (strategy - benchmark).rename("active_vs_benchmark")
    active_vs_equal = (strategy - equal_weight).rename("active_vs_equal_weight")

    table = pd.concat([strategy, benchmark, equal_weight, active_vs_benchmark, active_vs_equal], axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)

    annual = table.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual.to_csv(out.with_name(out.stem + "_annual.csv"))
    corr = table.corr()
    corr.to_csv(out.with_name(out.stem + "_correlation.csv"))

    print("Annual return attribution:")
    print(annual.tail(12).to_string())
    print("\nCorrelations:")
    print(corr.to_string())
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
