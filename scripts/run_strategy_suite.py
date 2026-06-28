from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine, save_report
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the standard strategy comparison suite.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--out", default="data/reports/strategy_suite")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--min-rows", type=int, default=252)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    strategy_files = {
        "v5_default_adaptive_total_return": "configs/strategy.yml",
        "v5_balanced": "configs/strategy_balanced.yml",
        "v5_growth": "configs/strategy_growth.yml",
        "v5_sharpe": "configs/strategy_sharpe.yml",
        "v5_equal_weight_plus": "configs/strategy_equal_weight_plus.yml",
        "v4_adaptive_total_return": "configs/strategy_v4_adaptive_total_return.yml",
        "v4_balanced": "configs/strategy_v4_balanced.yml",
        "v4_high_participation": "configs/strategy_v4_high_participation.yml",
        "v4_sharpe_defensive": "configs/strategy_v4_sharpe_defensive.yml",
        "v3_baseline": "configs/strategy_v3_baseline.yml",
        "long_only": "configs/strategy_long_only.yml",
        "market_neutral": "configs/strategy_market_neutral.yml",
    }
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    bars = load_market_bars(args.data_dir, instruments, use_adjusted_close=True, allow_missing=args.allow_missing, min_rows=args.min_rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, strategy_path in strategy_files.items():
        cfg = load_yaml(strategy_path)
        res = BacktestEngine(cfg, sector_map=sector_map).run(bars)
        save_report(res, out_dir / name)
        rows.append({"strategy": name, **res.metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "strategy_suite_summary.csv", index=False)
    cols = [
        "strategy", "portfolio_mode", "total_return", "cagr", "annualized_volatility", "sharpe", "sortino",
        "max_drawdown", "benchmark_beta", "annualized_alpha_vs_benchmark", "average_turnover",
        "average_gross_exposure", "average_net_exposure", "tradable_symbols",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
