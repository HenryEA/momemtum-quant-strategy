from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine, save_report
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select among predefined strategy configs on validation data, then report out-of-sample test performance."
    )
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--out", default="data/reports/oos_config_selection")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--min-rows", type=int, default=252)
    p.add_argument("--train-start", default="2014-01-01")
    p.add_argument("--train-end", default="2018-12-31")
    p.add_argument("--validation-start", default="2019-01-01")
    p.add_argument("--validation-end", default="2021-12-31")
    p.add_argument("--test-start", default="2022-01-01")
    p.add_argument("--test-end", default=None)
    return p.parse_args()


def robust_score(metrics: dict) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    calmar = float(metrics.get("calmar", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    dd = abs(float(metrics.get("max_drawdown", 0.0)))
    turnover = float(metrics.get("average_turnover", 0.0))
    # Score prefers persistent risk-adjusted returns and penalizes deep drawdown
    # and excessive turnover. It is deliberately simple to reduce selection bias.
    return sharpe + 0.35 * calmar + 0.50 * cagr - max(0.0, dd - 0.25) - 0.50 * turnover


def with_dates(cfg: dict, start: str | None, end: str | None) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("backtest", {})["start_date"] = start
    out.setdefault("backtest", {})["end_date"] = end
    return out


def main() -> None:
    args = parse_args()
    strategy_files = {
        "v5_default": "configs/strategy.yml",
        "v5_balanced": "configs/strategy_balanced.yml",
        "v5_growth": "configs/strategy_growth.yml",
        "v5_sharpe": "configs/strategy_sharpe.yml",
        "v5_equal_weight_plus": "configs/strategy_equal_weight_plus.yml",
        "v4_default": "configs/strategy_v4_adaptive_total_return.yml",
        "v4_balanced": "configs/strategy_v4_balanced.yml",
        "v4_high_participation": "configs/strategy_v4_high_participation.yml",
        "v4_sharpe_defensive": "configs/strategy_v4_sharpe_defensive.yml",
        "v3_baseline": "configs/strategy_v3_baseline.yml",
    }
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    bars = load_market_bars(
        args.data_dir,
        instruments,
        use_adjusted_close=True,
        allow_missing=args.allow_missing,
        min_rows=args.min_rows,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    full_results = {}
    for name, path in strategy_files.items():
        base = load_yaml(path)
        train = BacktestEngine(with_dates(base, args.train_start, args.train_end), sector_map=sector_map).run(bars)
        val = BacktestEngine(with_dates(base, args.validation_start, args.validation_end), sector_map=sector_map).run(bars)
        test = BacktestEngine(with_dates(base, args.test_start, args.test_end), sector_map=sector_map).run(bars)
        full = BacktestEngine(with_dates(base, None, None), sector_map=sector_map).run(bars)
        full_results[name] = full
        rows.append(
            {
                "strategy": name,
                "config": path,
                "validation_score": robust_score(val.metrics),
                "train_sharpe": train.metrics.get("sharpe"),
                "train_cagr": train.metrics.get("cagr"),
                "validation_sharpe": val.metrics.get("sharpe"),
                "validation_cagr": val.metrics.get("cagr"),
                "validation_max_drawdown": val.metrics.get("max_drawdown"),
                "test_sharpe": test.metrics.get("sharpe"),
                "test_cagr": test.metrics.get("cagr"),
                "test_max_drawdown": test.metrics.get("max_drawdown"),
                "full_sharpe": full.metrics.get("sharpe"),
                "full_cagr": full.metrics.get("cagr"),
                "full_max_drawdown": full.metrics.get("max_drawdown"),
            }
        )

    summary = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    summary.to_csv(out_dir / "oos_config_selection_summary.csv", index=False)
    best = str(summary.iloc[0]["strategy"])
    save_report(full_results[best], out_dir / f"selected_{best}_full_backtest")
    print(summary.to_string(index=False))
    print(f"Selected on validation: {best}")
    print(f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
