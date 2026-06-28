from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine, save_report
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run transaction cost stress tests.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--strategy", default="configs/strategy.yml")
    p.add_argument("--out", default="data/reports/cost_stress")
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.strategy)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    allow_missing = bool(args.allow_missing or base_cfg.get("backtest", {}).get("allow_missing_data", False))
    bars = load_market_bars(args.data_dir, instruments, use_adjusted_close=True, allow_missing=allow_missing)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "low": 0.50,
        "base": 1.00,
        "medium": 2.00,
        "high": 4.00,
    }
    rows = []
    for name, mult in scenarios.items():
        cfg = copy.deepcopy(base_cfg)
        exec_cfg = cfg.setdefault("execution", {})
        for key in ["commission_bps", "default_half_spread_bps", "slippage_bps", "impact_bps_per_100pct_turnover"]:
            exec_cfg[key] = float(exec_cfg.get(key, 0.0)) * mult
        if "half_spread_bps_by_symbol" in exec_cfg:
            exec_cfg["half_spread_bps_by_symbol"] = {k: float(v) * mult for k, v in exec_cfg["half_spread_bps_by_symbol"].items()}
        res = BacktestEngine(cfg, sector_map=sector_map).run(bars)
        save_report(res, out_dir / name)
        rows.append({"scenario": name, "cost_multiplier": mult, **res.metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cost_stress_summary.csv", index=False)
    print(summary[["scenario", "sharpe", "cagr", "max_drawdown", "annualized_cost_drag", "average_turnover"]].to_string(index=False))
    print(f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
