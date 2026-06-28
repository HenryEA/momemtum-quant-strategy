from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a compact robustness grid for v5 stock momentum.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--strategy", default="configs/strategy.yml")
    p.add_argument("--out", default="data/reports/parameter_sensitivity.csv")
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_yaml(args.strategy)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    allow_missing = bool(args.allow_missing or base.get("backtest", {}).get("allow_missing_data", False))
    bars = load_market_bars(args.data_dir, instruments, use_adjusted_close=True, allow_missing=allow_missing)

    rows = []
    for lookback in [189, 252, 315]:
        for skip in [21, 42]:
            for top_q, bottom_q in [(0.40, 0.10), (0.50, 0.07), (0.60, 0.05)]:
                for target_vol in [0.16, 0.185, 0.205]:
                    for mode in ["adaptive_total_return_v5", "adaptive_total_return", "adaptive_core_satellite", "long_only"]:
                        cfg = copy.deepcopy(base)
                        cfg.setdefault("strategy", {})["formation_lookback"] = lookback
                        cfg.setdefault("strategy", {})["skip_recent_days"] = skip
                        cfg["strategy"]["top_quantile"] = top_q
                        cfg["strategy"]["bottom_quantile"] = bottom_q
                        cfg["strategy"]["long_short"] = mode != "long_only"
                        cfg.setdefault("portfolio", {})["target_vol"] = target_vol
                        cfg["portfolio"]["mode"] = mode
                        res = BacktestEngine(cfg, sector_map=sector_map).run(bars)
                        robust_score = float(res.metrics.get("sharpe", 0.0)) + 0.4 * float(res.metrics.get("calmar", 0.0))
                        robust_score -= max(0.0, abs(float(res.metrics.get("max_drawdown", 0.0))) - 0.22)
                        rows.append(
                            {
                                "formation_lookback": lookback,
                                "skip_recent_days": skip,
                                "top_quantile": top_q,
                                "bottom_quantile": bottom_q,
                                "target_vol": target_vol,
                                "mode": mode,
                                "robust_score": robust_score,
                                **res.metrics,
                            }
                        )
    df = pd.DataFrame(rows).sort_values(["robust_score", "sharpe", "calmar"], ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    show = ["mode", "formation_lookback", "skip_recent_days", "top_quantile", "bottom_quantile", "target_vol", "robust_score", "sharpe", "cagr", "max_drawdown", "average_turnover"]
    print(df[[c for c in show if c in df.columns]].head(20).to_string(index=False))
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
