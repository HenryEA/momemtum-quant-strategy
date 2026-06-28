from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from quant_stock_momentum.backtest.engine import BacktestEngine, save_report
from quant_stock_momentum.config import load_sector_map, load_stock_universe, load_yaml
from quant_stock_momentum.data.loader import load_market_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for stock momentum strategy.")
    p.add_argument("--data-dir", default="data/raw/yfinance")
    p.add_argument("--instruments", default="configs/instruments.yml")
    p.add_argument("--strategy", default="configs/strategy.yml")
    p.add_argument("--out", default="data/reports/walk_forward")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--min-rows", type=int, default=252)
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--validation-years", type=int, default=1)
    p.add_argument("--test-years", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.strategy)
    instruments = load_stock_universe(args.instruments, include_benchmark=True)
    sector_map = load_sector_map(args.instruments)
    allow_missing = bool(args.allow_missing or base_cfg.get("backtest", {}).get("allow_missing_data", False))
    bars = load_market_bars(args.data_dir, instruments, use_adjusted_close=True, allow_missing=allow_missing, min_rows=args.min_rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_calendar_folds(bars.close.index, args.train_years, args.validation_years, args.test_years)
    if not folds:
        raise RuntimeError("Not enough data to build walk-forward folds.")

    grid = build_parameter_grid()
    rows = []
    oos_returns = []
    for fold_no, fold in enumerate(folds, start=1):
        val_scores = []
        for candidate_no, patch in enumerate(grid, start=1):
            cfg = deep_merge(copy.deepcopy(base_cfg), patch)
            cfg.setdefault("backtest", {})["start_date"] = fold["validation_start"]
            cfg.setdefault("backtest", {})["end_date"] = fold["validation_end"]
            res = BacktestEngine(cfg, sector_map=sector_map).run(bars)
            score = float(res.metrics.get("sharpe", 0.0)) + 0.5 * float(res.metrics.get("calmar", 0.0))
            # Penalize very large drawdown and excessive turnover.
            score -= max(0.0, abs(float(res.metrics.get("max_drawdown", 0.0))) - 0.20)
            score -= max(0.0, float(res.metrics.get("average_turnover", 0.0)) - 0.20)
            val_scores.append((score, candidate_no, patch, res.metrics))
        val_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_no, best_patch, best_val_metrics = val_scores[0]
        test_cfg = deep_merge(copy.deepcopy(base_cfg), best_patch)
        test_cfg.setdefault("backtest", {})["start_date"] = fold["test_start"]
        test_cfg.setdefault("backtest", {})["end_date"] = fold["test_end"]
        test_res = BacktestEngine(test_cfg, sector_map=sector_map).run(bars)
        fold_dir = out_dir / f"fold_{fold_no:02d}"
        save_report(test_res, fold_dir)
        r = test_res.returns[["net_return"]].rename(columns={"net_return": f"fold_{fold_no}_net_return"})
        oos_returns.append(r)
        rows.append(
            {
                "fold": fold_no,
                **fold,
                "chosen_candidate": best_no,
                "validation_score": best_score,
                "validation_sharpe": best_val_metrics.get("sharpe", 0.0),
                "validation_max_drawdown": best_val_metrics.get("max_drawdown", 0.0),
                "test_sharpe": test_res.metrics.get("sharpe", 0.0),
                "test_cagr": test_res.metrics.get("cagr", 0.0),
                "test_max_drawdown": test_res.metrics.get("max_drawdown", 0.0),
                "test_total_return": test_res.metrics.get("total_return", 0.0),
                "patch": json.dumps(best_patch, sort_keys=True),
            }
        )
        print(f"Fold {fold_no}: selected candidate {best_no}; test Sharpe={test_res.metrics.get('sharpe', 0.0):.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    combined = pd.concat(oos_returns, axis=1).fillna(0.0)
    combined["oos_net_return"] = combined.sum(axis=1)
    combined.to_csv(out_dir / "oos_returns.csv")
    print(summary.to_string(index=False))
    print(f"Saved -> {out_dir.resolve()}")


def make_calendar_folds(index: pd.DatetimeIndex, train_years: int, validation_years: int, test_years: int) -> list[dict[str, str]]:
    start_year = int(index.min().year)
    end_year = int(index.max().year)
    folds = []
    y = start_year
    while True:
        train_start = y
        train_end = y + train_years - 1
        val_start = train_end + 1
        val_end = val_start + validation_years - 1
        test_start = val_end + 1
        test_end = test_start + test_years - 1
        if test_end > end_year:
            break
        folds.append(
            {
                "train_start": f"{train_start}-01-01",
                "train_end": f"{train_end}-12-31",
                "validation_start": f"{val_start}-01-01",
                "validation_end": f"{val_end}-12-31",
                "test_start": f"{test_start}-01-01",
                "test_end": f"{test_end}-12-31",
            }
        )
        y += test_years
    return folds


def build_parameter_grid() -> list[dict]:
    # Conservative walk-forward grid. It is broad enough to test the major design
    # choices but small enough to avoid turning validation into an overfit search.
    grid = []
    for mode in ["adaptive_total_return", "adaptive_core_satellite", "long_only", "core_satellite", "long_short_beta_neutral"]:
        for target_vol in [0.14, 0.18, 0.20]:
            for top_q, bottom_q in [(0.35, 0.15), (0.40, 0.15), (0.45, 0.10), (0.55, 0.10)]:
                for rebalance in ["ME", "W-FRI"]:
                    patch = {
                        "strategy": {
                            "top_quantile": top_q,
                            "bottom_quantile": bottom_q,
                            "long_short": mode != "long_only",
                            "sector_neutralize": mode == "long_short_beta_neutral",
                            "cross_sectional_demean": mode == "long_short_beta_neutral",
                        },
                        "portfolio": {
                            "mode": mode,
                            "target_vol": target_vol,
                            "rebalance_frequency": rebalance,
                        },
                    }
                    if mode == "adaptive_total_return":
                        patch["portfolio"].update({
                            "core_weight": 0.55,
                            "diversified_weight": 0.22,
                            "universe_weight": 0.18,
                            "satellite_weight": 0.05,
                            "max_net_exposure": 1.20,
                            "max_gross_leverage": 1.35,
                        })
                    elif mode == "adaptive_core_satellite":
                        patch["portfolio"].update({
                            "core_weight": 0.72,
                            "diversified_weight": 0.23,
                            "satellite_weight": 0.05,
                            "max_net_exposure": 1.10,
                            "max_gross_leverage": 1.25,
                        })
                    elif mode == "core_satellite":
                        patch["portfolio"].update({"core_weight": 0.88, "satellite_weight": 0.12, "max_net_exposure": 1.00})
                    elif mode == "long_only":
                        patch["portfolio"].update({"max_net_exposure": 1.00, "max_gross_leverage": 1.10})
                    else:
                        patch["portfolio"].update({"max_net_exposure": 0.10, "max_gross_leverage": 1.25})
                    grid.append(patch)
    return grid


def deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


if __name__ == "__main__":
    main()
