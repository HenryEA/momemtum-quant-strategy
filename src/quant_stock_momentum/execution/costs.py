from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionCostConfig:
    commission_bps: float = 0.50
    default_half_spread_bps: float = 1.50
    slippage_bps: float = 1.00
    impact_bps_per_100pct_turnover: float = 0.25
    volatility_slippage_multiplier: float = 0.00
    half_spread_bps_by_symbol: dict[str, float] | None = None

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "ExecutionCostConfig":
        return ExecutionCostConfig(
            commission_bps=float(cfg.get("commission_bps", 0.50)),
            default_half_spread_bps=float(cfg.get("default_half_spread_bps", 1.50)),
            slippage_bps=float(cfg.get("slippage_bps", 1.00)),
            impact_bps_per_100pct_turnover=float(cfg.get("impact_bps_per_100pct_turnover", 0.25)),
            volatility_slippage_multiplier=float(cfg.get("volatility_slippage_multiplier", 0.00)),
            half_spread_bps_by_symbol={
                str(k).upper(): float(v) for k, v in (cfg.get("half_spread_bps_by_symbol", {}) or {}).items()
            },
        )


class ExecutionCostModel:
    """Weight-turnover transaction cost model.

    Cost is charged when target weights change. For a delta weight of 0.10 and a 5 bps all-in
    cost assumption, the portfolio return is reduced by 0.10 * 0.0005 = 0.00005.
    """

    def __init__(self, cfg: ExecutionCostConfig) -> None:
        self.cfg = cfg

    def turnover_by_symbol(self, weights: pd.DataFrame) -> pd.DataFrame:
        return weights.diff().fillna(weights).abs()

    def turnover_series(self, weights: pd.DataFrame) -> pd.Series:
        return self.turnover_by_symbol(weights).sum(axis=1).rename("turnover")

    def cost_series(self, weights: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.Series:
        delta = self.turnover_by_symbol(weights)
        per_symbol_bps = pd.Series(
            {
                col: self.cfg.commission_bps
                + self.cfg.slippage_bps
                + (self.cfg.half_spread_bps_by_symbol or {}).get(col.upper(), self.cfg.default_half_spread_bps)
                for col in weights.columns
            }
        )
        linear = delta.mul(per_symbol_bps, axis=1).sum(axis=1) / 10000.0
        turnover = delta.sum(axis=1)
        impact = self.cfg.impact_bps_per_100pct_turnover * (turnover**2) / 10000.0
        vol_slippage = pd.Series(0.0, index=weights.index)
        if returns is not None and self.cfg.volatility_slippage_multiplier > 0:
            # Conservative, simple volatility-aware slippage: when today's rebalance occurs in a volatile
            # stock, charged slippage increases with recent daily volatility.
            daily_vol = returns.rolling(20, min_periods=5).std().reindex(delta.index).fillna(0.0)
            vol_slippage = (delta * daily_vol * self.cfg.volatility_slippage_multiplier).sum(axis=1)
        return (linear + impact + vol_slippage).rename("cost_return")

    def with_cost_multiplier(self, multiplier: float) -> "ExecutionCostModel":
        cfg = ExecutionCostConfig(
            commission_bps=self.cfg.commission_bps * multiplier,
            default_half_spread_bps=self.cfg.default_half_spread_bps * multiplier,
            slippage_bps=self.cfg.slippage_bps * multiplier,
            impact_bps_per_100pct_turnover=self.cfg.impact_bps_per_100pct_turnover * multiplier,
            volatility_slippage_multiplier=self.cfg.volatility_slippage_multiplier * multiplier,
            half_spread_bps_by_symbol={k: v * multiplier for k, v in (self.cfg.half_spread_bps_by_symbol or {}).items()},
        )
        return ExecutionCostModel(cfg)
