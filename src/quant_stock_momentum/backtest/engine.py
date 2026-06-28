from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_stock_momentum.backtest.metrics import drawdown, monthly_returns_table, summarize_performance
from quant_stock_momentum.backtest.result import BacktestResult
from quant_stock_momentum.data.loader import MarketBars
from quant_stock_momentum.execution.costs import ExecutionCostConfig, ExecutionCostModel
from quant_stock_momentum.features.indicators import sma
from quant_stock_momentum.features.signals import StockMomentumSignalConfig, StockMomentumStrategy
from quant_stock_momentum.portfolio.optimizer import PortfolioConfig
from quant_stock_momentum.portfolio.optimizer import VolTargetStockOptimizer


class BacktestEngine:
    def __init__(self, config: dict[str, Any], sector_map: dict[str, str] | None = None) -> None:
        self.config = config
        self.sector_map = sector_map or {}
        self.trading_days = int(config.get("backtest", {}).get("trading_days", 252))
        self.initial_capital = float(config.get("backtest", {}).get("initial_capital", 10000))
        self.signal_cfg = StockMomentumSignalConfig.from_dict(config.get("strategy", {}) or {})
        self.portfolio_cfg = PortfolioConfig.from_dict(config.get("portfolio", {}) or {})
        self.execution_cfg = ExecutionCostConfig.from_dict(config.get("execution", {}) or {})

    def run(self, bars: MarketBars) -> BacktestResult:
        start_date = self.config.get("backtest", {}).get("start_date")
        end_date = self.config.get("backtest", {}).get("end_date")
        bars = bars.slice_dates(start_date, end_date)

        close = bars.close.astype(float)
        returns = close.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

        benchmark_symbol = self.portfolio_cfg.market_regime.benchmark
        benchmark_close = close[benchmark_symbol] if benchmark_symbol in close.columns else None
        benchmark_returns = returns[benchmark_symbol] if benchmark_symbol in returns.columns else None
        trade_cols = [c for c in close.columns if c != benchmark_symbol]
        if len(trade_cols) < 2:
            raise ValueError("Need at least two tradable stock columns after excluding benchmark.")

        strategy = StockMomentumStrategy(self.signal_cfg, sector_map=self.sector_map)
        signals = strategy.generate(bars, trading_days=self.trading_days).reindex(columns=trade_cols).fillna(0.0)
        returns_trade = returns[trade_cols]
        benchmark_trade = benchmark_returns.reindex(returns_trade.index).fillna(0.0) if benchmark_returns is not None else None

        optimizer = VolTargetStockOptimizer(self.portfolio_cfg, trading_days=self.trading_days, sector_map=self.sector_map)
        target_weights = optimizer.allocate(returns_trade, signals, benchmark_returns=benchmark_trade)
        target_weights = self._apply_exogenous_market_regime(target_weights, benchmark_close)

        sim = self._simulate(returns_trade, target_weights)
        executed_weights = sim["weights"]
        gross_ret = sim["gross_return"]
        cost = sim["cost_return"]
        turnover = sim["turnover"]
        net_ret = sim["net_return"]
        equity = self.initial_capital * (1.0 + net_ret).cumprod()

        equal_weight_ret = returns_trade.mean(axis=1).fillna(0.0)
        equal_weight_equity = self.initial_capital * (1.0 + equal_weight_ret).cumprod()

        returns_df = pd.DataFrame(
            {
                "gross_return": gross_ret,
                "cost_return": cost,
                "net_return": net_ret,
                "turnover": turnover,
                "equal_weight_stock_return": equal_weight_ret,
            }
        )
        equity_df = pd.DataFrame(
            {
                "equity": equity,
                "drawdown": drawdown(equity),
                "equal_weight_stock_equity": equal_weight_equity,
            }
        )
        if benchmark_returns is not None:
            bret = benchmark_returns.reindex(equity.index).fillna(0.0)
            equity_df[f"{benchmark_symbol}_equity"] = self.initial_capital * (1.0 + bret).cumprod()
            returns_df[f"{benchmark_symbol}_return"] = bret

        metrics = summarize_performance(
            net_returns=net_ret,
            gross_returns=gross_ret,
            equity=equity,
            costs=cost,
            turnover=turnover,
            weights=executed_weights,
            trading_days=self.trading_days,
            benchmark_returns=benchmark_returns if benchmark_returns is not None else None,
        )
        from quant_stock_momentum.backtest.metrics import annualized_vol, cagr, sharpe_ratio

        metrics["portfolio_mode"] = self.portfolio_cfg.mode
        metrics["tradable_symbols"] = int(len(trade_cols))
        metrics["equal_weight_cagr"] = cagr(equal_weight_equity, self.trading_days)
        metrics["equal_weight_sharpe"] = sharpe_ratio(equal_weight_ret, self.trading_days)
        metrics["equal_weight_volatility"] = annualized_vol(equal_weight_ret, self.trading_days)
        metrics["avg_active_names"] = float((executed_weights.abs() > 0).sum(axis=1).mean()) if not executed_weights.empty else 0.0

        return BacktestResult(
            returns=returns_df,
            equity_curve=equity_df,
            weights=executed_weights,
            signals=signals,
            costs=pd.DataFrame({"cost_return": cost, "turnover": turnover}),
            metrics=metrics,
        )

    def _simulate(self, returns_trade: pd.DataFrame, target_weights: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        cost_model = ExecutionCostModel(self.execution_cfg)
        idx = returns_trade.index
        cols = returns_trade.columns
        weights = pd.DataFrame(0.0, index=idx, columns=cols)
        gross_ret = pd.Series(0.0, index=idx, name="gross_return")
        cost = pd.Series(0.0, index=idx, name="cost_return")
        turnover = pd.Series(0.0, index=idx, name="turnover")
        net_ret = pd.Series(0.0, index=idx, name="net_return")
        equity = pd.Series(self.initial_capital, index=idx, dtype=float)
        prev_weight = pd.Series(0.0, index=cols)
        peak = self.initial_capital

        for i, dt in enumerate(idx):
            desired = target_weights.shift(1).fillna(0.0).loc[dt].reindex(cols).fillna(0.0)
            dd = equity.iloc[i - 1] / peak - 1.0 if i > 0 and peak > 0 else 0.0
            multiplier = self._drawdown_multiplier(dd)
            w = desired * multiplier
            weights.loc[dt] = w
            delta = (w - prev_weight).abs().to_frame().T
            delta.index = [dt]
            # Cost model expects full weight frames; pass two-row frame to keep diff logic simple.
            tmp_weights = pd.DataFrame([prev_weight, w], index=[dt - pd.Timedelta(nanoseconds=1), dt])
            tmp_returns = returns_trade.loc[:dt].tail(21)
            c = float(cost_model.cost_series(tmp_weights, tmp_returns).iloc[-1])
            t = float(delta.sum(axis=1).iloc[0])
            gr = float((w * returns_trade.loc[dt]).sum())
            nr = gr - c
            gross_ret.loc[dt] = gr
            cost.loc[dt] = c
            turnover.loc[dt] = t
            net_ret.loc[dt] = nr
            equity.iloc[i] = (equity.iloc[i - 1] if i > 0 else self.initial_capital) * (1.0 + nr)
            peak = max(peak, float(equity.iloc[i]))
            prev_weight = w

        return {"weights": weights, "gross_return": gross_ret, "cost_return": cost, "turnover": turnover, "net_return": net_ret}

    def _drawdown_multiplier(self, strategy_drawdown: float) -> float:
        cfg = self.portfolio_cfg.drawdown_brake
        if not cfg.enabled:
            return 1.0
        if strategy_drawdown <= cfg.hard_drawdown:
            return cfg.hard_multiplier
        if strategy_drawdown <= cfg.soft_drawdown:
            return cfg.soft_multiplier
        return 1.0

    def _apply_exogenous_market_regime(self, weights: pd.DataFrame, benchmark_close: pd.Series | None) -> pd.DataFrame:
        cfg = self.portfolio_cfg.market_regime
        if not cfg.enabled or benchmark_close is None:
            return weights
        benchmark_close = benchmark_close.reindex(weights.index).ffill()
        regime_sma = sma(benchmark_close, cfg.sma)
        benchmark_ret = benchmark_close.pct_change(fill_method=None).fillna(0.0)
        realized_vol = benchmark_ret.rolling(cfg.high_vol_lookback, min_periods=max(10, cfg.high_vol_lookback // 3)).std() * np.sqrt(self.trading_days)
        risk_off = benchmark_close < regime_sma
        high_vol = realized_vol > cfg.high_vol_threshold
        risk_on = (benchmark_close > regime_sma) & (realized_vol < cfg.risk_on_low_vol_threshold)

        # Breadth is computed from the number of positive holdings. It avoids
        # aggressive risk-on scaling when leadership is narrow.
        long_breadth = (weights > 0).sum(axis=1) / max(1, weights.shape[1])
        breadth_ok = long_breadth.reindex(weights.index).fillna(0.0) >= cfg.min_risk_on_breadth

        panic = (benchmark_close / benchmark_close.shift(cfg.panic_lookback) - 1.0) < cfg.panic_drawdown_threshold
        rebound = (benchmark_close / benchmark_close.shift(cfg.rebound_lookback) - 1.0) > cfg.rebound_threshold
        panic_rebound = panic & rebound

        out = weights.copy()
        multiplier = pd.Series(1.0, index=weights.index)
        risk_on_idx = (risk_on.reindex(weights.index).fillna(False) & breadth_ok)
        multiplier.loc[risk_on_idx] *= cfg.risk_on_gross_multiplier * cfg.risk_on_breadth_multiplier
        multiplier.loc[risk_off.reindex(weights.index).fillna(False)] *= cfg.risk_off_gross_multiplier
        multiplier.loc[high_vol.reindex(weights.index).fillna(False)] *= cfg.high_vol_multiplier
        out = out.mul(multiplier, axis=0)
        # Momentum crashes often come from shorting rebound losers. Reduce shorts in those states.
        pr = panic_rebound.reindex(weights.index).fillna(False)
        if pr.any():
            short_part = out.where(out < 0, 0.0)
            long_part = out.where(out > 0, 0.0)
            out.loc[pr] = long_part.loc[pr] + short_part.loc[pr] * cfg.panic_rebound_short_multiplier
        # Keep exogenous scaling inside the configured gross cap.
        gross = out.abs().sum(axis=1).replace(0, np.nan)
        too_big = gross > self.portfolio_cfg.max_gross_leverage
        if too_big.any():
            out.loc[too_big] = out.loc[too_big].div(gross.loc[too_big], axis=0) * self.portfolio_cfg.max_gross_leverage
        return out.fillna(0.0)


def save_report(result: BacktestResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(out_dir / "returns.csv")
    result.equity_curve.to_csv(out_dir / "equity_curve.csv")
    result.weights.to_csv(out_dir / "weights.csv")
    result.signals.to_csv(out_dir / "signals.csv")
    result.costs.to_csv(out_dir / "costs_turnover.csv")
    monthly_returns_table(result.returns["net_return"]).to_csv(out_dir / "monthly_returns.csv")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2, sort_keys=True)
    _save_tearsheet(result, out_dir / "tearsheet.png")


def _save_tearsheet(result: BacktestResult, path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    equity_cols = [c for c in result.equity_curve.columns if c.endswith("equity") or c == "equity"]
    result.equity_curve[equity_cols].plot(ax=axes[0], title="Equity Curve")
    result.equity_curve["drawdown"].plot(ax=axes[1], title="Strategy Drawdown")
    result.weights.abs().sum(axis=1).plot(ax=axes[2], title="Gross Exposure")
    result.weights.sum(axis=1).plot(ax=axes[3], title="Net Exposure")
    result.returns["turnover"].plot(ax=axes[4], title="Turnover")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
