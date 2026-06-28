from __future__ import annotations

import math

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def cagr(equity: pd.Series, trading_days: int = 252) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / trading_days
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    vol = annualized_vol(returns, trading_days)
    if vol == 0:
        return 0.0
    return float(returns.mean() * trading_days / vol)


def sortino_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = downside.std(ddof=1) * math.sqrt(trading_days)
    if downside_vol == 0:
        return 0.0
    return float(returns.mean() * trading_days / downside_vol)


def beta_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series, trading_days: int = 252) -> tuple[float, float, float]:
    df = pd.concat([strategy_returns.rename("s"), benchmark_returns.rename("b")], axis=1).dropna()
    if len(df) < 20 or df["b"].var() == 0:
        return 0.0, 0.0, 0.0
    beta = float(df["s"].cov(df["b"]) / df["b"].var())
    alpha_daily = float(df["s"].mean() - beta * df["b"].mean())
    corr = float(df["s"].corr(df["b"]))
    return beta, alpha_daily * trading_days, corr


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0, trading_days: int = 252) -> float:
    # Bailey & Lopez de Prado style PSR approximation. This is not a substitute for proper
    # DSR when many trials are tested, but it gives a useful non-normality-aware statistic.
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return 0.0
    sr = sharpe_ratio(r, trading_days)
    skew = float(r.skew())
    kurt = float(r.kurtosis() + 3.0)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark_sr) * math.sqrt(n - 1.0) / denom
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def summarize_performance(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    equity: pd.Series,
    costs: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    trading_days: int = 252,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float | int | str | None]:
    dd = drawdown(equity)
    max_dd = float(dd.min()) if len(dd) else 0.0
    ann_return = cagr(equity, trading_days)
    ann_vol = annualized_vol(net_returns, trading_days)
    metrics: dict[str, float | int | str | None] = {
        "start": str(equity.index.min().date()) if len(equity) else None,
        "end": str(equity.index.max().date()) if len(equity) else None,
        "observations": int(len(equity)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "cagr": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe_ratio(net_returns, trading_days),
        "sortino": sortino_ratio(net_returns, trading_days),
        "probabilistic_sharpe_gt_0": probabilistic_sharpe_ratio(net_returns, 0.0, trading_days),
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0,
        "hit_rate": float((net_returns > 0).mean()) if len(net_returns) else 0.0,
        "best_day": float(net_returns.max()) if len(net_returns) else 0.0,
        "worst_day": float(net_returns.min()) if len(net_returns) else 0.0,
        "average_daily_cost": float(costs.mean()) if len(costs) else 0.0,
        "annualized_cost_drag": float(costs.mean() * trading_days) if len(costs) else 0.0,
        "average_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "average_gross_exposure": float(weights.abs().sum(axis=1).mean()) if not weights.empty else 0.0,
        "max_gross_exposure": float(weights.abs().sum(axis=1).max()) if not weights.empty else 0.0,
        "average_net_exposure": float(weights.sum(axis=1).mean()) if not weights.empty else 0.0,
        "gross_sharpe_before_costs": sharpe_ratio(gross_returns, trading_days),
        "nonzero_signal_or_weight_days": int((weights.abs().sum(axis=1) > 0).sum()) if not weights.empty else 0,
    }
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(net_returns.index).fillna(0.0)
        beta, alpha, corr = beta_alpha(net_returns, bench, trading_days)
        bench_equity = (1.0 + bench).cumprod()
        metrics.update(
            {
                "benchmark_beta": beta,
                "annualized_alpha_vs_benchmark": alpha,
                "benchmark_correlation": corr,
                "benchmark_cagr": cagr(bench_equity, trading_days),
                "benchmark_sharpe": sharpe_ratio(bench, trading_days),
                "benchmark_max_drawdown": float(drawdown(bench_equity).min()) if len(bench_equity) else 0.0,
            }
        )
    return metrics


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return pd.DataFrame()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="return").fillna(0.0)
