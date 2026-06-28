from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    return df.rolling(window=window, min_periods=max(2, window // 2)).mean()


def ema(df: pd.DataFrame | pd.Series, span: int) -> pd.DataFrame | pd.Series:
    return df.ewm(span=span, min_periods=max(2, span // 2), adjust=False).mean()


def ewma_vol(returns: pd.DataFrame | pd.Series, span: int, trading_days: int = 252) -> pd.DataFrame | pd.Series:
    return returns.ewm(span=span, min_periods=max(5, span // 3), adjust=False).std() * np.sqrt(trading_days)


def row_zscore(df: pd.DataFrame, clip: float | None = None, min_count: int = 3) -> pd.DataFrame:
    count = df.notna().sum(axis=1)
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    z = z.where(count >= min_count, 0.0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def row_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df.rank(axis=1, pct=True)
    # Map [0,1] rank to approximately [-1,1]
    return (2.0 * ranks - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sector_demean(signal: pd.DataFrame, sector_map: dict[str, str] | None) -> pd.DataFrame:
    if not sector_map:
        return signal
    out = signal.copy()
    sectors = pd.Series({c: sector_map.get(c, "Unknown") for c in signal.columns})
    for sector in sorted(sectors.unique()):
        cols = list(sectors[sectors == sector].index)
        if len(cols) >= 2:
            out[cols] = signal[cols].sub(signal[cols].mean(axis=1), axis=0)
    return out


def average_true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).stack(),
            (high - prev_close).abs().stack(),
            (low - prev_close).abs().stack(),
        ],
        axis=1,
    ).max(axis=1).unstack()
    return tr.rolling(window=window, min_periods=max(2, window // 2)).mean()


def rolling_beta(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, lookback: int) -> pd.DataFrame:
    bench = benchmark_returns.reindex(asset_returns.index).fillna(0.0)
    var_b = bench.rolling(lookback, min_periods=max(20, lookback // 3)).var().replace(0, np.nan)
    betas: dict[str, pd.Series] = {}
    for col in asset_returns.columns:
        cov = asset_returns[col].rolling(lookback, min_periods=max(20, lookback // 3)).cov(bench)
        betas[col] = (cov / var_b).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pd.DataFrame(betas, index=asset_returns.index)


def cumulative_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return (1.0 + returns).rolling(window, min_periods=max(5, window // 3)).apply(np.prod, raw=True) - 1.0
