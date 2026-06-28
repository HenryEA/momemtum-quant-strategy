from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_stock_momentum.data.loader import MarketBars
from quant_stock_momentum.features.indicators import (
    cumulative_return,
    ewma_vol,
    rolling_beta,
    row_rank_score,
    row_zscore,
    sector_demean,
    sma,
)


@dataclass(frozen=True)
class SignalComponentWeights:
    """Weights for the v4 ensemble momentum signal.

    v4 keeps the robust v3 momentum stack and adds three practitioner-oriented
    features that are common in systematic equity momentum research notes:

    - 52-week-high proximity: price strength near prior highs.
    - smooth/FIP momentum: gradual, persistent advances are rewarded more than
      jumpy one-day moves with the same cumulative return.
    - volatility-contraction and drawdown quality: favor trend strength with less
      path damage and less short-term volatility expansion.

    All components are cross-sectionally standardized each day before blending.
    This keeps any single indicator from dominating the portfolio.
    """

    momentum_12_1: float = 0.17
    intermediate_12_7: float = 0.14
    momentum_9_1: float = 0.08
    momentum_6_1: float = 0.09
    residual_12_1: float = 0.16
    relative_strength_12_1: float = 0.10
    information_ratio_6_1: float = 0.08
    time_series_3m: float = 0.04
    trend_strength_200: float = 0.04
    breakout_252: float = 0.04
    fip_smooth_momentum: float = 0.02
    volatility_contraction: float = 0.01
    drawdown_quality: float = 0.01
    downside_quality: float = 0.01
    low_volatility_quality: float = 0.005
    return_consistency: float = 0.010
    efficiency_ratio_6m: float = 0.020
    new_high_persistence: float = 0.020
    volatility_adjusted_12_1: float = 0.030
    short_reversal_1m: float = 0.005

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "SignalComponentWeights":
        return SignalComponentWeights(
            momentum_12_1=float(cfg.get("momentum_12_1", 0.17)),
            intermediate_12_7=float(cfg.get("intermediate_12_7", 0.14)),
            momentum_9_1=float(cfg.get("momentum_9_1", 0.08)),
            momentum_6_1=float(cfg.get("momentum_6_1", 0.09)),
            residual_12_1=float(cfg.get("residual_12_1", 0.16)),
            relative_strength_12_1=float(cfg.get("relative_strength_12_1", 0.10)),
            information_ratio_6_1=float(cfg.get("information_ratio_6_1", 0.08)),
            time_series_3m=float(cfg.get("time_series_3m", 0.04)),
            trend_strength_200=float(cfg.get("trend_strength_200", 0.04)),
            breakout_252=float(cfg.get("breakout_252", 0.04)),
            fip_smooth_momentum=float(cfg.get("fip_smooth_momentum", 0.02)),
            volatility_contraction=float(cfg.get("volatility_contraction", 0.01)),
            drawdown_quality=float(cfg.get("drawdown_quality", 0.01)),
            downside_quality=float(cfg.get("downside_quality", 0.01)),
            low_volatility_quality=float(cfg.get("low_volatility_quality", 0.005)),
            return_consistency=float(cfg.get("return_consistency", 0.010)),
            efficiency_ratio_6m=float(cfg.get("efficiency_ratio_6m", 0.020)),
            new_high_persistence=float(cfg.get("new_high_persistence", 0.020)),
            volatility_adjusted_12_1=float(cfg.get("volatility_adjusted_12_1", 0.030)),
            short_reversal_1m=float(cfg.get("short_reversal_1m", 0.005)),
        )

    def normalized(self) -> "SignalComponentWeights":
        total = sum(abs(x) for x in self.__dict__.values())
        if total <= 0:
            return self
        return SignalComponentWeights(**{k: float(v) / total for k, v in self.__dict__.items()})


@dataclass(frozen=True)
class StockMomentumSignalConfig:
    formation_lookback: int = 252
    skip_recent_days: int = 21
    intermediate_start: int = 252
    intermediate_end: int = 126
    nine_month_lookback: int = 189
    six_month_lookback: int = 126
    short_lookback: int = 63
    reversal_lookback: int = 21
    residual_beta_lookback: int = 252
    fip_lookback: int = 252
    drawdown_lookback: int = 126
    volatility_contraction_short: int = 63
    volatility_contraction_long: int = 252
    efficiency_lookback: int = 126
    new_high_persistence_lookback: int = 126
    benchmark: str = "SPY"
    component_weights: SignalComponentWeights = SignalComponentWeights()
    long_short: bool = True
    top_quantile: float = 0.45
    bottom_quantile: float = 0.10
    score_method: str = "zscore"  # zscore or rank
    zscore_clip: float = 3.0
    signal_clip: float = 1.0
    sector_neutralize: bool = False
    sector_neutralize_long_only: bool = False
    cross_sectional_demean: bool = False
    trend_filter_enabled: bool = True
    trend_fast_ma: int = 50
    trend_slow_ma: int = 200
    trend_disagreement_multiplier: float = 0.35
    trend_agreement_multiplier: float = 1.12
    liquidity_filter_enabled: bool = True
    min_price: float = 5.0
    volume_lookback: int = 20
    min_average_volume: float = 100000.0
    allow_zero_volume_for_large_caps: bool = True
    max_missing_fraction: float = 0.10

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "StockMomentumSignalConfig":
        trend = cfg.get("trend_filter", {}) or {}
        liq = cfg.get("liquidity_filter", {}) or {}
        comps = cfg.get("component_weights", {}) or {}
        return StockMomentumSignalConfig(
            formation_lookback=int(cfg.get("formation_lookback", 252)),
            skip_recent_days=int(cfg.get("skip_recent_days", 21)),
            intermediate_start=int(cfg.get("intermediate_start", 252)),
            intermediate_end=int(cfg.get("intermediate_end", 126)),
            nine_month_lookback=int(cfg.get("nine_month_lookback", 189)),
            six_month_lookback=int(cfg.get("six_month_lookback", 126)),
            short_lookback=int(cfg.get("short_lookback", 63)),
            reversal_lookback=int(cfg.get("reversal_lookback", 21)),
            residual_beta_lookback=int(cfg.get("residual_beta_lookback", 252)),
            fip_lookback=int(cfg.get("fip_lookback", 252)),
            drawdown_lookback=int(cfg.get("drawdown_lookback", 126)),
            volatility_contraction_short=int(cfg.get("volatility_contraction_short", 63)),
            volatility_contraction_long=int(cfg.get("volatility_contraction_long", 252)),
            efficiency_lookback=int(cfg.get("efficiency_lookback", 126)),
            new_high_persistence_lookback=int(cfg.get("new_high_persistence_lookback", 126)),
            benchmark=str(cfg.get("benchmark", "SPY")).upper(),
            component_weights=SignalComponentWeights.from_dict(comps).normalized(),
            long_short=bool(cfg.get("long_short", True)),
            top_quantile=float(cfg.get("top_quantile", 0.45)),
            bottom_quantile=float(cfg.get("bottom_quantile", 0.10)),
            score_method=str(cfg.get("score_method", "zscore")).lower(),
            zscore_clip=float(cfg.get("zscore_clip", 3.0)),
            signal_clip=float(cfg.get("signal_clip", 1.0)),
            sector_neutralize=bool(cfg.get("sector_neutralize", False)),
            sector_neutralize_long_only=bool(cfg.get("sector_neutralize_long_only", False)),
            cross_sectional_demean=bool(cfg.get("cross_sectional_demean", False)),
            trend_filter_enabled=bool(trend.get("enabled", True)),
            trend_fast_ma=int(trend.get("fast_ma", 50)),
            trend_slow_ma=int(trend.get("slow_ma", 200)),
            trend_disagreement_multiplier=float(trend.get("disagreement_multiplier", 0.35)),
            trend_agreement_multiplier=float(trend.get("agreement_multiplier", 1.12)),
            liquidity_filter_enabled=bool(liq.get("enabled", True)),
            min_price=float(liq.get("min_price", 5.0)),
            volume_lookback=int(liq.get("volume_lookback", 20)),
            min_average_volume=float(liq.get("min_average_volume", 100000.0)),
            allow_zero_volume_for_large_caps=bool(liq.get("allow_zero_volume_for_large_caps", True)),
            max_missing_fraction=float(cfg.get("max_missing_fraction", 0.10)),
        )


class StockMomentumStrategy:
    """Composite stock momentum signal with residual momentum, quality-of-trend filters and liquidity gates."""

    def __init__(self, cfg: StockMomentumSignalConfig, sector_map: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.sector_map = sector_map or {}

    def generate(self, bars: MarketBars, trading_days: int = 252) -> pd.DataFrame:
        close_all = bars.close.astype(float)
        volume_all = bars.volume.astype(float)
        benchmark = self.cfg.benchmark
        if benchmark in close_all.columns:
            benchmark_close = close_all[benchmark]
            trade_cols = [c for c in close_all.columns if c != benchmark]
        else:
            benchmark_close = close_all.mean(axis=1)
            trade_cols = list(close_all.columns)

        close = close_all[trade_cols]
        volume = volume_all[trade_cols]
        returns = close.pct_change(fill_method=None)
        benchmark_returns = benchmark_close.pct_change(fill_method=None).reindex(close.index).fillna(0.0)

        raw_components = self._raw_components(close, returns, benchmark_returns, trading_days)
        scored = {name: self._score(values) for name, values in raw_components.items()}
        w = self.cfg.component_weights
        signal = (
            w.momentum_12_1 * scored["momentum_12_1"]
            + w.intermediate_12_7 * scored["intermediate_12_7"]
            + w.momentum_9_1 * scored["momentum_9_1"]
            + w.momentum_6_1 * scored["momentum_6_1"]
            + w.residual_12_1 * scored["residual_12_1"]
            + w.relative_strength_12_1 * scored["relative_strength_12_1"]
            + w.information_ratio_6_1 * scored["information_ratio_6_1"]
            + w.time_series_3m * scored["time_series_3m"]
            + w.trend_strength_200 * scored["trend_strength_200"]
            + w.breakout_252 * scored["breakout_252"]
            + w.fip_smooth_momentum * scored["fip_smooth_momentum"]
            + w.volatility_contraction * scored["volatility_contraction"]
            + w.drawdown_quality * scored["drawdown_quality"]
            + w.downside_quality * scored["downside_quality"]
            + w.low_volatility_quality * scored["low_volatility_quality"]
            + w.return_consistency * scored["return_consistency"]
            + w.efficiency_ratio_6m * scored["efficiency_ratio_6m"]
            + w.new_high_persistence * scored["new_high_persistence"]
            + w.volatility_adjusted_12_1 * scored["volatility_adjusted_12_1"]
            + w.short_reversal_1m * scored["short_reversal_1m"]
        )

        if self.cfg.cross_sectional_demean:
            signal = signal.sub(signal.mean(axis=1), axis=0)

        if self.cfg.sector_neutralize and (self.cfg.long_short or self.cfg.sector_neutralize_long_only):
            signal = sector_demean(signal, self.sector_map)

        signal = self._keep_top_bottom(signal)
        signal = self._apply_trend_filter(signal, close)
        signal = self._apply_liquidity_filter(signal, close, volume)

        if not self.cfg.long_short:
            signal = signal.clip(lower=0.0)

        return signal.clip(-self.cfg.signal_clip, self.cfg.signal_clip).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _raw_components(
        self,
        close: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        trading_days: int,
    ) -> dict[str, pd.DataFrame]:
        c = self.cfg
        formation = close.shift(c.skip_recent_days) / close.shift(c.formation_lookback) - 1.0
        nine_one = close.shift(c.skip_recent_days) / close.shift(c.nine_month_lookback) - 1.0
        six_one = close.shift(c.skip_recent_days) / close.shift(c.six_month_lookback) - 1.0
        intermediate = close.shift(c.intermediate_end) / close.shift(c.intermediate_start) - 1.0

        betas = rolling_beta(returns.fillna(0.0), benchmark_returns, c.residual_beta_lookback)
        residual_returns = returns.sub(betas.mul(benchmark_returns, axis=0), fill_value=0.0)
        residual_mom = cumulative_return(residual_returns.shift(c.skip_recent_days), c.formation_lookback - c.skip_recent_days)

        bench_mom = cumulative_return(benchmark_returns.to_frame("benchmark").shift(c.skip_recent_days), c.formation_lookback - c.skip_recent_days)["benchmark"]
        relative_strength = formation.sub(bench_mom, axis=0)

        ann_vol_short = ewma_vol(returns, span=max(21, c.six_month_lookback // 2), trading_days=trading_days)
        daily_vol_short = ann_vol_short / np.sqrt(trading_days)
        horizon_vol = daily_vol_short * np.sqrt(max(1, c.six_month_lookback - c.skip_recent_days))
        information_ratio = six_one / horizon_vol.replace(0, np.nan)

        ann_vol = ewma_vol(returns, span=max(21, c.short_lookback), trading_days=trading_days)
        short_momentum = close / close.shift(c.short_lookback) - 1.0
        expected_move = ann_vol / np.sqrt(trading_days) * np.sqrt(c.short_lookback)
        ts = np.tanh((short_momentum / expected_move.replace(0, np.nan)) / 2.0)

        rolling_high = close.rolling(c.formation_lookback, min_periods=max(20, c.formation_lookback // 3)).max()
        breakout = close / rolling_high - 1.0

        slow = sma(close, 200)
        trend_strength = (close / slow - 1.0) / (ann_vol.replace(0, np.nan) / np.sqrt(trading_days))
        trend_strength = trend_strength.replace([np.inf, -np.inf], np.nan).clip(-10, 10)

        # Frog-in-the-pan / smooth-momentum proxy. Gradual trend persistence is
        # rewarded; large jumpy moves with the same total return receive less score.
        signed_days = np.sign(returns)
        positive_fraction = (signed_days > 0).rolling(c.fip_lookback, min_periods=max(40, c.fip_lookback // 3)).mean()
        negative_fraction = (signed_days < 0).rolling(c.fip_lookback, min_periods=max(40, c.fip_lookback // 3)).mean()
        smoothness = (positive_fraction - negative_fraction).abs()
        max_abs_day = returns.abs().rolling(c.fip_lookback, min_periods=max(40, c.fip_lookback // 3)).max()
        fip = formation * (0.50 + smoothness) - max_abs_day

        short_vol = returns.rolling(c.volatility_contraction_short, min_periods=20).std() * np.sqrt(trading_days)
        long_vol = returns.rolling(c.volatility_contraction_long, min_periods=60).std() * np.sqrt(trading_days)
        volatility_contraction = (long_vol - short_vol) / long_vol.replace(0, np.nan)

        drawdown_quality = -_rolling_max_drawdown(close, c.drawdown_lookback)

        downside = returns.where(returns < 0.0, 0.0).rolling(126, min_periods=30).std() * np.sqrt(trading_days)
        downside_quality = -downside
        low_vol_quality = -ann_vol
        mean_126 = returns.rolling(126, min_periods=30).mean()
        std_126 = returns.rolling(126, min_periods=30).std()
        return_consistency = mean_126 / std_126.replace(0, np.nan)

        # Kaufman-style efficiency ratio: persistent directional trends get high
        # scores, choppy paths with the same endpoint get lower scores.
        eff_ret = close / close.shift(c.efficiency_lookback) - 1.0
        path_length = returns.abs().rolling(c.efficiency_lookback, min_periods=max(20, c.efficiency_lookback // 3)).sum()
        efficiency_ratio = eff_ret / path_length.replace(0, np.nan)
        efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).clip(-5, 5)

        # Persistence near the 52-week high. George-Hwang-style high proximity is
        # useful, but persistence avoids overrewarding a single one-day spike.
        near_high = (close >= rolling_high * 0.95).astype(float)
        new_high_persistence = near_high.rolling(
            c.new_high_persistence_lookback,
            min_periods=max(20, c.new_high_persistence_lookback // 3),
        ).mean()

        horizon_vol_12 = daily_vol_short * np.sqrt(max(1, c.formation_lookback - c.skip_recent_days))
        volatility_adjusted_12_1 = formation / horizon_vol_12.replace(0, np.nan)

        reversal = -(close / close.shift(c.reversal_lookback) - 1.0)

        return {
            "momentum_12_1": formation,
            "intermediate_12_7": intermediate,
            "momentum_9_1": nine_one,
            "momentum_6_1": six_one,
            "residual_12_1": residual_mom,
            "relative_strength_12_1": relative_strength,
            "information_ratio_6_1": information_ratio,
            "time_series_3m": ts,
            "trend_strength_200": trend_strength,
            "breakout_252": breakout,
            "fip_smooth_momentum": fip,
            "volatility_contraction": volatility_contraction,
            "drawdown_quality": drawdown_quality,
            "downside_quality": downside_quality,
            "low_volatility_quality": low_vol_quality,
            "return_consistency": return_consistency,
            "efficiency_ratio_6m": efficiency_ratio,
            "new_high_persistence": new_high_persistence,
            "volatility_adjusted_12_1": volatility_adjusted_12_1,
            "short_reversal_1m": reversal,
        }

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.score_method == "rank":
            return row_rank_score(df)
        return row_zscore(df, clip=self.cfg.zscore_clip)

    def _apply_trend_filter(self, signal: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.trend_filter_enabled:
            return signal
        fast = sma(close, self.cfg.trend_fast_ma)
        slow = sma(close, self.cfg.trend_slow_ma)
        trend = np.sign(fast - slow)
        disagreement = (np.sign(signal) != trend) & trend.notna() & (trend != 0)
        agreement = (np.sign(signal) == trend) & trend.notna() & (trend != 0)
        out = signal.mask(disagreement, signal * self.cfg.trend_disagreement_multiplier)
        out = out.mask(agreement, out * self.cfg.trend_agreement_multiplier)
        return out

    def _apply_liquidity_filter(self, signal: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.liquidity_filter_enabled:
            return signal
        avg_volume = volume.rolling(self.cfg.volume_lookback, min_periods=max(2, self.cfg.volume_lookback // 2)).mean()
        liquid = close >= self.cfg.min_price
        has_real_volume = avg_volume > 0
        if self.cfg.allow_zero_volume_for_large_caps:
            symbol_has_any_volume = (volume > 0).any(axis=0)
            for col in close.columns:
                if symbol_has_any_volume.get(col, False):
                    liquid[col] = liquid[col] & (avg_volume[col] >= self.cfg.min_average_volume)
        else:
            liquid = liquid & (avg_volume >= self.cfg.min_average_volume) & has_real_volume
        return signal.where(liquid, 0.0)

    def _keep_top_bottom(self, signal: pd.DataFrame) -> pd.DataFrame:
        out = signal.copy() * 0.0
        if signal.shape[1] <= 1:
            return signal.copy()
        for dt, row in signal.iterrows():
            valid = row.dropna()
            valid = valid[valid != 0]
            if valid.empty:
                continue
            if self.cfg.long_short:
                long_cut = valid.quantile(1.0 - self.cfg.top_quantile)
                short_cut = valid.quantile(self.cfg.bottom_quantile)
                selected = valid[(valid >= long_cut) | (valid <= short_cut)]
            else:
                long_cut = valid.quantile(1.0 - self.cfg.top_quantile)
                selected = valid[valid >= long_cut]
            out.loc[dt, selected.index] = selected
        return out


def _rolling_max_drawdown(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling maximum drawdown over a trailing window, returned as negative values."""
    rolling_peak = close.rolling(window, min_periods=max(20, window // 3)).max()
    dd = close / rolling_peak - 1.0
    return dd.rolling(window, min_periods=max(20, window // 3)).min()
