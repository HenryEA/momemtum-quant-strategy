from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_stock_momentum.features.indicators import rolling_beta

PortfolioMode = Literal[
    "long_only",
    "long_short",
    "long_short_beta_neutral",
    "core_satellite",
    "adaptive_core_satellite",
    "adaptive_total_return",
    "adaptive_total_return_v5",
]


@dataclass(frozen=True)
class MarketRegimeConfig:
    enabled: bool = True
    benchmark: str = "SPY"
    sma: int = 200
    risk_off_gross_multiplier: float = 0.55
    high_vol_lookback: int = 63
    high_vol_threshold: float = 0.30
    high_vol_multiplier: float = 0.75
    rebound_lookback: int = 21
    panic_lookback: int = 63
    panic_drawdown_threshold: float = -0.12
    rebound_threshold: float = 0.08
    panic_rebound_short_multiplier: float = 0.25
    risk_on_gross_multiplier: float = 1.12
    risk_on_low_vol_threshold: float = 0.22
    risk_on_breadth_multiplier: float = 1.05
    min_risk_on_breadth: float = 0.55

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "MarketRegimeConfig":
        return MarketRegimeConfig(
            enabled=bool(cfg.get("enabled", True)),
            benchmark=str(cfg.get("benchmark", "SPY")).upper(),
            sma=int(cfg.get("sma", 200)),
            risk_off_gross_multiplier=float(cfg.get("risk_off_gross_multiplier", 0.55)),
            high_vol_lookback=int(cfg.get("high_vol_lookback", 63)),
            high_vol_threshold=float(cfg.get("high_vol_threshold", 0.30)),
            high_vol_multiplier=float(cfg.get("high_vol_multiplier", 0.75)),
            rebound_lookback=int(cfg.get("rebound_lookback", 21)),
            panic_lookback=int(cfg.get("panic_lookback", 63)),
            panic_drawdown_threshold=float(cfg.get("panic_drawdown_threshold", -0.12)),
            rebound_threshold=float(cfg.get("rebound_threshold", 0.08)),
            panic_rebound_short_multiplier=float(cfg.get("panic_rebound_short_multiplier", 0.25)),
            risk_on_gross_multiplier=float(cfg.get("risk_on_gross_multiplier", 1.12)),
            risk_on_low_vol_threshold=float(cfg.get("risk_on_low_vol_threshold", 0.22)),
            risk_on_breadth_multiplier=float(cfg.get("risk_on_breadth_multiplier", 1.05)),
            min_risk_on_breadth=float(cfg.get("min_risk_on_breadth", 0.55)),
        )


@dataclass(frozen=True)
class DrawdownBrakeConfig:
    enabled: bool = True
    soft_drawdown: float = -0.14
    hard_drawdown: float = -0.24
    soft_multiplier: float = 0.80
    hard_multiplier: float = 0.45
    recovery_drawdown: float = -0.05

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "DrawdownBrakeConfig":
        return DrawdownBrakeConfig(
            enabled=bool(cfg.get("enabled", True)),
            soft_drawdown=float(cfg.get("soft_drawdown", -0.14)),
            hard_drawdown=float(cfg.get("hard_drawdown", -0.24)),
            soft_multiplier=float(cfg.get("soft_multiplier", 0.80)),
            hard_multiplier=float(cfg.get("hard_multiplier", 0.45)),
            recovery_drawdown=float(cfg.get("recovery_drawdown", -0.05)),
        )


@dataclass(frozen=True)
class PortfolioConfig:
    mode: PortfolioMode = "adaptive_total_return"
    target_vol: float = 0.19
    vol_lookback: int = 63
    cov_lookback: int = 126
    covariance_shrinkage: float = 0.60
    max_gross_leverage: float = 1.35
    max_net_exposure: float = 1.20
    max_weight_per_stock: float = 0.10
    max_sector_weight: float = 0.55
    min_abs_weight: float = 0.0015
    rebalance_frequency: str = "ME"
    turnover_aversion: float = 0.10
    beta_neutral_lookback: int = 126
    # Core-satellite uses a long-only winner portfolio as the return engine and a smaller
    # residual/beta-neutral overlay for alpha diversification.
    core_weight: float = 0.55
    diversified_weight: float = 0.22
    universe_weight: float = 0.18
    satellite_weight: float = 0.05
    satellite_beta_neutral: bool = True
    adaptive_risk_budget: bool = True
    min_names_for_diversified_sleeve: int = 8
    min_names_for_universe_sleeve: int = 12
    # v5 broad sleeve controls. Equal-weight sleeves often beat fragile optimized
    # portfolios out of sample, while a modest signal tilt keeps participation
    # high without becoming a concentrated momentum book.
    universe_sleeve_method: str = "equal_weight_quality"  # inverse_vol, equal_weight, equal_weight_quality
    universe_signal_tilt: float = 0.35
    universe_max_weight_multiplier: float = 2.0
    core_signal_power: float = 1.20
    diversified_signal_power: float = 0.75
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    drawdown_brake: DrawdownBrakeConfig = DrawdownBrakeConfig()

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "PortfolioConfig":
        mode = str(cfg.get("mode", cfg.get("portfolio_mode", "adaptive_total_return"))).lower()
        if bool(cfg.get("long_only", False)):
            mode = "long_only"
        valid = {"long_only", "long_short", "long_short_beta_neutral", "core_satellite", "adaptive_core_satellite", "adaptive_total_return", "adaptive_total_return_v5"}
        if mode not in valid:
            raise ValueError(f"Unsupported portfolio.mode={mode}. Valid values are {sorted(valid)}")
        return PortfolioConfig(
            mode=mode,  # type: ignore[arg-type]
            target_vol=float(cfg.get("target_vol", 0.19)),
            vol_lookback=int(cfg.get("vol_lookback", 63)),
            cov_lookback=int(cfg.get("cov_lookback", 126)),
            covariance_shrinkage=float(cfg.get("covariance_shrinkage", 0.60)),
            max_gross_leverage=float(cfg.get("max_gross_leverage", 1.35)),
            max_net_exposure=float(cfg.get("max_net_exposure", 1.20)),
            max_weight_per_stock=float(cfg.get("max_weight_per_stock", 0.10)),
            max_sector_weight=float(cfg.get("max_sector_weight", 0.55)),
            min_abs_weight=float(cfg.get("min_abs_weight", 0.0015)),
            rebalance_frequency=str(cfg.get("rebalance_frequency", "ME")),
            turnover_aversion=float(cfg.get("turnover_aversion", 0.10)),
            beta_neutral_lookback=int(cfg.get("beta_neutral_lookback", 126)),
            core_weight=float(cfg.get("core_weight", 0.55)),
            diversified_weight=float(cfg.get("diversified_weight", 0.22)),
            universe_weight=float(cfg.get("universe_weight", 0.18)),
            satellite_weight=float(cfg.get("satellite_weight", 0.05)),
            satellite_beta_neutral=bool(cfg.get("satellite_beta_neutral", True)),
            adaptive_risk_budget=bool(cfg.get("adaptive_risk_budget", True)),
            min_names_for_diversified_sleeve=int(cfg.get("min_names_for_diversified_sleeve", 8)),
            min_names_for_universe_sleeve=int(cfg.get("min_names_for_universe_sleeve", 12)),
            universe_sleeve_method=str(cfg.get("universe_sleeve_method", "equal_weight_quality")).lower(),
            universe_signal_tilt=float(cfg.get("universe_signal_tilt", 0.35)),
            universe_max_weight_multiplier=float(cfg.get("universe_max_weight_multiplier", 2.0)),
            core_signal_power=float(cfg.get("core_signal_power", 1.20)),
            diversified_signal_power=float(cfg.get("diversified_signal_power", 0.75)),
            market_regime=MarketRegimeConfig.from_dict(cfg.get("market_regime_filter", {}) or {}),
            drawdown_brake=DrawdownBrakeConfig.from_dict(cfg.get("drawdown_brake", {}) or {}),
        )


class VolTargetStockOptimizer:
    """Volatility-targeted stock allocation with shrinkage, caps, sectors, and beta controls.

    Supported modes:
    - long_only: practical capital-growth portfolio of high-momentum stocks.
    - long_short: long winners and short losers with controlled net exposure.
    - long_short_beta_neutral: cleaner alpha test with near-zero benchmark beta.
    - core_satellite: long-only momentum core plus smaller beta-neutral long/short overlay.
    - adaptive_core_satellite: v3 blend of concentrated momentum, diversified
      positive-signal sleeve, and a small residual overlay.
    - adaptive_total_return: v4 default. Adds a broad equal-risk universe sleeve
      so the strategy can participate when the whole large-cap universe is strong,
      while still using momentum selection, residual overlay and crash controls.
    - adaptive_total_return_v5: v5 default. Uses a 1/N-inspired equal-weight/
      quality-tilted broad sleeve plus momentum and residual overlays. This follows
      the practical lesson that simple diversified sleeves can be more robust than
      noisy optimized weights, while still harvesting momentum alpha.
    """

    def __init__(self, cfg: PortfolioConfig, trading_days: int = 252, sector_map: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.trading_days = trading_days
        self.sector_map = sector_map or {}

    def allocate(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        idx = signals.index
        cols = list(signals.columns)
        weights = pd.DataFrame(0.0, index=idx, columns=cols)
        rebalance_dates = self._rebalance_dates(idx)
        prev = pd.Series(0.0, index=cols)

        for dt in idx:
            if dt not in rebalance_dates:
                weights.loc[dt] = prev
                continue
            hist = returns.loc[:dt].tail(self.cfg.cov_lookback).reindex(columns=cols)
            sig = signals.loc[dt].reindex(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            bench_hist = benchmark_returns.loc[:dt].tail(self.cfg.beta_neutral_lookback) if benchmark_returns is not None else None
            target = self._target_for_date(hist, sig, bench_hist)
            if self.cfg.turnover_aversion > 0:
                target = prev * self.cfg.turnover_aversion + target * (1.0 - self.cfg.turnover_aversion)
            target = self._final_clean(target)
            weights.loc[dt] = target
            prev = target
        return weights.fillna(0.0)

    def _target_for_date(
        self,
        hist_returns: pd.DataFrame,
        signal: pd.Series,
        bench_hist_returns: pd.Series | None,
    ) -> pd.Series:
        cols = signal.index
        if signal.abs().sum() == 0:
            return pd.Series(0.0, index=cols)
        realized_vol = hist_returns.ewm(span=self.cfg.vol_lookback, min_periods=max(5, self.cfg.vol_lookback // 3)).std().iloc[-1]
        realized_vol = realized_vol.replace(0, np.nan) * np.sqrt(self.trading_days)
        raw = signal / realized_vol.reindex(cols)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        betas = None
        if bench_hist_returns is not None and len(bench_hist_returns) >= 20:
            stock_hist = hist_returns.reindex(bench_hist_returns.index).fillna(0.0)
            betas = rolling_beta(stock_hist, bench_hist_returns.fillna(0.0), min(self.cfg.beta_neutral_lookback, len(stock_hist))).iloc[-1]
            betas = betas.reindex(cols).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        if self.cfg.mode == "long_only":
            w = self._normalize_gross(raw.clip(lower=0.0), target_gross=1.0)
        elif self.cfg.mode in {"core_satellite", "adaptive_core_satellite", "adaptive_total_return", "adaptive_total_return_v5"}:
            w = self._core_satellite_weights(raw, betas, hist_returns)
        else:
            w = self._long_short_normalize(raw)
            if self.cfg.mode == "long_short_beta_neutral" and betas is not None:
                w = self._beta_neutralize(w, betas)
                w = self._long_short_normalize(w)

        w = self._risk_scale_and_cap(w, hist_returns.reindex(columns=cols))
        return w

    def _core_satellite_weights(self, raw: pd.Series, betas: pd.Series | None, hist_returns: pd.DataFrame) -> pd.Series:
        if self.cfg.mode == "core_satellite":
            total = max(1e-12, self.cfg.core_weight + self.cfg.satellite_weight)
            core_share = self.cfg.core_weight / total
            diversified_share = 0.0
            universe_share = 0.0
            satellite_share = self.cfg.satellite_weight / total
        elif self.cfg.mode == "adaptive_core_satellite":
            total = max(1e-12, self.cfg.core_weight + self.cfg.diversified_weight + self.cfg.satellite_weight)
            core_share = self.cfg.core_weight / total
            diversified_share = self.cfg.diversified_weight / total
            universe_share = 0.0
            satellite_share = self.cfg.satellite_weight / total
        else:
            total = max(
                1e-12,
                self.cfg.core_weight + self.cfg.diversified_weight + self.cfg.universe_weight + self.cfg.satellite_weight,
            )
            core_share = self.cfg.core_weight / total
            diversified_share = self.cfg.diversified_weight / total
            universe_share = self.cfg.universe_weight / total
            satellite_share = self.cfg.satellite_weight / total

            # v4/v5: dynamic breadth-aware budget. If a broad fraction of the
            # universe has positive scores, lean more into the broad sleeve; if
            # breadth is poor, reduce market-like exposure and let the concentrated
            # momentum core dominate. v5 is more participation-oriented because
            # long-run equity returns are positive and equal-weight baskets can be
            # hard to beat with noisy expected-return estimates.
            positive_breadth = float((raw > 0).mean()) if len(raw) else 0.0
            if self.cfg.mode == "adaptive_total_return_v5":
                if positive_breadth >= 0.55:
                    universe_share *= 1.35
                    diversified_share *= 1.10
                    satellite_share *= 0.55
                elif positive_breadth <= 0.30:
                    universe_share *= 0.45
                    core_share *= 1.18
                else:
                    universe_share *= 1.05
            else:
                if positive_breadth >= 0.55:
                    universe_share *= 1.20
                    satellite_share *= 0.75
                elif positive_breadth <= 0.30:
                    universe_share *= 0.55
                    core_share *= 1.08
            norm = max(1e-12, core_share + diversified_share + universe_share + satellite_share)
            core_share /= norm
            diversified_share /= norm
            universe_share /= norm
            satellite_share /= norm

        core_raw = raw.clip(lower=0.0)
        if self.cfg.core_signal_power != 1.0 and core_raw.abs().sum() > 0:
            core_raw = np.sign(core_raw) * core_raw.abs().pow(self.cfg.core_signal_power)
        core = self._normalize_gross(core_raw, target_gross=core_share)
        diversified = self._diversified_positive_sleeve(raw, hist_returns, target_gross=diversified_share)
        universe = self._universe_broad_sleeve(raw, hist_returns, target_gross=universe_share)
        satellite = self._long_short_normalize(raw) * satellite_share
        if self.cfg.satellite_beta_neutral and betas is not None:
            satellite = self._beta_neutralize(satellite, betas)
            gross = satellite.abs().sum()
            if gross > 0:
                satellite = satellite / gross * satellite_share
        return core + diversified + universe + satellite

    def _diversified_positive_sleeve(self, raw: pd.Series, hist_returns: pd.DataFrame, target_gross: float) -> pd.Series:
        if target_gross <= 0:
            return pd.Series(0.0, index=raw.index)
        positive = raw[raw > 0].index
        if len(positive) < self.cfg.min_names_for_diversified_sleeve:
            return pd.Series(0.0, index=raw.index)
        vol = hist_returns[positive].ewm(span=self.cfg.vol_lookback, min_periods=max(5, self.cfg.vol_lookback // 3)).std().iloc[-1]
        inv_vol = 1.0 / (vol.replace(0, np.nan) * np.sqrt(self.trading_days))
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Softly tilt the broad sleeve toward stronger signals but keep it diversified.
        strength = raw[positive].rank(pct=True).clip(0.25, 1.0)
        if self.cfg.diversified_signal_power != 1.0:
            strength = strength.pow(self.cfg.diversified_signal_power)
        sleeve = inv_vol * strength
        out = pd.Series(0.0, index=raw.index)
        out.loc[positive] = self._normalize_gross(sleeve, target_gross=target_gross)
        return out

    def _universe_broad_sleeve(self, raw: pd.Series, hist_returns: pd.DataFrame, target_gross: float) -> pd.Series:
        """Broad participation sleeve used by v5.

        Academic and practitioner studies repeatedly show that simple 1/N or
        equal-risk benchmarks are difficult to beat out of sample. v5 therefore
        supports a plain equal-weight sleeve and a mild quality/momentum-tilted
        equal-weight sleeve. This preserves broad-market participation while using
        signals only as a modest tilt, not as a fragile optimizer input.
        """
        cols = list(hist_returns.columns)
        out = pd.Series(0.0, index=cols)
        if target_gross <= 0 or hist_returns.empty:
            return out
        recent = hist_returns.tail(min(len(hist_returns), self.cfg.vol_lookback))
        valid = recent.notna().sum(axis=0) >= max(10, min(len(recent), self.cfg.vol_lookback) // 2)
        valid_cols = list(valid[valid].index)
        if len(valid_cols) < self.cfg.min_names_for_universe_sleeve:
            return out

        method = self.cfg.universe_sleeve_method
        if method == "equal_weight":
            base = pd.Series(1.0, index=valid_cols)
        elif method == "equal_weight_quality":
            # A low-dimensional tilt: each stock starts with a 1/N allocation,
            # then receives a bounded boost/penalty based on signal rank. This
            # often keeps the benefits of equal-weighting while leaning away from
            # the weakest trends.
            ranks = raw.reindex(valid_cols).rank(pct=True).fillna(0.5)
            tilt = 1.0 + self.cfg.universe_signal_tilt * (ranks - 0.5) * 2.0
            tilt = tilt.clip(1.0 / max(1.0, self.cfg.universe_max_weight_multiplier), self.cfg.universe_max_weight_multiplier)
            base = tilt
        else:
            vol = recent[valid_cols].std() * np.sqrt(self.trading_days)
            base = 1.0 / vol.replace(0, np.nan)
            base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sleeve = self._normalize_gross(base, target_gross=target_gross)
        out.loc[valid_cols] = sleeve
        return out

    def _risk_scale_and_cap(self, w: pd.Series, hist_returns: pd.DataFrame) -> pd.Series:
        w = self._cap_single_names(w)
        w = self._apply_sector_cap(w)
        w = self._control_net_exposure(w)

        cov = self._annualized_cov(hist_returns.reindex(columns=w.index))
        port_vol = self._portfolio_vol(w, cov)
        if np.isfinite(port_vol) and port_vol > 0:
            target_vol = self._adaptive_target_vol(hist_returns)
            scale = target_vol / port_vol
        else:
            scale = 1.0
        w = w * min(scale, self.cfg.max_gross_leverage / max(w.abs().sum(), 1e-12))
        w = self._cap_single_names(w)
        w = self._apply_sector_cap(w)
        w = self._control_net_exposure(w)
        return w

    def _adaptive_target_vol(self, hist_returns: pd.DataFrame) -> float:
        if not self.cfg.adaptive_risk_budget or hist_returns.empty:
            return self.cfg.target_vol
        # Cross-sectional breadth and realized opportunity set determine whether
        # we lean toward full risk budget or keep the vol target conservative.
        last_63 = hist_returns.tail(min(63, len(hist_returns)))
        breadth = float((last_63.sum(axis=0) > 0).mean()) if last_63.shape[1] else 0.5
        median_vol = float((last_63.std() * np.sqrt(self.trading_days)).median()) if len(last_63) >= 10 else np.nan
        multiplier = 1.0
        if breadth > 0.60 and (not np.isfinite(median_vol) or median_vol < 0.45):
            multiplier *= 1.08
        if breadth < 0.40:
            multiplier *= 0.85
        return float(np.clip(self.cfg.target_vol * multiplier, 0.08, self.cfg.target_vol * 1.15))

    def _normalize_gross(self, w: pd.Series, target_gross: float) -> pd.Series:
        gross = w.abs().sum()
        if gross <= 0:
            return w * 0.0
        return w / gross * target_gross

    def _long_short_normalize(self, w: pd.Series) -> pd.Series:
        w = w.copy()
        if w.abs().sum() <= 0:
            return w * 0.0
        longs = w.clip(lower=0.0)
        shorts = w.clip(upper=0.0)
        long_sum = longs.sum()
        short_sum = shorts.abs().sum()
        if long_sum > 0:
            longs = longs / long_sum * 0.5
        if short_sum > 0:
            shorts = shorts / short_sum * 0.5
        return longs + shorts

    def _beta_neutralize(self, w: pd.Series, beta: pd.Series) -> pd.Series:
        beta = beta.reindex(w.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        denom = float((beta * beta).sum())
        if denom <= 1e-12:
            return w
        beta_exp = float((w * beta).sum())
        return w - beta_exp * beta / denom

    def _annualized_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.cov(min_periods=max(5, min(20, len(returns) // 2))).fillna(0.0)
        if cov.empty:
            return cov
        diag = np.diag(np.diag(cov.to_numpy()))
        shrunk = (1.0 - self.cfg.covariance_shrinkage) * cov.to_numpy() + self.cfg.covariance_shrinkage * diag
        return pd.DataFrame(shrunk * self.trading_days, index=cov.index, columns=cov.columns)

    def _portfolio_vol(self, w: pd.Series, cov: pd.DataFrame) -> float:
        if cov.empty:
            return np.nan
        wv = w.reindex(cov.index).fillna(0.0).to_numpy()
        var = float(wv.T @ cov.to_numpy() @ wv)
        return float(np.sqrt(max(var, 0.0)))

    def _cap_single_names(self, w: pd.Series) -> pd.Series:
        cap = self.cfg.max_weight_per_stock
        if cap <= 0:
            return w * 0
        w = w.clip(-cap, cap)
        gross = w.abs().sum()
        if gross > self.cfg.max_gross_leverage:
            w = w * (self.cfg.max_gross_leverage / gross)
        return w

    def _apply_sector_cap(self, w: pd.Series) -> pd.Series:
        cap = self.cfg.max_sector_weight
        if cap <= 0 or not self.sector_map:
            return w
        out = w.copy()
        sectors = pd.Series({c: self.sector_map.get(c, "Unknown") for c in out.index})
        for sector in sectors.unique():
            cols = list(sectors[sectors == sector].index)
            gross = out[cols].abs().sum()
            if gross > cap:
                out[cols] *= cap / gross
        return out

    def _control_net_exposure(self, w: pd.Series) -> pd.Series:
        if self.cfg.mode == "long_only":
            return w.clip(lower=0.0)
        net = float(w.sum())
        if abs(net) <= self.cfg.max_net_exposure:
            return w
        adjustment = (abs(net) - self.cfg.max_net_exposure) * np.sign(net)
        candidates = w[w * np.sign(net) > 0]
        if candidates.empty:
            return w
        reduction = candidates / candidates.sum() * adjustment
        w.loc[candidates.index] = w.loc[candidates.index] - reduction
        return w

    def _final_clean(self, w: pd.Series) -> pd.Series:
        w = w.where(w.abs() >= self.cfg.min_abs_weight, 0.0)
        gross = w.abs().sum()
        if gross > self.cfg.max_gross_leverage:
            w = w * (self.cfg.max_gross_leverage / gross)
        return w.fillna(0.0)

    def _rebalance_dates(self, idx: pd.DatetimeIndex) -> set[pd.Timestamp]:
        freq = self.cfg.rebalance_frequency.lower()
        if freq in {"daily", "d", "1d"}:
            return set(idx)
        s = pd.Series(index=idx, data=np.arange(len(idx)))
        labels = s.resample(self.cfg.rebalance_frequency).last().dropna().astype(int)
        return set(idx[labels.to_numpy()])
