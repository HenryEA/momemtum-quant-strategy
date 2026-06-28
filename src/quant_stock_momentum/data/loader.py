from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_stock_momentum.config import StockInstrument


@dataclass(frozen=True)
class MarketBars:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def slice_dates(self, start: str | None = None, end: str | None = None) -> "MarketBars":
        def _slice(df: pd.DataFrame) -> pd.DataFrame:
            out = df
            if start:
                out = out.loc[pd.Timestamp(start, tz="UTC") :]
            if end:
                out = out.loc[: pd.Timestamp(end, tz="UTC")]
            return out

        return MarketBars(
            open=_slice(self.open),
            high=_slice(self.high),
            low=_slice(self.low),
            close=_slice(self.close),
            volume=_slice(self.volume),
        )

    def drop_columns(self, columns: list[str]) -> "MarketBars":
        keep = [c for c in self.close.columns if c not in set(columns)]
        return MarketBars(
            open=self.open[keep], high=self.high[keep], low=self.low[keep], close=self.close[keep], volume=self.volume[keep]
        )

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


def load_market_bars(
    data_dir: str | Path,
    instruments: list[StockInstrument],
    use_adjusted_close: bool = True,
    allow_missing: bool = False,
    min_rows: int = 252,
) -> MarketBars:
    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for inst in instruments:
        try:
            path = _find_symbol_file(data_dir, inst)
            df = read_symbol_csv(path, inst.ticker, use_adjusted_close=use_adjusted_close)
            if len(df) < min_rows:
                raise ValueError(f"{path} has only {len(df)} valid rows; min_rows={min_rows}")
            frames[inst.ticker] = df
        except Exception as exc:
            msg = f"{inst.ticker}: {exc}"
            if allow_missing:
                print(f"WARNING: skipping missing/unusable symbol -> {msg}")
                errors.append(msg)
                continue
            raise

    if not frames:
        raise ValueError("No stock data loaded. Check --data-dir, CSV files, and --allow-missing.")

    all_index = sorted(set().union(*[set(df.index) for df in frames.values()]))
    index = pd.DatetimeIndex(all_index, name="timestamp")

    def wide(field: str) -> pd.DataFrame:
        cols = []
        for ticker, df in frames.items():
            cols.append(df[field].reindex(index).rename(ticker))
        out = pd.concat(cols, axis=1).sort_index()
        if field != "volume":
            out = out.ffill()
        return out

    bars = MarketBars(
        open=wide("open"),
        high=wide("high"),
        low=wide("low"),
        close=wide("close"),
        volume=wide("volume").fillna(0.0),
    )
    valid = bars.close.dropna(how="all").index
    # Remove full non-trading flat rows that appear in some CFD exports.
    close = bars.close.loc[valid]
    non_flat = close.pct_change(fill_method=None).abs().sum(axis=1).fillna(1.0) > 0
    valid = close.loc[non_flat].index
    return MarketBars(
        open=bars.open.loc[valid],
        high=bars.high.loc[valid],
        low=bars.low.loc[valid],
        close=bars.close.loc[valid],
        volume=bars.volume.loc[valid],
    )


def _find_symbol_file(data_dir: Path, inst: StockInstrument) -> Path:
    candidates = [
        data_dir / f"{inst.ticker}.csv",
        data_dir / f"{inst.ticker.upper()}.csv",
        data_dir / f"{inst.ticker.lower()}.csv",
    ]
    if inst.dukascopy_symbol:
        candidates.extend(
            [
                data_dir / f"{inst.dukascopy_symbol}.csv",
                data_dir / f"{inst.dukascopy_symbol.lower()}.csv",
                data_dir / f"{inst.dukascopy_symbol.upper()}.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    looked = ", ".join(str(p.name) for p in candidates)
    raise FileNotFoundError(f"Could not find CSV for {inst.ticker} in {data_dir}. Looked for: {looked}")


def read_symbol_csv(path: str | Path, ticker: str, use_adjusted_close: bool = True) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    df = normalize_ohlcv_columns(df)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} must contain a timestamp/date column")
    df["timestamp"] = parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            else:
                raise ValueError(f"{path} is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if use_adjusted_close and "adjusted_close" in df.columns:
        adj = pd.to_numeric(df["adjusted_close"], errors="coerce")
        ratio = (adj / df["close"]).replace([np.inf, -np.inf], np.nan)
        # Do not propagate pathological split-adjustment ratios.
        ratio = ratio.where((ratio > 0) & (ratio < 1000))
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * ratio

    df["symbol"] = ticker.upper()
    df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"]).set_index("timestamp")
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(numeric) else 0.0
    if numeric_ratio >= 0.90:
        median_abs = numeric.dropna().abs().median()
        if pd.isna(median_abs):
            return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
        if median_abs > 1e14:
            unit = "ns"
        elif median_abs > 1e11:
            unit = "ms"
        elif median_abs > 1e8:
            unit = "s"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "adj_close": "adjusted_close",
        "adj_close_": "adjusted_close",
        "adjustedclose": "adjusted_close",
        "adjusted_close_price": "adjusted_close",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "vol": "volume",
    }
    out = out.rename(columns={c: aliases.get(c, c) for c in out.columns})
    return out


def save_symbol_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = normalize_ohlcv_columns(df)
    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in out.columns:
            if col == "symbol":
                out[col] = path.stem.upper()
            elif col == "volume":
                out[col] = 0.0
            else:
                raise ValueError(f"Cannot save {path}: missing {col}")
    extra = [c for c in ["adjusted_close"] if c in out.columns]
    out[required + extra].to_csv(path, index=False)
