from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_stock_momentum.config import StockInstrument, yfinance_ticker


class YFinanceStockClient:
    """Adjusted stock-data downloader for research.

    For stock momentum research, adjusted stock prices are preferable because they handle
    splits and dividends. Dukascopy stock CFD data is still supported separately.
    """

    def download(
        self,
        instruments: list[StockInstrument],
        out_dir: str | Path,
        period: str = "10y",
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        skip_failed: bool = False,
    ) -> list[Path]:
        import yfinance as yf

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        failures: list[str] = []
        for inst in instruments:
            ticker = inst.ticker.upper()
            yf_ticker = yfinance_ticker(ticker)
            try:
                if start or end:
                    df = yf.download(yf_ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
                else:
                    df = yf.download(yf_ticker, period=period, auto_adjust=auto_adjust, progress=False)
                if df.empty:
                    raise RuntimeError(f"No yfinance data returned for {ticker}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                out = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(df.index, utc=True),
                        "symbol": ticker,
                        "open": df["Open"].to_numpy(),
                        "high": df["High"].to_numpy(),
                        "low": df["Low"].to_numpy(),
                        "close": df["Close"].to_numpy(),
                        "volume": df.get("Volume", pd.Series(0, index=df.index)).to_numpy(),
                    }
                )
                if "Adj Close" in df.columns and not auto_adjust:
                    out["adjusted_close"] = df["Adj Close"].to_numpy()
                path = out_dir / f"{ticker}.csv"
                out.to_csv(path, index=False)
                saved.append(path)
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")
                print(f"FAILED -> {ticker}: {exc}")
                if not skip_failed:
                    raise
        if failures:
            print("\nyfinance download completed with failures:")
            for item in failures:
                print(f"- {item}")
        return saved
