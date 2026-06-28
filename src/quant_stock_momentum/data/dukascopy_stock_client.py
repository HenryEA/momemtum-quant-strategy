from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Literal

import pandas as pd

from quant_stock_momentum.data.loader import normalize_ohlcv_columns, save_symbol_csv


Timeframe = Literal["d1", "h1", "m30", "m15", "m5", "m1"]


@dataclass(frozen=True)
class DukascopyStockRequest:
    ticker: str
    dukascopy_symbol: str
    start: datetime
    end: datetime
    timeframe: Timeframe = "d1"


class DukascopyStockClient:
    """Downloader for Dukascopy stock CFD candles through the dukascopy-node CLI.

    The dukascopy-node CLI writes downloaded files into a download directory.
    Earlier versions of this adapter incorrectly assumed the CSV would always be
    returned on stdout. This implementation passes an explicit temporary output
    directory, reads the generated CSV from disk, then normalizes it for the
    Python backtester.
    """

    def __init__(self, npx_command: str | None = None, package: str = "dukascopy-node") -> None:
        # On Windows, subprocess needs npx.cmd; on macOS/Linux, npx is correct.
        self.npx_command = npx_command or ("npx.cmd" if os.name == "nt" else "npx")
        self.package = package

    def fetch(self, req: DukascopyStockRequest) -> pd.DataFrame:
        if not shutil.which(self.npx_command):
            raise RuntimeError(
                "npx was not found. Install Node.js LTS, reopen VS Code, then retry. "
                "For an offline smoke test, run scripts/make_sample_data.py instead."
            )

        with tempfile.TemporaryDirectory(prefix=f"dukascopy_{req.ticker.lower()}_") as tmp:
            tmp_dir = Path(tmp)
            cmd = [
                self.npx_command,
                "-y",
                self.package,
                "-i",
                req.dukascopy_symbol,
                "-from",
                req.start.strftime("%Y-%m-%d"),
                "-to",
                req.end.strftime("%Y-%m-%d"),
                "-t",
                req.timeframe,
                "-f",
                "csv",
                "-dir",
                str(tmp_dir),
                "-fn",
                req.ticker.upper(),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Dukascopy download failed for {req.ticker} ({req.dukascopy_symbol}).\n"
                    f"Command: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
                )

            csv_files = sorted(
                [p for p in tmp_dir.rglob("*.csv") if p.is_file()],
                key=lambda p: p.stat().st_size,
                reverse=True,
            )

            raw = ""
            if csv_files:
                raw = csv_files[0].read_text(encoding="utf-8", errors="ignore").strip()
            else:
                # Some old/new CLI builds may still emit CSV on stdout. Keep this fallback.
                raw = _extract_csv_from_stdout(proc.stdout)

            if not raw:
                raise RuntimeError(
                    f"No CSV rows returned for {req.ticker} ({req.dukascopy_symbol}).\n"
                    f"Command: {' '.join(cmd)}\nSTDOUT preview:\n{proc.stdout[:1200]}\nSTDERR preview:\n{proc.stderr[:1200]}"
                )

            df = self._parse_csv(raw, req.ticker)
            if df.empty:
                raise RuntimeError(
                    f"Parsed Dukascopy CSV for {req.ticker} ({req.dukascopy_symbol}) but got zero valid rows.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Raw CSV preview:\n{raw[:1200]}\nSTDOUT preview:\n{proc.stdout[:1200]}\nSTDERR preview:\n{proc.stderr[:1200]}"
                )
            return df

    def download_to_csv(self, req: DukascopyStockRequest, out_dir: str | Path) -> Path:
        df = self.fetch(req)
        out = Path(out_dir) / f"{req.ticker.upper()}.csv"
        save_symbol_csv(df, out)
        return out

    def _parse_csv(self, raw_csv: str, ticker: str) -> pd.DataFrame:
        lines = [line.strip() for line in raw_csv.splitlines() if line.strip()]
        if not lines:
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        first_line = lines[0].lower()
        if "open" in first_line and "close" in first_line:
            df = pd.read_csv(StringIO("\n".join(lines)))
        else:
            df = pd.read_csv(StringIO("\n".join(lines)), header=None)
            # Common dukascopy-node OHLCV shape: timestamp,open,high,low,close,volume.
            if df.shape[1] >= 6:
                df = df.iloc[:, :6]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            elif df.shape[1] == 5:
                df.columns = ["timestamp", "open", "high", "low", "close"]
                df["volume"] = 0.0
            else:
                raise ValueError(f"Could not parse Dukascopy CSV for {ticker}. Preview: {raw_csv[:500]}")

        df = normalize_ohlcv_columns(df)
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Dukascopy CSV for {ticker} is missing {col}. Columns: {list(df.columns)}")
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["timestamp"] = _parse_dukascopy_timestamp(df["timestamp"])
        df["symbol"] = ticker.upper()
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def _extract_csv_from_stdout(stdout: str) -> str:
    """Return CSV-like lines from stdout when a CLI build prints CSV there."""
    lines = []
    for line in stdout.splitlines():
        clean = line.strip()
        if not clean:
            continue
        lower = clean.lower()
        # Header line or rows beginning with a Unix timestamp / ISO date.
        if ("timestamp" in lower and "open" in lower and "close" in lower) or clean[:1].isdigit():
            lines.append(clean)
    return "\n".join(lines).strip()


def _parse_dukascopy_timestamp(series: pd.Series) -> pd.Series:
    """Parse dukascopy-node timestamps robustly.

    dukascopy-node CSV commonly emits Unix timestamps in milliseconds, for
    example 1612137600000. Pandas needs the unit to parse these reliably.
    """
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
