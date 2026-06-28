from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StockInstrument:
    ticker: str
    dukascopy_symbol: str | None = None
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    source: str | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_stock_universe(path: str | Path, include_benchmark: bool = False) -> list[StockInstrument]:
    cfg = load_yaml(path)
    raw = cfg.get("stocks", [])
    if not raw:
        raise ValueError("Instrument config must define a non-empty 'stocks' list.")
    instruments: list[StockInstrument] = []
    for item in raw:
        instruments.append(_instrument_from_item(item))

    if include_benchmark and cfg.get("benchmark"):
        instruments.append(_instrument_from_item(cfg["benchmark"]))

    # De-duplicate while preserving order. This matters when benchmark is also in stocks.
    seen: set[str] = set()
    deduped: list[StockInstrument] = []
    for inst in instruments:
        if inst.ticker not in seen:
            deduped.append(inst)
            seen.add(inst.ticker)
    return deduped


def load_stock_tickers(path: str | Path, include_benchmark: bool = False) -> list[str]:
    return [x.ticker for x in load_stock_universe(path, include_benchmark=include_benchmark)]


def load_sector_map(path: str | Path) -> dict[str, str]:
    return {
        inst.ticker: inst.sector or "Unknown"
        for inst in load_stock_universe(path, include_benchmark=False)
    }


def _instrument_from_item(item: Any) -> StockInstrument:
    if isinstance(item, str):
        ticker = clean_ticker(item)
        return StockInstrument(ticker=ticker, dukascopy_symbol=default_dukascopy_symbol(ticker))
    if isinstance(item, dict):
        ticker = clean_ticker(str(item["ticker"]))
        return StockInstrument(
            ticker=ticker,
            dukascopy_symbol=str(item.get("dukascopy_symbol") or default_dukascopy_symbol(ticker)).strip(),
            name=item.get("name"),
            sector=item.get("sector"),
            industry=item.get("industry"),
            source=item.get("source"),
        )
    raise ValueError(f"Unsupported instrument item: {item!r}")


def clean_ticker(ticker: str) -> str:
    return ticker.upper().strip().replace("/", "-")


def yfinance_ticker(ticker: str) -> str:
    # yfinance uses BRK-B instead of BRK.B for Berkshire class B.
    return clean_ticker(ticker).replace(".", "-")


def default_dukascopy_symbol(ticker: str) -> str:
    """Best-effort Dukascopy stock CFD symbol convention used by dukascopy-node.

    Examples: AAPL -> aaplususd, MSFT -> msftususd. Symbols with dots/classes
    should be specified explicitly in configs/instruments.yml.
    """
    return clean_ticker(ticker).lower().replace(".", "").replace("-", "") + "ususd"
