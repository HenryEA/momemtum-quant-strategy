from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame
    costs: pd.DataFrame
    metrics: dict[str, float | int | str | None]
