import pandas as pd

from quant_stock_momentum.execution.costs import ExecutionCostConfig, ExecutionCostModel


def test_costs_positive_when_weights_change():
    weights = pd.DataFrame({"AAPL": [0.0, 0.5, 0.5, 0.2], "MSFT": [0.0, -0.5, -0.2, 0.0]})
    model = ExecutionCostModel(ExecutionCostConfig())
    costs = model.cost_series(weights)
    assert costs.iloc[0] == 0
    assert costs.iloc[1] > 0
    assert costs.iloc[2] > 0
    assert costs.iloc[3] > 0
