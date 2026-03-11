import numpy as np
import pandas as pd

from src.forecast import backtest_calibration_summary, walk_forward_backtest


def test_backtest_calibration_fields_exist():
    years = np.arange(2010, 2024)
    values = np.linspace(100.0, 70.0, len(years)) + np.sin(np.arange(len(years)))
    ts = pd.DataFrame({"Year": years, "Value": values})
    bt = walk_forward_backtest(ts.rename(columns={"Value": "y"}), "y", min_train_size=6)
    assert not bt.empty
    for c in ["PI80_lower", "PI80_upper", "Covered80", "PI95_lower", "PI95_upper", "Covered95"]:
        assert c in bt.columns
    cal = backtest_calibration_summary(bt)
    assert "coverage_80" in cal
    assert "coverage_95" in cal
