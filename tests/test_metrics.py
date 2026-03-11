import numpy as np

from src.metrics import calculate_budget_exhaustion_year, compute_kpis


def test_compute_kpis_basic():
    years = np.arange(2026, 2031)
    sims = np.array(
        [
            [100, 90, 80, 70, 60],
            [110, 95, 85, 75, 65],
        ]
    )
    outputs = {
        "years": years,
        "sims": sims,
        "p10": np.quantile(sims, 0.1, axis=0),
        "p50": np.quantile(sims, 0.5, axis=0),
        "p90": np.quantile(sims, 0.9, axis=0),
    }
    config = {
        "targets": {
            "target_values": {"2035": 100.0, "2050": 0.0, "2030": 70.0},
            "carbon_budget_total": 1.0,  # intentionally tiny to trigger gap
        }
    }
    k = compute_kpis(outputs, config)
    assert "budget_gap_mt" in k
    assert np.isfinite(k["budget_gap_mt"])


def test_exhaustion_year():
    years = np.array([2026, 2027, 2028])
    outputs = {"years": years, "p50": np.array([1000, 1000, 1000]), "sims": None}
    config = {"targets": {"carbon_budget_total": 1.5, "target_values": {"2035": 100, "2050": 0}}}
    yr = calculate_budget_exhaustion_year(outputs, config)
    assert yr == 2027
