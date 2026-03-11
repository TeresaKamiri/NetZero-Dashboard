import numpy as np

from src.uncertainty import generate_shared_scenario_draws


def test_shared_draws_reproducible():
    d1 = generate_shared_scenario_draws(50, 42, 2.0, 0.1)
    d2 = generate_shared_scenario_draws(50, 42, 2.0, 0.1)
    assert np.allclose(d1["demand_pp"], d2["demand_pp"])
    assert np.allclose(d1["rebound_delta"], d2["rebound_delta"])
    assert np.array_equal(d1["delay_years"], d2["delay_years"])


def test_shared_draws_include_emissions_intensity():
    d = generate_shared_scenario_draws(30, 7, 1.0, 0.05, sigma_emissions_intensity=0.02)
    assert "emissions_intensity_delta" in d
    assert d["emissions_intensity_delta"].shape == (30,)
