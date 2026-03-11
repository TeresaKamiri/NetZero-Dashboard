import numpy as np


def generate_shared_scenario_draws(
    n_sims: int,
    seed: int,
    sigma_demand_pp: float,
    sigma_rebound: float,
    delay_prob: float = 0.25,
    sigma_emissions_intensity: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Scenario uncertainty draws shared across option runs.
    These draws represent uncertainty in exogenous assumptions, not model residual noise.
    """
    rng = np.random.default_rng(seed)
    demand_pp = rng.normal(0.0, sigma_demand_pp, n_sims)
    rebound_delta = rng.normal(0.0, sigma_rebound, n_sims)
    delay_years = rng.binomial(2, delay_prob, n_sims)  # 0-2 year shock
    emissions_intensity_delta = rng.normal(0.0, sigma_emissions_intensity, n_sims)
    return {
        "demand_pp": demand_pp,
        "rebound_delta": rebound_delta,
        "delay_years": delay_years,
        "emissions_intensity_delta": emissions_intensity_delta,
    }


def apply_model_uncertainty(path: np.ndarray, sigma_model: float, rng: np.random.Generator) -> np.ndarray:
    """
    Model uncertainty layer (residual/process uncertainty).
    Applied after scenario path construction to avoid double counting with scenario inputs.
    """
    if sigma_model <= 0:
        return path
    noise = rng.normal(0.0, sigma_model, len(path))
    return np.clip(path * (1.0 + noise), 0.0, None)
