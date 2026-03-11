import numpy as np
import pandas as pd

from src.data import load_energy_data, load_policy_overlays
from src.uncertainty import apply_model_uncertainty, generate_shared_scenario_draws


def _starting_sector_values(baseline_year: int = 2022) -> dict[str, float]:
    """Get sector start values from baseline year, with deterministic fallback."""
    df = load_energy_data()
    base = df[df["Year"] == baseline_year].copy()
    if base.empty:
        base = df[df["Year"] == int(df["Year"].max())].copy()
    vals = base.groupby("Sector")["Emissions (ktCO2e)"].sum().to_dict()
    if not vals:
        return {"Domestic": 100000.0, "Industrial": 80000.0, "Services": 70000.0}
    return vals


def _overlay_multiplier(sector: str, year: int) -> float:
    """Return mean overlay multiplier for a sector-year pair."""
    ov = load_policy_overlays()
    if ov.empty:
        return 1.0
    hits = ov[(ov["sector_norm"] == sector) & (ov["year"] == year)]
    if hits.empty:
        return 1.0
    return float(hits["multiplier_emissions"].mean())


def _build_path(
    years: np.ndarray,
    start_val: float,
    sector: str,
    demand_growth_shock_pct: float,
    delay_years: int,
    rebound_intensity: float,
    efficiency_improvement_pct: float,
    electrification_pace: float,
    cooking_electrification_boost: float,
    emissions_intensity_delta: float,
    sector_interventions: dict[str, float],
    apply_overlays: bool,
    overlay_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Build a single sector trajectory from levers, interventions, and overlays."""
    t = len(years)
    decay = np.linspace(start_val, start_val * 0.1, t)
    shock = 1 + demand_growth_shock_pct / 100.0
    rebound = 1 + (0.10 * rebound_intensity)
    efficiency_factor = max(0.0, 1 - (efficiency_improvement_pct / 100.0))
    electrification_factor = max(0.0, 1 - (0.10 * electrification_pace))
    cooking_factor = 1.0
    if sector == "Domestic":
        cooking_factor = max(0.0, 1 - (cooking_electrification_boost / 100.0))
    intensity_factor = max(0.0, 1 + emissions_intensity_delta)

    path = (
        decay
        * shock
        * rebound
        * efficiency_factor
        * electrification_factor
        * cooking_factor
        * intensity_factor
        * sector_interventions.get(sector, 1.0)
    )
    if apply_overlays and overlay_vector is not None:
        path = path * overlay_vector
    if delay_years > 0:
        path = np.concatenate(
            [np.repeat(path[0], min(delay_years, t)), path[:-min(delay_years, t)]]
        )
    return path


def run_scenario(config: dict, shared_draws: dict[str, np.ndarray] | None = None) -> dict:
    """Simulate sector and total emissions for the configured horizon."""
    years = np.arange(config["horizon"]["start_year"], config["horizon"]["end_year"] + 1)
    t = len(years)
    n_sims = config["uncertainty"]["n_sims"] if config["uncertainty"]["enabled"] else 1

    starts = _starting_sector_values(config.get("baseline_year", 2022))
    base_shock_pct = float(config["levers"]["demand_growth_shock_pct"])
    base_delay = int(config["levers"]["policy_delay_years"])
    base_rebound = float(config["levers"]["rebound_intensity"])
    base_electrification = float(config["levers"].get("electrification_pace", 0.0))
    base_efficiency = float(config["levers"].get("efficiency_improvement_pct", 0.0))
    base_cooking = float(config.get("interventions", {}).get("cooking_electrification_boost", 0.0))
    apply_overlays = bool(config.get("policy", {}).get("apply_overlays", True))

    sector_interventions = {
        "Domestic": 1 - (config["interventions"]["space_heating_reduction_pct"] / 100.0),
        "Industrial": 1 - (config["interventions"]["industrial_efficiency_push_pct"] / 100.0),
        "Services": 1 - (config["interventions"]["services_demand_management_pct"] / 100.0),
    }

    scenario_uncertainty_enabled = bool(config["uncertainty"].get("scenario_uncertainty_enabled", True))
    model_uncertainty_enabled = bool(config["uncertainty"].get("model_uncertainty_enabled", True))
    sigma_model = float(config["uncertainty"].get("sigma_model", 0.015))
    sigma_emissions_intensity = float(config["uncertainty"].get("sigma_emissions_intensity", 0.0))

    if config["uncertainty"]["enabled"] and scenario_uncertainty_enabled:
        if shared_draws is None:
            shared_draws = generate_shared_scenario_draws(
                n_sims=n_sims,
                seed=int(config["uncertainty"]["seed"]),
                sigma_demand_pp=float(config["uncertainty"]["sigma_demand"] * 100.0),
                sigma_rebound=float(config["uncertainty"].get("sigma_rebound", 0.08)),
                sigma_emissions_intensity=sigma_emissions_intensity,
            )
    else:
        shared_draws = None

    # Precompute overlay vectors once per sector for speed (tornado calls run_scenario many times).
    overlay_by_sector: dict[str, np.ndarray] = {}
    if apply_overlays:
        for sector in starts:
            overlay_by_sector[sector] = np.array([_overlay_multiplier(sector, y) for y in years], dtype=float)

    sims_total = np.zeros((n_sims, t))
    sector_rows = []
    for sector, start_val in starts.items():
        if config["uncertainty"]["enabled"]:
            sims = np.zeros((n_sims, t))
            for i in range(n_sims):
                if shared_draws is not None:
                    d_shock = float(shared_draws["demand_pp"][i])
                    d_rebound = float(shared_draws["rebound_delta"][i])
                    d_delay = int(shared_draws["delay_years"][i])
                    d_intensity = float(shared_draws.get("emissions_intensity_delta", np.zeros(n_sims))[i])
                else:
                    d_shock = 0.0
                    d_rebound = 0.0
                    d_delay = 0
                    d_intensity = 0.0

                path = _build_path(
                    years=years,
                    start_val=float(start_val),
                    sector=sector,
                    demand_growth_shock_pct=base_shock_pct + d_shock,
                    delay_years=max(0, base_delay + d_delay),
                    rebound_intensity=max(0.0, base_rebound + d_rebound),
                    efficiency_improvement_pct=max(0.0, base_efficiency),
                    electrification_pace=max(0.0, base_electrification),
                    cooking_electrification_boost=max(0.0, base_cooking),
                    emissions_intensity_delta=d_intensity,
                    sector_interventions=sector_interventions,
                    apply_overlays=apply_overlays,
                    overlay_vector=overlay_by_sector.get(sector),
                )
                if model_uncertainty_enabled:
                    # Derive a stable RNG stream per sector/simulation for reproducibility.
                    rng_model = np.random.default_rng(
                        int(config["uncertainty"]["seed"]) + (hash((sector, i, "model")) % 1_000_000)
                    )
                    path = apply_model_uncertainty(path, sigma_model=sigma_model, rng=rng_model)
                sims[i, :] = path
        else:
            path = _build_path(
                years=years,
                start_val=float(start_val),
                sector=sector,
                demand_growth_shock_pct=base_shock_pct,
                delay_years=base_delay,
                rebound_intensity=base_rebound,
                efficiency_improvement_pct=max(0.0, base_efficiency),
                electrification_pace=max(0.0, base_electrification),
                cooking_electrification_boost=max(0.0, base_cooking),
                emissions_intensity_delta=0.0,
                sector_interventions=sector_interventions,
                apply_overlays=apply_overlays,
                overlay_vector=overlay_by_sector.get(sector),
            )
            sims = path.reshape(1, -1)

        sims_total += sims
        sector_rows.append(
            pd.DataFrame({"year": years, "sector": sector, "value": np.median(sims, axis=0)})
        )

    return {
        "years": years,
        "sims": sims_total if config["uncertainty"]["enabled"] else None,
        "p10": np.quantile(sims_total, 0.10, axis=0),
        "p50": np.quantile(sims_total, 0.50, axis=0),
        "p90": np.quantile(sims_total, 0.90, axis=0),
        "sector_emissions": pd.concat(sector_rows),
    }
