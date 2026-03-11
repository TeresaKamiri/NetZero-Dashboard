"""KPI and sensitivity calculations for stress-test outputs."""

import numpy as np
import copy


def _target_for_year(config: dict, year: int) -> float:
    """Return target value for a year or NaN if target is missing."""
    return float(config["targets"]["target_values"].get(str(year), np.nan))


def compute_kpis(outputs: dict, config: dict) -> dict:
    """Compute headline risk, budget, and pathway KPIs from scenario outputs."""
    years = outputs["years"]
    sims = outputs["sims"]
    p50 = outputs["p50"]
    peak_year = int(years[int(np.argmax(p50))])

    def _breach_risk(y: int) -> float:
        if sims is None:
            return np.nan
        idx = np.where(years == y)[0]
        if len(idx) == 0:
            return np.nan
        i = int(idx[0])
        target = _target_for_year(config, y)
        return float(np.mean(sims[:, i] > target))

    def _budget_gap() -> float:
        if sims is None:
            return np.nan
        budget = float(config["targets"]["carbon_budget_total"])
        cum_mt = np.sum(sims, axis=1) / 1000.0
        return float(np.mean(np.maximum(0, cum_mt - budget)))

    return {
        "breach_risk_2035": _breach_risk(2035),
        "breach_risk_2050": _breach_risk(2050),
        "budget_gap_mt": _budget_gap(),
        "peak_year": peak_year,
        "cumulative_emissions_p50_mt": float(np.sum(p50) / 1000.0),
    }


def calculate_budget_exhaustion_year(outputs: dict, config: dict):
    """Find the first year cumulative p50 emissions exceed the configured budget."""
    years = outputs["years"]
    budget_kt = float(config["targets"]["carbon_budget_total"]) * 1000.0
    cum = np.cumsum(outputs["p50"])
    idx = np.where(cum > budget_kt)[0]
    if len(idx) == 0:
        return "Post-2050"
    return int(years[int(idx[0])])


def assess_budget_plausibility(outputs: dict, config: dict) -> dict:
    """
    Guardrail for unit/scale plausibility of configured carbon budget
    relative to modeled cumulative emissions over the chosen horizon.
    """
    budget_mt = float(config["targets"]["carbon_budget_total"])
    cumulative_mt = float(np.sum(outputs["p50"]) / 1000.0)
    if cumulative_mt <= 0:
        return {
            "status": "invalid",
            "ratio_budget_to_cumulative": np.nan,
            "message": "Modeled cumulative emissions are non-positive; check data and configuration.",
        }

    ratio = budget_mt / cumulative_mt
    if ratio < 0.30:
        status = "too_low"
        msg = (
            f"Configured budget ({budget_mt:.1f} Mt) is very low versus modeled cumulative emissions "
            f"({cumulative_mt:.1f} Mt, ratio={ratio:.2f}). Early exhaustion may reflect scale mismatch "
            "rather than policy failure."
        )
    elif ratio > 3.0:
        status = "too_high"
        msg = (
            f"Configured budget ({budget_mt:.1f} Mt) is very high versus modeled cumulative emissions "
            f"({cumulative_mt:.1f} Mt, ratio={ratio:.2f}). Breach risk may be understated."
        )
    else:
        status = "ok"
        msg = (
            f"Budget scale appears plausible against modeled cumulative emissions "
            f"(ratio={ratio:.2f})."
        )

    return {
        "status": status,
        "ratio_budget_to_cumulative": float(ratio),
        "budget_mt": budget_mt,
        "cumulative_emissions_p50_mt": cumulative_mt,
        "message": msg,
    }


def compute_sensitivity_tornado(base_config: dict) -> list[dict]:
    """Estimate one-at-a-time lever influence for tornado chart ranking."""
    from src.stress_engine import run_scenario

    # Use a lighter sensitivity run config so chart renders quickly in Streamlit.
    cfg = copy.deepcopy(base_config)
    if cfg.get("uncertainty", {}).get("enabled", False):
        cfg["uncertainty"]["n_sims"] = int(min(cfg["uncertainty"].get("n_sims", 200), 150))
        cfg["uncertainty"]["model_uncertainty_enabled"] = False

    base_out = run_scenario(cfg)
    base_k = compute_kpis(base_out, cfg)
    base_r2050 = base_k["breach_risk_2050"]
    base_r2035 = base_k["breach_risk_2035"]

    # Adaptive metric selection: if breach risk is saturated (near 0% or 100%),
    # use a continuous outcome so the tornado remains informative.
    if np.isnan(base_r2050):
        return []
    if 0.01 < base_r2050 < 0.99:
        metric_key = "breach_risk_2050"
        unit = "pp"
        scale = 100.0
        label = "2050 breach risk"
    elif 0.01 < base_r2035 < 0.99:
        metric_key = "breach_risk_2035"
        unit = "pp"
        scale = 100.0
        label = "2035 breach risk"
    else:
        metric_key = "cumulative_emissions_p50_mt"
        unit = "Mt"
        scale = 1.0
        label = "cumulative emissions p50"

    deltas = {
        "demand_growth_shock_pct": 2.0,
        "efficiency_improvement_pct": 2.0,
        "policy_delay_years": 1,
        "rebound_intensity": 0.1,
        "electrification_pace": 0.1,
    }
    rows = []
    for lever, delta in deltas.items():
        # Symmetric +/- shocks reduce directional bias in finite-difference deltas.
        hi = copy.deepcopy(cfg)
        lo = copy.deepcopy(cfg)
        hi["levers"][lever] = hi["levers"].get(lever, 0.0) + delta
        lo["levers"][lever] = lo["levers"].get(lever, 0.0) - delta
        hi_val = compute_kpis(run_scenario(hi), hi)[metric_key]
        lo_val = compute_kpis(run_scenario(lo), lo)[metric_key]
        rows.append(
            {
                "lever": lever.replace("_", " ").title(),
                "delta_value": float((hi_val - lo_val) * scale),
                "unit": unit,
                "metric": label,
            }
        )
    rows.sort(key=lambda x: abs(x["delta_value"]), reverse=True)
    return rows
