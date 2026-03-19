"""KPI, calibration, and sensitivity calculations for stress-test outputs."""

import copy

import numpy as np


KT_PER_MT = 1000.0


def get_configured_target_years(config: dict) -> list[int]:
    """Return configured target years as a sorted unique list of ints."""
    targets = config.get("targets", {})
    years: list[int] = []
    for raw in targets.get("target_years", []):
        try:
            years.append(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in targets.get("target_values", {}).keys():
        try:
            years.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(years))


def _target_for_year(config: dict, year: int) -> float:
    """Return target value in MtCO2e for a year or NaN if target is missing."""
    return float(config.get("targets", {}).get("target_values", {}).get(str(year), np.nan))


def _year_index(outputs: dict, year: int) -> int | None:
    """Return the time index for a target year if it exists in the simulated horizon."""
    idx = np.where(outputs["years"] == year)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def _summary_years(target_summary: dict[int, dict]) -> tuple[int | None, int | None]:
    """Pick primary and secondary target years, favoring configured years in horizon."""
    years_in_horizon = [y for y, info in target_summary.items() if info["in_horizon"]]
    if years_in_horizon:
        primary = max(years_in_horizon)
        secondary_candidates = sorted(y for y in years_in_horizon if y != primary)
        secondary = secondary_candidates[0] if secondary_candidates else None
        return primary, secondary

    configured = sorted(target_summary)
    if not configured:
        return None, None
    primary = configured[-1]
    secondary = configured[0] if len(configured) > 1 else None
    if secondary == primary:
        secondary = None
    return primary, secondary


def _series_mt_for_year(outputs: dict, year: int) -> np.ndarray | None:
    """Return per-simulation emissions in MtCO2e for a given year when simulations exist."""
    sims = outputs.get("sims")
    idx = _year_index(outputs, year)
    if sims is None or idx is None:
        return None
    return sims[:, idx] / KT_PER_MT


def compute_target_breach_summary(outputs: dict, config: dict) -> dict[int, dict]:
    """Summarize breach risk and pathway position for each configured target year."""
    target_summary: dict[int, dict] = {}
    for year in get_configured_target_years(config):
        idx = _year_index(outputs, year)
        target_mt = _target_for_year(config, year)
        p10_mt = np.nan
        p50_mt = np.nan
        p90_mt = np.nan
        breach_risk = np.nan
        attainment_prob = np.nan
        status = "outside_horizon"
        target_position = "outside_horizon"
        saturation = "none"

        if idx is not None:
            p10_mt = float(outputs["p10"][idx] / KT_PER_MT)
            p50_mt = float(outputs["p50"][idx] / KT_PER_MT)
            p90_mt = float(outputs["p90"][idx] / KT_PER_MT)
            sims_mt = _series_mt_for_year(outputs, year)
            status = "ok"
            if sims_mt is not None:
                breach_risk = float(np.mean(sims_mt > target_mt))
                attainment_prob = 1.0 - breach_risk
                if breach_risk >= 0.99:
                    saturation = "high"
                elif breach_risk <= 0.01:
                    saturation = "low"
            else:
                status = "deterministic"

            if np.isfinite(target_mt):
                if target_mt < p10_mt:
                    target_position = "below_p10"
                elif target_mt > p90_mt:
                    target_position = "above_p90"
                else:
                    target_position = "within_p10_p90"

        target_summary[year] = {
            "year": year,
            "target_mt": target_mt,
            "breach_risk": breach_risk,
            "attainment_prob": attainment_prob,
            "p10_mt": p10_mt,
            "p50_mt": p50_mt,
            "p90_mt": p90_mt,
            "in_horizon": idx is not None,
            "status": status,
            "target_position": target_position,
            "saturation": saturation,
        }
    return target_summary


def solve_target_for_breach(outputs: dict, year: int, desired_breach_risk: float) -> dict:
    """Solve for the target threshold in MtCO2e that matches a desired breach probability."""
    if not 0.0 <= desired_breach_risk <= 1.0:
        raise ValueError("desired_breach_risk must be between 0 and 1.")

    sims_mt = _series_mt_for_year(outputs, year)
    if sims_mt is None:
        return {
            "year": year,
            "desired_breach_risk": desired_breach_risk,
            "target_mt": np.nan,
            "achieved_breach_risk": np.nan,
            "status": "unavailable",
            "message": "Target year is outside the current horizon or uncertainty is disabled.",
        }

    quantile = float(np.quantile(sims_mt, 1.0 - desired_breach_risk))
    achieved = float(np.mean(sims_mt > quantile))
    return {
        "year": year,
        "desired_breach_risk": desired_breach_risk,
        "target_mt": quantile,
        "achieved_breach_risk": achieved,
        "p10_mt": float(np.quantile(sims_mt, 0.10)),
        "p50_mt": float(np.quantile(sims_mt, 0.50)),
        "p90_mt": float(np.quantile(sims_mt, 0.90)),
        "status": "ok",
        "message": "Solved from the current simulation distribution for the selected year.",
    }


def solve_single_lever_for_breach(
    base_config: dict,
    lever_group: str,
    lever_name: str,
    candidate_values: list[float],
    target_year: int,
    desired_breach_risk: float,
) -> dict:
    """Grid-search a single lever and return the closest reachable breach risk."""
    from src.stress_engine import run_scenario

    if not candidate_values:
        return {
            "status": "unavailable",
            "message": "No candidate lever values were provided.",
            "evaluated": [],
        }

    evaluated = []
    best = None
    for value in candidate_values:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault(lever_group, {})
        cfg[lever_group][lever_name] = value
        out = run_scenario(cfg)
        kpis = compute_kpis(out, cfg)
        risk = float(kpis.get(f"breach_risk_{target_year}", np.nan))
        row = {
            "value": value,
            "breach_risk": risk,
            "distance_to_target": float(abs(risk - desired_breach_risk)) if np.isfinite(risk) else np.nan,
        }
        evaluated.append(row)
        if np.isfinite(risk):
            if best is None or row["distance_to_target"] < best["distance_to_target"]:
                best = row

    finite_risks = [r["breach_risk"] for r in evaluated if np.isfinite(r["breach_risk"])]
    if best is None:
        return {
            "status": "unavailable",
            "message": "The selected target year is outside the current horizon or uncertainty is disabled.",
            "evaluated": evaluated,
        }

    risk_min = float(min(finite_risks))
    risk_max = float(max(finite_risks))
    reachable = risk_min <= desired_breach_risk <= risk_max or risk_max <= desired_breach_risk <= risk_min
    return {
        "status": "ok",
        "message": "Closest value found from a single-lever grid search.",
        "lever_group": lever_group,
        "lever_name": lever_name,
        "target_year": target_year,
        "desired_breach_risk": desired_breach_risk,
        "best_value": best["value"],
        "best_breach_risk": best["breach_risk"],
        "distance_to_target": best["distance_to_target"],
        "risk_min": risk_min,
        "risk_max": risk_max,
        "target_reachable": reachable,
        "evaluated": evaluated,
    }


def compute_kpis(outputs: dict, config: dict) -> dict:
    """Compute headline risk, budget, and pathway KPIs from scenario outputs."""
    years = outputs["years"]
    sims = outputs["sims"]
    p50 = outputs["p50"]
    peak_year = int(years[int(np.argmax(p50))])
    target_summary = compute_target_breach_summary(outputs, config)
    primary_year, secondary_year = _summary_years(target_summary)

    def _budget_gap() -> float:
        if sims is None:
            return np.nan
        budget = float(config["targets"]["carbon_budget_total"])
        cum_mt = np.sum(sims, axis=1) / KT_PER_MT
        return float(np.mean(np.maximum(0, cum_mt - budget)))

    out = {
        "target_breach_risks": target_summary,
        "configured_target_years": list(target_summary),
        "primary_target_year": primary_year,
        "secondary_target_year": secondary_year,
        "breach_risk_primary": target_summary.get(primary_year, {}).get("breach_risk", np.nan),
        "attain_prob_primary": target_summary.get(primary_year, {}).get("attainment_prob", np.nan),
        "breach_risk_secondary": target_summary.get(secondary_year, {}).get("breach_risk", np.nan),
        "attain_prob_secondary": target_summary.get(secondary_year, {}).get("attainment_prob", np.nan),
        "budget_gap_mt": _budget_gap(),
        "peak_year": peak_year,
        "cumulative_emissions_p50_mt": float(np.sum(p50) / KT_PER_MT),
    }
    for year, info in target_summary.items():
        out[f"breach_risk_{year}"] = info["breach_risk"]
        out[f"attain_prob_{year}"] = info["attainment_prob"]
    for legacy_year in (2035, 2050):
        out.setdefault(f"breach_risk_{legacy_year}", np.nan)
        out.setdefault(f"attain_prob_{legacy_year}", np.nan)
    return out


def calculate_budget_exhaustion_year(outputs: dict, config: dict):
    """Find the first year cumulative p50 emissions exceed the configured budget."""
    years = outputs["years"]
    budget_kt = float(config["targets"]["carbon_budget_total"]) * KT_PER_MT
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
    cumulative_mt = float(np.sum(outputs["p50"]) / KT_PER_MT)
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
    target_summary = base_k["target_breach_risks"]
    candidate_years = [y for y, info in target_summary.items() if info["in_horizon"] and np.isfinite(info["breach_risk"])]

    # Adaptive metric selection: if breach risk is saturated (near 0% or 100%),
    # use a continuous outcome so the tornado remains informative.
    if not candidate_years and np.isnan(base_k["cumulative_emissions_p50_mt"]):
        return []
    selected_year = None
    for year in sorted(candidate_years, reverse=True):
        risk = target_summary[year]["breach_risk"]
        if 0.01 < risk < 0.99:
            selected_year = year
            break
    if selected_year is None:
        metric_key = "cumulative_emissions_p50_mt"
        unit = "Mt"
        scale = 1.0
        label = "cumulative emissions p50"
    else:
        metric_key = f"breach_risk_{selected_year}"
        unit = "pp"
        scale = 100.0
        label = f"{selected_year} breach risk"

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
