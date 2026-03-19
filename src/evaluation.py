import copy
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import compute_kpis
from src.stress_engine import run_scenario
from src.uncertainty import generate_shared_scenario_draws


def patch_config(base_config: dict, patch: dict[str, Any]) -> dict:
    """Apply a shallow-per-group patch on top of a deep-copied base config."""
    out = copy.deepcopy(base_config)
    for group, values in patch.items():
        if isinstance(values, dict):
            out.setdefault(group, {})
            out[group].update(values)
        else:
            out[group] = values
    return out


def _sim_totals_mt(outputs: dict) -> np.ndarray:
    """Return cumulative emissions totals per simulation in MtCO2e."""
    sims = outputs.get("sims")
    if sims is None:
        return np.array([float(np.sum(outputs["p50"]) / 1000.0)])
    return np.sum(sims, axis=1) / 1000.0


def evaluate_options(base_config: dict, options: list[dict]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run all options on shared draws and compute comparative scorecard metrics."""
    n_sims = int(base_config["uncertainty"]["n_sims"]) if base_config["uncertainty"]["enabled"] else 1
    shared_draws = None
    if base_config["uncertainty"]["enabled"]:
        shared_draws = generate_shared_scenario_draws(
            n_sims=n_sims,
            seed=int(base_config["uncertainty"]["seed"]),
            sigma_demand_pp=float(base_config["uncertainty"]["sigma_demand"] * 100.0),
            sigma_rebound=float(base_config["uncertainty"].get("sigma_rebound", 0.08)),
            sigma_emissions_intensity=float(base_config["uncertainty"].get("sigma_emissions_intensity", 0.0)),
        )

    rows = []
    per_option = {}
    totals_lookup = {}
    for opt in options:
        cfg = patch_config(base_config, opt.get("patch", {}))
        out = run_scenario(cfg, shared_draws=shared_draws)
        kpis = compute_kpis(out, cfg)
        totals_mt = _sim_totals_mt(out)
        totals_lookup[opt["name"]] = totals_mt
        per_option[opt["name"]] = pd.DataFrame(
            {"simulation_id": np.arange(len(totals_mt)), "cum_emissions_mt": totals_mt}
        )
        rows.append(
            {
                "Option": opt["name"],
                "PrimaryTargetYear": kpis["primary_target_year"],
                "SecondaryTargetYear": kpis["secondary_target_year"],
                "BreachRiskPrimary": kpis["breach_risk_primary"],
                "AttainProbPrimary": kpis["attain_prob_primary"],
                "BreachRiskSecondary": kpis["breach_risk_secondary"],
                "AttainProbSecondary": kpis["attain_prob_secondary"],
                "BreachRisk2035": kpis["breach_risk_2035"],
                "BreachRisk2050": kpis["breach_risk_2050"],
                "AttainProb2035": 1.0 - kpis["breach_risk_2035"],
                "AttainProb2050": 1.0 - kpis["breach_risk_2050"],
                "ExpectedCumEmissionsMt": float(np.mean(totals_mt)),
                "BudgetGapMt": kpis["budget_gap_mt"],
                "CostBand": opt.get("cost_band", "Unknown"),
                "DeliveryRiskBand": opt.get("delivery_risk", "Unknown"),
            }
        )
        for year, info in kpis["target_breach_risks"].items():
            rows[-1][f"BreachRisk{year}"] = info["breach_risk"]
            rows[-1][f"AttainProb{year}"] = info["attainment_prob"]

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard, per_option

    matrix = np.vstack([totals_lookup[n] for n in scorecard["Option"]])
    # Regret is measured against the best option for each simulation draw.
    min_by_sim = matrix.min(axis=0)
    regrets = matrix - min_by_sim
    winners = np.argmin(matrix, axis=0)

    exp_regret = regrets.mean(axis=1)
    dominance = np.array([(winners == i).mean() for i in range(matrix.shape[0])])
    scorecard["ExpectedRegretMt"] = exp_regret
    scorecard["DominanceFreq"] = dominance
    scorecard = scorecard.sort_values(
        ["AttainProbPrimary", "ExpectedRegretMt", "DeliveryRiskBand"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return scorecard, per_option
