from src.evaluation import evaluate_options


def test_option_scorecard_columns():
    base_cfg = {
        "horizon": {"start_year": 2026, "end_year": 2030},
        "baseline_year": 2022,
        "levers": {
            "demand_growth_shock_pct": 0.0,
            "electrification_pace": 0.0,
            "efficiency_improvement_pct": 0.0,
            "policy_delay_years": 0,
            "rebound_intensity": 0.0,
        },
        "interventions": {
            "space_heating_reduction_pct": 0.0,
            "industrial_efficiency_push_pct": 0.0,
            "services_demand_management_pct": 0.0,
        },
        "targets": {"target_values": {"2035": 100.0, "2050": 0.0}, "carbon_budget_total": 1500.0},
        "uncertainty": {
            "enabled": True,
            "scenario_uncertainty_enabled": True,
            "model_uncertainty_enabled": True,
            "n_sims": 30,
            "sigma_demand": 0.03,
            "sigma_rebound": 0.08,
            "sigma_model": 0.01,
            "seed": 42,
        },
        "policy": {"apply_overlays": True},
    }

    options = [
        {"name": "A", "patch": {}, "cost_band": "Low", "delivery_risk": "Low"},
        {"name": "B", "patch": {"interventions": {"space_heating_reduction_pct": 10.0}}, "cost_band": "Medium", "delivery_risk": "Medium"},
    ]
    scorecard, _ = evaluate_options(base_cfg, options)
    assert not scorecard.empty
    expected = {
        "Option",
        "AttainProbPrimary",
        "ExpectedRegretMt",
        "DominanceFreq",
        "CostBand",
        "DeliveryRiskBand",
    }
    assert expected.issubset(set(scorecard.columns))
