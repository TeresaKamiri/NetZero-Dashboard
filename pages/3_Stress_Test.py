import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_policy_events
from src.export import export_dataframe_csv, export_dict_json, export_matplotlib_png
from src.metrics import (
    assess_budget_plausibility,
    calculate_budget_exhaustion_year,
    compute_kpis,
    compute_sensitivity_tornado,
    solve_single_lever_for_breach,
    solve_target_for_breach,
)
from src.policy import compute_policy_gap_windows
from src.stress_engine import run_scenario
from src.viz import plot_fan_chart, plot_tornado


LEVER_SPECS = {
    "Demand growth shock (%)": {"group": "levers", "name": "demand_growth_shock_pct", "min": -10.0, "max": 10.0, "step": 0.5},
    "Policy delay (years)": {"group": "levers", "name": "policy_delay_years", "min": 0.0, "max": 8.0, "step": 1.0},
    "Rebound intensity": {"group": "levers", "name": "rebound_intensity", "min": 0.0, "max": 1.0, "step": 0.05},
    "Electrification pace": {"group": "levers", "name": "electrification_pace", "min": 0.0, "max": 1.0, "step": 0.05},
    "Efficiency improvement (%)": {"group": "levers", "name": "efficiency_improvement_pct", "min": 0.0, "max": 10.0, "step": 0.5},
    "Space heating reduction (%)": {"group": "interventions", "name": "space_heating_reduction_pct", "min": 0.0, "max": 30.0, "step": 1.0},
    "Cooking electrification boost (%)": {"group": "interventions", "name": "cooking_electrification_boost", "min": 0.0, "max": 30.0, "step": 1.0},
    "Industrial efficiency push (%)": {"group": "interventions", "name": "industrial_efficiency_push_pct", "min": 0.0, "max": 20.0, "step": 0.5},
    "Services demand management (%)": {"group": "interventions", "name": "services_demand_management_pct", "min": 0.0, "max": 15.0, "step": 0.5},
}


def _target_assessment(info: dict) -> str:
    """Translate target summary diagnostics into user-facing text."""
    if info["status"] == "outside_horizon":
        return "Target year is outside the current horizon."
    if info["status"] == "deterministic":
        return "Enable uncertainty to estimate a breach probability."
    if info["saturation"] == "high":
        return "Configured target sits below nearly the entire simulated range."
    if info["saturation"] == "low":
        return "Configured target sits above nearly the entire simulated range."
    return "Configured target cuts through the simulated range."


st.title("Stress Test")
st.caption("Scenario uncertainty, target-breach risk, and fragility diagnostics.")

if "config" not in st.session_state:
    st.warning("Please initialize assumptions in Executive Overview first.")
    st.stop()

cfg = st.session_state["config"]

with st.sidebar:
    # Sidebar controls mutate shared scenario config used by downstream modules.
    st.subheader("Scenario levers")
    cfg["levers"]["demand_growth_shock_pct"] = st.slider(
        "Demand growth shock (%)", -10.0, 10.0, float(cfg["levers"]["demand_growth_shock_pct"]), 0.5
    )
    cfg["levers"]["policy_delay_years"] = st.slider(
        "Policy delay (years)", 0, 8, int(cfg["levers"]["policy_delay_years"]), 1
    )
    cfg["levers"]["rebound_intensity"] = st.slider(
        "Rebound intensity", 0.0, 1.0, float(cfg["levers"]["rebound_intensity"]), 0.05
    )
    cfg["levers"]["electrification_pace"] = st.slider(
        "Electrification pace", 0.0, 1.0, float(cfg["levers"].get("electrification_pace", 0.0)), 0.05
    )
    cfg["levers"]["efficiency_improvement_pct"] = st.slider(
        "Efficiency improvement (%)",
        0.0,
        10.0,
        float(cfg["levers"].get("efficiency_improvement_pct", 0.0)),
        0.5,
    )

    st.subheader("Interventions")
    cfg["interventions"]["space_heating_reduction_pct"] = st.slider(
        "Space heating reduction (%)", 0.0, 30.0, float(cfg["interventions"]["space_heating_reduction_pct"]), 1.0
    )
    cfg["interventions"]["cooking_electrification_boost"] = st.slider(
        "Cooking electrification boost (%)",
        0.0,
        30.0,
        float(cfg["interventions"].get("cooking_electrification_boost", 0.0)),
        1.0,
    )
    cfg["interventions"]["industrial_efficiency_push_pct"] = st.slider(
        "Industrial efficiency push (%)", 0.0, 20.0, float(cfg["interventions"]["industrial_efficiency_push_pct"]), 0.5
    )
    cfg["interventions"]["services_demand_management_pct"] = st.slider(
        "Services demand management (%)", 0.0, 15.0, float(cfg["interventions"]["services_demand_management_pct"]), 0.5
    )
    cfg["policy"]["apply_overlays"] = st.checkbox(
        "Apply policy overlays", bool(cfg["policy"].get("apply_overlays", True))
    )
    cfg["policy"]["default_coverage_years"] = st.slider(
        "Policy coverage window (years)",
        1,
        10,
        int(cfg["policy"].get("default_coverage_years", 3)),
        1,
        help="If an event has no explicit start/end, treat it as covering this many years from its event year.",
    )

    cfg["uncertainty"]["enabled"] = st.checkbox("Enable uncertainty", bool(cfg["uncertainty"]["enabled"]))
    cfg["uncertainty"]["scenario_uncertainty_enabled"] = st.checkbox(
        "Scenario uncertainty enabled", bool(cfg["uncertainty"].get("scenario_uncertainty_enabled", True))
    )
    cfg["uncertainty"]["model_uncertainty_enabled"] = st.checkbox(
        "Model uncertainty enabled", bool(cfg["uncertainty"].get("model_uncertainty_enabled", True))
    )
    cfg["uncertainty"]["n_sims"] = st.slider("n sims", 100, 1000, int(cfg["uncertainty"]["n_sims"]), 50)
    cfg["uncertainty"]["sigma_demand"] = st.slider(
        "Scenario sigma demand", 0.005, 0.10, float(cfg["uncertainty"]["sigma_demand"]), 0.005
    )
    cfg["uncertainty"]["sigma_rebound"] = st.slider(
        "Scenario sigma rebound", 0.01, 0.30, float(cfg["uncertainty"].get("sigma_rebound", 0.08)), 0.01
    )
    cfg["uncertainty"]["sigma_emissions_intensity"] = st.slider(
        "Scenario sigma emissions intensity",
        0.0,
        0.10,
        float(cfg["uncertainty"].get("sigma_emissions_intensity", 0.02)),
        0.005,
    )
    cfg["uncertainty"]["sigma_model"] = st.slider(
        "Model sigma", 0.0, 0.10, float(cfg["uncertainty"].get("sigma_model", 0.015)), 0.005
    )

st.session_state["config"] = cfg

outputs = run_scenario(cfg)
kpis = compute_kpis(outputs, cfg)
exh = calculate_budget_exhaustion_year(outputs, cfg)
budget_diag = assess_budget_plausibility(outputs, cfg)
target_summary = kpis["target_breach_risks"]

metric_cols = st.columns(max(2, len(target_summary) + 2))
for i, year in enumerate(sorted(target_summary)):
    info = target_summary[year]
    value = f"{info['breach_risk']:.1%}" if np.isfinite(info["breach_risk"]) else "N/A"
    delta = f"target {info['target_mt']:.1f} Mt" if np.isfinite(info["target_mt"]) else "no target"
    if not info["in_horizon"]:
        delta = "outside horizon"
    elif info["status"] == "deterministic":
        delta = "enable uncertainty"
    metric_cols[i].metric(f"Breach risk {year}", value, delta)
metric_cols[-2].metric("Budget gap", f"{kpis['budget_gap_mt']:.1f} Mt")
metric_cols[-1].metric("Budget exhaustion", str(exh))

diag_rows = []
for year in sorted(target_summary):
    info = target_summary[year]
    diag_rows.append(
        {
            "Target year": year,
            "Target (MtCO2e)": info["target_mt"],
            "P10 (MtCO2e)": info["p10_mt"],
            "P50 (MtCO2e)": info["p50_mt"],
            "P90 (MtCO2e)": info["p90_mt"],
            "Breach risk": info["breach_risk"],
            "Attainment probability": info["attainment_prob"],
            "Assessment": _target_assessment(info),
        }
    )
if diag_rows:
    st.subheader("Target diagnostics")
    diag_df = pd.DataFrame(diag_rows)
    st.dataframe(diag_df, width="stretch")
    for row in diag_rows:
        if "outside the current horizon" in row["Assessment"]:
            st.info(f"{int(row['Target year'])} target is outside the selected horizon, so no breach probability is computed for it.")
        elif "below nearly the entire simulated range" in row["Assessment"]:
            st.warning(f"{int(row['Target year'])} target is far below the modeled pathway, so breach risk is effectively saturated high.")
        elif "above nearly the entire simulated range" in row["Assessment"]:
            st.info(f"{int(row['Target year'])} target is far above the modeled pathway, so breach risk is effectively saturated low.")

if budget_diag["status"] == "too_low":
    st.warning(f"Budget plausibility check: {budget_diag['message']}")
elif budget_diag["status"] == "too_high":
    st.warning(f"Budget plausibility check: {budget_diag['message']}")
elif budget_diag["status"] == "invalid":
    st.error(f"Budget plausibility check: {budget_diag['message']}")

st.subheader("Fan chart")
fan_fig = plot_fan_chart(outputs, cfg)
st.pyplot(fan_fig, clear_figure=True)

st.subheader("Breach calibration helper")
if not target_summary:
    st.info("Add at least one target year and value in Executive Overview to use the calibration helper.")
else:
    helper_years = sorted(target_summary)
    helper_year = st.selectbox("Helper target year", options=helper_years, index=len(helper_years) - 1)
    desired_breach_pct = st.slider("Desired breach risk (%)", 0, 100, 70, 1)
    helper_mode = st.radio(
        "Helper mode",
        ["Solve target threshold", "Solve single lever"],
        horizontal=True,
    )
    desired_breach = desired_breach_pct / 100.0
    helper_info = target_summary[helper_year]

    if not helper_info["in_horizon"]:
        st.info("The selected helper year is outside the current horizon. Extend the horizon or pick another target year.")
    elif not cfg["uncertainty"]["enabled"]:
        st.info("Enable uncertainty to estimate a breach probability and use the calibration helper.")
    elif helper_mode == "Solve target threshold":
        solved_target = solve_target_for_breach(outputs, helper_year, desired_breach)
        st.success(
            f"To target about {desired_breach_pct}% breach in {helper_year}, set the threshold near "
            f"{solved_target['target_mt']:.1f} MtCO2e under the current scenario."
        )
        st.caption(
            f"Current pathway band for {helper_year}: p10={solved_target['p10_mt']:.1f} Mt, "
            f"p50={solved_target['p50_mt']:.1f} Mt, p90={solved_target['p90_mt']:.1f} Mt."
        )
    else:
        lever_label = st.selectbox("Lever to tune", options=list(LEVER_SPECS))
        spec = LEVER_SPECS[lever_label]
        candidate_values = np.arange(spec["min"], spec["max"] + (spec["step"] / 2.0), spec["step"]).tolist()
        lever_result = solve_single_lever_for_breach(
            cfg,
            spec["group"],
            spec["name"],
            candidate_values,
            helper_year,
            desired_breach,
        )
        current_value = float(cfg[spec["group"]].get(spec["name"], 0.0))
        c1, c2, c3 = st.columns(3)
        c1.metric("Current value", f"{current_value:g}")
        c2.metric("Suggested value", f"{float(lever_result['best_value']):g}")
        c3.metric("Solved breach", f"{lever_result['best_breach_risk']:.1%}")
        if lever_result["target_reachable"]:
            st.success(
                f"{lever_label} can get close to {desired_breach_pct}% breach for {helper_year} within its current slider range."
            )
        else:
            st.warning(
                f"{lever_label} alone cannot fully reach {desired_breach_pct}% breach for {helper_year}; "
                f"the reachable range is about {lever_result['risk_min']:.1%} to {lever_result['risk_max']:.1%}."
            )
        lever_df = pd.DataFrame(lever_result["evaluated"])
        st.dataframe(lever_df, width="stretch")

st.subheader("Sensitivity tornado")
with st.spinner("Computing sensitivity tornado..."):
    # Tornado uses multiple scenario reruns; keep spinner explicit for user feedback.
    tornado = compute_sensitivity_tornado(cfg)
if not tornado:
    st.warning("Sensitivity tornado unavailable for current settings. Keep uncertainty enabled and at least one target year inside the horizon.")
else:
    tornado_fig = plot_tornado(tornado)
    st.pyplot(tornado_fig, clear_figure=True)
    st.dataframe(pd.DataFrame(tornado), width="stretch")

events = load_policy_events()
gaps = compute_policy_gap_windows(
    events,
    cfg["horizon"]["start_year"],
    cfg["horizon"]["end_year"],
    ["Domestic", "Industrial", "Services"],
    default_coverage_years=int(cfg["policy"].get("default_coverage_years", 3)),
)
rows = []
for sec, info in gaps.items():
    rows.append(
        {
            "Sector": sec,
            "Gap years count": info["gap_length"],
            "First gap year": info["gap_years"][0] if info["gap_years"] else None,
        }
    )
st.subheader("Policy gap windows")
gap_df = pd.DataFrame(rows)
st.dataframe(gap_df, width="stretch")

if st.checkbox("Auto export report artifacts", value=False, key="export_stress"):
    kpi_df = pd.DataFrame(
        [
            kpis
            | {"budget_exhaustion_year": exh}
            | {
                "budget_plausibility_status": budget_diag["status"],
                "budget_ratio": budget_diag["ratio_budget_to_cumulative"],
            }
        ]
    )
    p1 = export_matplotlib_png(fan_fig, "stress_fan_chart")
    if tornado:
        p2 = export_matplotlib_png(tornado_fig, "stress_tornado")
        st.success(f"Exported: {p2}")
    p3 = export_dataframe_csv(kpi_df, "stress_kpis")
    p4 = export_dataframe_csv(gap_df, "stress_policy_gaps")
    p5 = export_dict_json(cfg, "stress_config")
    st.success(f"Exported: {p1}")
    st.success(f"Exported: {p3}")
    st.success(f"Exported: {p4}")
    st.success(f"Exported: {p5}")
