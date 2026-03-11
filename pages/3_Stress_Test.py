import streamlit as st
import pandas as pd

from src.data import load_policy_events
from src.export import export_dataframe_csv, export_dict_json, export_matplotlib_png
from src.metrics import (
    assess_budget_plausibility,
    calculate_budget_exhaustion_year,
    compute_kpis,
    compute_sensitivity_tornado,
)
from src.policy import compute_policy_gap_windows
from src.stress_engine import run_scenario
from src.viz import plot_fan_chart, plot_tornado


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

k1, k2, k3, k4 = st.columns(4)
k1.metric("Breach risk 2035", f"{kpis['breach_risk_2035']:.1%}")
k2.metric("Breach risk 2050", f"{kpis['breach_risk_2050']:.1%}")
k3.metric("Budget gap", f"{kpis['budget_gap_mt']:.1f} Mt")
k4.metric("Budget exhaustion", str(exh))

if budget_diag["status"] == "too_low":
    st.warning(f"Budget plausibility check: {budget_diag['message']}")
elif budget_diag["status"] == "too_high":
    st.warning(f"Budget plausibility check: {budget_diag['message']}")
elif budget_diag["status"] == "invalid":
    st.error(f"Budget plausibility check: {budget_diag['message']}")

st.subheader("Fan chart")
fan_fig = plot_fan_chart(outputs, cfg)
st.pyplot(fan_fig, clear_figure=True)

st.subheader("Sensitivity tornado")
with st.spinner("Computing sensitivity tornado..."):
    # Tornado uses multiple scenario reruns; keep spinner explicit for user feedback.
    tornado = compute_sensitivity_tornado(cfg)
if not tornado:
    st.warning("Sensitivity tornado unavailable for current settings. Keep uncertainty enabled and horizon including 2050.")
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
    kpi_df = pd.DataFrame([
        kpis
        | {"budget_exhaustion_year": exh}
        | {
            "budget_plausibility_status": budget_diag["status"],
            "budget_ratio": budget_diag["ratio_budget_to_cumulative"],
        }
    ])
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
