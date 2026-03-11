import copy

import numpy as np
import streamlit as st

from src.metrics import compute_kpis
from src.stress_engine import run_scenario
from src.viz import plot_fan_chart


st.title("Intervention Lab")
st.caption("Compare baseline vs accelerated policy strategy using shared assumptions.")

if "config" not in st.session_state:
    st.warning("Initialize assumptions in Executive Overview.")
    st.stop()

base_cfg = copy.deepcopy(st.session_state["config"])
test_cfg = copy.deepcopy(base_cfg)

with st.sidebar:
    st.subheader("Intervention Controls")
    # Support delta and absolute modes so users can test relative uplift or fixed targets.
    mode = st.radio("Control mode", ["Delta (pp)", "Absolute target (%)"], index=0)
    if mode == "Delta (pp)":
        hp = st.slider("Domestic heating delta (pp)", -10.0, 30.0, 5.0, 0.5)
        ind = st.slider("Industrial efficiency delta (pp)", -10.0, 20.0, 5.0, 0.5)
        srv = st.slider("Services demand management delta (pp)", -10.0, 15.0, 3.0, 0.5)
    else:
        hp = st.slider("Domestic heating target (%)", 0.0, 30.0, max(5.0, float(base_cfg["interventions"]["space_heating_reduction_pct"])), 0.5)
        ind = st.slider("Industrial efficiency target (%)", 0.0, 20.0, max(5.0, float(base_cfg["interventions"]["industrial_efficiency_push_pct"])), 0.5)
        srv = st.slider("Services demand management target (%)", 0.0, 15.0, max(3.0, float(base_cfg["interventions"]["services_demand_management_pct"])), 0.5)

if mode == "Delta (pp)":
    test_cfg["interventions"]["space_heating_reduction_pct"] = max(
        0.0, float(base_cfg["interventions"]["space_heating_reduction_pct"]) + hp
    )
    test_cfg["interventions"]["industrial_efficiency_push_pct"] = max(
        0.0, float(base_cfg["interventions"]["industrial_efficiency_push_pct"]) + ind
    )
    test_cfg["interventions"]["services_demand_management_pct"] = max(
        0.0, float(base_cfg["interventions"]["services_demand_management_pct"]) + srv
    )
else:
    test_cfg["interventions"]["space_heating_reduction_pct"] = float(hp)
    test_cfg["interventions"]["industrial_efficiency_push_pct"] = float(ind)
    test_cfg["interventions"]["services_demand_management_pct"] = float(srv)

base_out = run_scenario(base_cfg)
test_out = run_scenario(test_cfg)
base_k = compute_kpis(base_out, base_cfg)
test_k = compute_kpis(test_out, test_cfg)

st.pyplot(plot_fan_chart(base_out, base_cfg, comparison_outputs=test_out), clear_figure=True)

if np.allclose(base_out["p50"], test_out["p50"]):
    # Guardrail to catch cases where UI changes do not propagate to a distinct trajectory.
    st.warning("Intervention and baseline trajectories are identical under current settings. Increase intervention levels or adjust stress levers.")

c1, c2, c3 = st.columns(3)
c1.metric("2050 breach risk", f"{test_k['breach_risk_2050']:.1%}", f"{(base_k['breach_risk_2050']-test_k['breach_risk_2050']):+.1%}")
c2.metric("Budget gap (Mt)", f"{test_k['budget_gap_mt']:.1f}", f"{(base_k['budget_gap_mt']-test_k['budget_gap_mt']):+.1f}")
c3.metric("Cum emissions p50 (Mt)", f"{test_k['cumulative_emissions_p50_mt']:.1f}", f"{(base_k['cumulative_emissions_p50_mt']-test_k['cumulative_emissions_p50_mt']):+.1f}")
