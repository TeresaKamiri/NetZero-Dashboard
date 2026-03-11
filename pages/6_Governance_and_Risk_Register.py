import streamlit as st

from src.data import load_policy_events
from src.metrics import assess_budget_plausibility, calculate_budget_exhaustion_year, compute_kpis
from src.policy import build_risk_register, compute_policy_gap_windows
from src.stress_engine import run_scenario


st.title("Governance and Risk Register")
st.caption("Translate model outputs into policy-facing governance actions.")

if "config" not in st.session_state:
    st.warning("Initialize scenario config in Executive Overview first.")
    st.stop()

cfg = st.session_state["config"]
out = run_scenario(cfg)
kpis = compute_kpis(out, cfg)
exh = calculate_budget_exhaustion_year(out, cfg)
budget_diag = assess_budget_plausibility(out, cfg)

events = load_policy_events()
gaps = compute_policy_gap_windows(
    events,
    cfg["horizon"]["start_year"],
    cfg["horizon"]["end_year"],
    ["Domestic", "Industrial", "Services"],
    default_coverage_years=int(cfg.get("policy", {}).get("default_coverage_years", 3)),
)

register = build_risk_register(kpis, gaps, exh)
st.dataframe(register, width="stretch")

if budget_diag["status"] != "ok":
    st.warning(f"Budget plausibility check: {budget_diag['message']}")

st.subheader("Conditional recommendations")
# Thresholds intentionally align with simple governance trigger bands for fast triage.
if kpis["breach_risk_2050"] > 0.5:
    st.error("If 2050 breach risk exceeds 50%, trigger accelerated intervention package and tighten delivery governance.")
elif kpis["breach_risk_2050"] > 0.2:
    st.warning("If 2050 breach risk is between 20%-50%, prioritize gap closure in highest-fragility sectors.")
else:
    st.success("If 2050 breach risk is below 20%, maintain current pathway and monitor leading indicators quarterly.")
