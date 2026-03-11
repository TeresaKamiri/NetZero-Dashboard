import streamlit as st

from src.data import load_energy_data, load_policy_events


st.set_page_config(page_title="Hybrid Net-Zero Dashboard", page_icon="🌍", layout="wide")

st.title("Hybrid Net-Zero Dashboard")
st.caption("Forecasting + stress testing + policy diagnostics + explainability")

df = load_energy_data()
events = load_policy_events()

latest_year = int(df["Year"].max())
latest_mt = float(df[df["Year"] == latest_year]["Emissions (ktCO2e)"].sum() / 1000.0)
prev_year = latest_year - 1
prev_slice = df[df["Year"] == prev_year]
if not prev_slice.empty:
    # Surface both relative and absolute movement to avoid misleading small-base percentages.
    prev_mt = float(prev_slice["Emissions (ktCO2e)"].sum() / 1000.0)
    yoy_pct = ((latest_mt - prev_mt) / prev_mt * 100.0) if prev_mt != 0 else None
    yoy_abs_mt = latest_mt - prev_mt
else:
    yoy_pct = None
    yoy_abs_mt = None

m1, m2, m3 = st.columns(3)
m1.metric("Latest historical year", latest_year)
if yoy_pct is None:
    m2.metric("Latest emissions", f"{latest_mt:.1f} MtCO2e")
else:
    m2.metric(
        "Latest emissions",
        f"{latest_mt:.1f} MtCO2e",
        delta=f"{yoy_pct:+.1f}% ({yoy_abs_mt:+.1f} Mt) vs {prev_year}",
    )
m3.metric("Policy milestones tracked", int(len(events)))

st.markdown("### Modules")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("1. Forecasting + Backtesting")
    st.write("ARIMA projections and rolling-origin validation.")
    if st.button("Open module", key="to_forecast", width="stretch"):
        # Use explicit page path to keep navigation robust if menu order changes.
        st.switch_page("pages/2_Forecasting_and_Backtesting.py")

with c2:
    st.subheader("2. Stress Test")
    st.write("Monte Carlo breach risk, fan chart, and sensitivity.")
    if st.button("Open stress test", key="to_stress", width="stretch"):
        st.switch_page("pages/3_Stress_Test.py")

with c3:
    st.subheader("3. Governance")
    st.write("Policy gap windows and risk register framing.")
    if st.button("Open governance", key="to_gov", width="stretch"):
        st.switch_page("pages/6_Governance_and_Risk_Register.py")

with c4:
    st.subheader("4. Option Scorecard")
    st.write("Attainment, regret, dominance, and delivery trade-offs.")
    if st.button("Open scorecard", key="to_scorecard", width="stretch"):
        st.switch_page("pages/7_Option_Scorecard.py")

st.info("Start with Forecasting, then calibrate assumptions in Stress Test.")
