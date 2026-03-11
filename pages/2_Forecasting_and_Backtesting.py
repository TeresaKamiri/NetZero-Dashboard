import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_energy_data
from src.export import export_dataframe_csv, export_matplotlib_png
from src.forecast import backtest_calibration_summary, build_forecast_long, walk_forward_backtest
from src.viz import plot_forecast_plotly


st.title("Forecasting and Backtesting")
st.caption("Deterministic baseline forecast + rolling-origin evaluation.")

df = load_energy_data()
sectors = sorted(df["Sector"].dropna().unique().tolist())

col1, col2, col3 = st.columns(3)
sector = col1.selectbox("Sector", sectors)
end_uses = sorted(df[df["Sector"] == sector]["End Use"].dropna().unique().tolist())
end_use = col2.selectbox("End use", end_uses)
metric = col3.selectbox("Metric", ["Emissions (ktCO2e)", "Energy Consumption (ktoe)"])

series = df[(df["Sector"] == sector) & (df["End Use"] == end_use)].sort_values("Year")
forecast_long, _ = build_forecast_long(series, metric, forecast_to=2050)
if forecast_long.empty:
    st.warning("No valid data for this selection.")
    st.stop()

fig = plot_forecast_plotly(
    forecast_long, title=f"{sector} - {end_use} ({metric})", y_label=metric
)
st.plotly_chart(fig, width="stretch")

st.subheader("Walk-forward backtest")
bt = walk_forward_backtest(series, metric, min_train_size=6)
if bt.empty:
    st.info("Backtest not available. Need longer historical series.")
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("Final RMSE", f"{bt['RMSE_running'].iloc[-1]:.2f}")
    k2.metric("Final MAE", f"{bt['MAE_running'].iloc[-1]:.2f}")
    k3.metric("Final MAPE", f"{bt['MAPE_running'].iloc[-1]:.2f}%")
    cal = backtest_calibration_summary(bt)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PI80 coverage", f"{cal['coverage_80']:.1%}")
    c2.metric("PI95 coverage", f"{cal['coverage_95']:.1%}")
    c3.metric("PI80 gap", f"{cal['coverage_gap_80']:+.1%}")
    c4.metric("PI95 gap", f"{cal['coverage_gap_95']:+.1%}")
    st.dataframe(bt, width="stretch")

    st.subheader("Calibration visuals")
    fig_bt = plt.figure(figsize=(10, 4.5))
    ax1 = fig_bt.add_subplot(111)
    ax1.plot(bt["Year"], bt["Actual"], label="Actual", color="#1f77b4", linewidth=2)
    ax1.plot(bt["Year"], bt["Predicted"], label="Predicted", color="#ff7f0e", linewidth=2)
    ax1.fill_between(bt["Year"], bt["PI95_lower"], bt["PI95_upper"], color="gray", alpha=0.15, label="PI95")
    ax1.fill_between(bt["Year"], bt["PI80_lower"], bt["PI80_upper"], color="gray", alpha=0.30, label="PI80")
    ax1.set_title("Walk-forward predictions with interval bands")
    ax1.set_xlabel("Year")
    ax1.set_ylabel(metric)
    ax1.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig_bt, clear_figure=False)

    cov = bt[["Year", "Covered80", "Covered95"]].copy()
    # Expanding means make calibration drift over time visible at a glance.
    cov["Coverage80_cum"] = cov["Covered80"].expanding().mean()
    cov["Coverage95_cum"] = cov["Covered95"].expanding().mean()

    fig_cov = plt.figure(figsize=(10, 3.8))
    ax2 = fig_cov.add_subplot(111)
    ax2.plot(cov["Year"], cov["Coverage80_cum"], label="Empirical 80% coverage", color="#2ca02c")
    ax2.plot(cov["Year"], cov["Coverage95_cum"], label="Empirical 95% coverage", color="#9467bd")
    ax2.axhline(0.80, linestyle="--", color="#2ca02c", alpha=0.5, label="Nominal 80%")
    ax2.axhline(0.95, linestyle="--", color="#9467bd", alpha=0.5, label="Nominal 95%")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Cumulative interval coverage by year")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Coverage")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_cov, clear_figure=False)

    if st.checkbox("Auto export report artifacts", value=False, key="export_forecast"):
        # Export the same artifacts shown in-page so reports are reproducible from UI state.
        cal_df = pd.DataFrame([cal])
        p1 = export_dataframe_csv(forecast_long, "forecast_long")
        p2 = export_dataframe_csv(bt, "backtest_table")
        p3 = export_dataframe_csv(cal_df, "backtest_calibration_summary")
        p4 = export_dataframe_csv(cov, "backtest_coverage_by_year")
        p5 = export_matplotlib_png(fig_bt, "backtest_predictions_intervals")
        p6 = export_matplotlib_png(fig_cov, "backtest_coverage_plot")
        st.success(f"Exported: {p1}")
        st.success(f"Exported: {p2}")
        st.success(f"Exported: {p3}")
        st.success(f"Exported: {p4}")
        st.success(f"Exported: {p5}")
        st.success(f"Exported: {p6}")
