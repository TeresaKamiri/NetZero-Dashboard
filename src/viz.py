"""Reusable plotting utilities for Streamlit modules."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_fan_chart(outputs: dict, config: dict, comparison_outputs: dict | None = None):
    """Plot uncertainty fan chart with optional intervention comparison median."""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    years = outputs["years"]
    ax.fill_between(years, outputs["p10"] / 1000.0, outputs["p90"] / 1000.0, color="gray", alpha=0.2, label="P10-P90")
    ax.plot(years, outputs["p50"] / 1000.0, color="#1f77b4", linewidth=2, label="Base median")
    if comparison_outputs is not None:
        ax.plot(
            years,
            comparison_outputs["p50"] / 1000.0,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label="Intervention median",
        )
    for yr, val in config["targets"]["target_values"].items():
        ax.axhline(float(val), color="black", linestyle="--", alpha=0.45)
        ax.text(years[0], float(val), f"{yr} target", fontsize=8, va="bottom")
    ax.set_title("Stress-Test Emissions Pathway")
    ax.set_ylabel("MtCO2e")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_tornado(rows: list[dict]):
    """Plot signed one-at-a-time sensitivity bars for key levers."""
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    if not rows:
        ax.text(0.5, 0.5, "Uncertainty must be enabled", ha="center", va="center")
        ax.set_axis_off()
        plt.close(fig)
        return fig
    labels = [r["lever"] for r in rows]
    values = [r.get("delta_value", r.get("delta_risk_pp", 0.0)) for r in rows]
    metric = rows[0].get("metric", "2050 breach risk")
    unit = rows[0].get("unit", "pp")
    y = np.arange(len(labels))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    xlabel = "Delta (pp)" if unit == "pp" else "Delta (Mt)"
    ax.set_xlabel(xlabel)
    ax.set_title(f"Sensitivity Tornado ({metric})")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_forecast_plotly(long_df, title: str, y_label: str):
    """Render historical and forecast trajectories in a Plotly line chart."""
    fig = go.Figure()
    for typ in long_df["Type"].unique():
        cur = long_df[long_df["Type"] == typ]
        fig.add_trace(go.Scatter(x=cur["Year"], y=cur["Value"], mode="lines+markers", name=typ))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title=y_label)
    return fig
