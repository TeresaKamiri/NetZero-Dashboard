import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import evaluate_options
from src.export import export_dataframe_csv, export_dict_json, export_matplotlib_png


st.title("Option Scorecard")
st.caption("Target attainment, expected regret, dominance, and delivery trade-offs.")

if "config" not in st.session_state:
    st.warning("Initialize assumptions in Executive Overview first.")
    st.stop()

cfg = st.session_state["config"]

options = [
    {
        "name": "Status quo",
        "patch": {},
        "cost_band": "Low",
        "delivery_risk": "Low",
    },
    {
        "name": "Heat-focused",
        "patch": {"interventions": {"space_heating_reduction_pct": 15.0}},
        "cost_band": "Medium",
        "delivery_risk": "Medium",
    },
    {
        "name": "Efficiency-first",
        "patch": {"interventions": {"industrial_efficiency_push_pct": 10.0}},
        "cost_band": "Medium",
        "delivery_risk": "Low",
    },
    {
        "name": "Balanced bundle",
        "patch": {
            "interventions": {
                "space_heating_reduction_pct": 10.0,
                "industrial_efficiency_push_pct": 7.0,
                "services_demand_management_pct": 5.0,
            },
            "levers": {"policy_delay_years": 0},
        },
        "cost_band": "High",
        "delivery_risk": "Medium",
    },
]

scorecard, detail = evaluate_options(cfg, options)
if scorecard.empty:
    st.info("No scorecard available.")
    st.stop()

ranked = scorecard.copy()
# Min-max normalization keeps metrics on comparable [0,1] scales for aggregation.
for col in ["AttainProb2050", "DominanceFreq"]:
    lo = ranked[col].min()
    hi = ranked[col].max()
    ranked[f"{col}_norm"] = 0.0 if hi == lo else (ranked[col] - lo) / (hi - lo)
for col in ["ExpectedRegretMt"]:
    lo = ranked[col].min()
    hi = ranked[col].max()
    ranked[f"{col}_norm"] = 0.0 if hi == lo else (ranked[col] - lo) / (hi - lo)

ranked["CompositeScore"] = (
    # Weighting prioritizes long-run attainment, then robustness, then regret.
    0.50 * ranked["AttainProb2050_norm"]
    + 0.30 * ranked["DominanceFreq_norm"]
    + 0.20 * (1.0 - ranked["ExpectedRegretMt_norm"])
)
ranked = ranked.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
ranked["Rank"] = ranked.index + 1

st.subheader("Ranked options")
st.dataframe(
    ranked[
        [
            "Rank",
            "Option",
            "AttainProb2035",
            "AttainProb2050",
            "ExpectedRegretMt",
            "DominanceFreq",
            "BudgetGapMt",
            "CostBand",
            "DeliveryRiskBand",
            "CompositeScore",
        ]
    ],
    width="stretch",
)

best = ranked.iloc[0]
st.success(
    f"Top option: {best['Option']} | 2050 attainment={best['AttainProb2050']:.1%}, "
    f"expected regret={best['ExpectedRegretMt']:.1f} Mt, dominance={best['DominanceFreq']:.1%}."
)

st.subheader("Attainment vs regret")
plot_df = ranked.set_index("Option")[["AttainProb2050", "ExpectedRegretMt", "DominanceFreq"]]
st.bar_chart(plot_df)

fig_rank = plt.figure(figsize=(10, 4.5))
ax1 = fig_rank.add_subplot(111)
x = range(len(ranked))
ax1.bar(x, ranked["AttainProb2050"], label="Attainment 2050")
ax1.bar(x, ranked["DominanceFreq"], bottom=ranked["AttainProb2050"], label="Dominance")
ax1.set_xticks(list(x))
ax1.set_xticklabels(ranked["Option"], rotation=15, ha="right")
ax1.set_ylim(0, 2.05)
ax1.set_ylabel("Probability scale")
ax1.set_title("Option comparison: attainment and dominance")
ax1.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig_rank, clear_figure=False)

fig_scatter = plt.figure(figsize=(8, 4.5))
ax2 = fig_scatter.add_subplot(111)

# If 2050 attainment is saturated (e.g., all zeros), use fallback y metric.
y_col = "AttainProb2050"
y_label = "2050 attainment probability"
y_title_suffix = "attainment (2050)"
if ranked["AttainProb2050"].nunique() <= 1:
    if ranked["AttainProb2035"].nunique() > 1:
        y_col = "AttainProb2035"
        y_label = "2035 attainment probability"
        y_title_suffix = "attainment (2035 fallback)"
    else:
        y_col = "DominanceFreq"
        y_label = "Dominance frequency"
        y_title_suffix = "dominance fallback"
    st.info("2050 attainment is saturated across options; using fallback y-axis for a more informative frontier view.")

ax2.scatter(ranked["ExpectedRegretMt"], ranked[y_col], s=120, alpha=0.9)
for i, (_, r) in enumerate(ranked.iterrows()):
    xoff = 6
    yoff = 6 if i % 2 == 0 else -10
    ax2.annotate(r["Option"], (r["ExpectedRegretMt"], r[y_col]), fontsize=8, xytext=(xoff, yoff), textcoords="offset points")
ax2.set_xlabel("Expected regret (Mt)")
ax2.set_ylabel(y_label)
ax2.set_title(f"Trade-off frontier: regret vs {y_title_suffix}")
plt.tight_layout()
st.pyplot(fig_scatter, clear_figure=False)

if st.checkbox("Auto export report artifacts", value=False, key="export_scorecard"):
    p1 = export_dataframe_csv(scorecard, "option_scorecard_raw")
    p2 = export_dataframe_csv(ranked, "option_scorecard_ranked")
    p3 = export_matplotlib_png(fig_rank, "option_attainment_dominance")
    p4 = export_matplotlib_png(fig_scatter, "option_regret_attainment_frontier")
    p5 = export_dict_json({"base_config": cfg, "options": options}, "option_scorecard_assumptions")
    st.success(f"Exported: {p1}")
    st.success(f"Exported: {p2}")
    st.success(f"Exported: {p3}")
    st.success(f"Exported: {p4}")
    st.success(f"Exported: {p5}")
    for opt_name, opt_df in detail.items():
        p = export_dataframe_csv(opt_df, f"scorecard_detail_{opt_name.replace(' ', '_').lower()}")
        st.success(f"Exported: {p}")
