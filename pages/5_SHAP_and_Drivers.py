import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.data import load_energy_data


st.title("SHAP and Drivers")
st.caption("Model driver analysis for emissions explainability.")

df = load_energy_data()
required = ["Energy Consumption (ktoe)", "Annual_HDD", "Emissions (ktCO2e)", "Sector"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

work = df.dropna(subset=required).copy()
# Derived intensity is included as an explanatory feature, not the target.
work["Emission Intensity"] = work["Emissions (ktCO2e)"] / work["Energy Consumption (ktoe)"]
work = work.replace([float("inf"), float("-inf")], float("nan")).dropna()

encoded = work.copy()
sector_dummies = pd.get_dummies(encoded["Sector"], prefix="Sector")
encoded = encoded.join(sector_dummies)

feature_cols = ["Energy Consumption (ktoe)", "Annual_HDD", "Emission Intensity"] + [
    c for c in encoded.columns if c.startswith("Sector_")
]
X = encoded[feature_cols]
y = encoded["Emissions (ktCO2e)"]

X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Tree explainer provides local contribution values used in the beeswarm plot.
explainer = shap.Explainer(model)
sv = explainer(X_test)

fig, _ = plt.subplots()
shap.plots.beeswarm(sv, show=False)
st.pyplot(fig, clear_figure=True)
