# NetZero Dashboard

A Streamlit decision-support dashboard for UK-focused net-zero planning, combining forecasting, stress testing, governance diagnostics, and explainability.

## What This Project Does

- Forecasts emissions trajectories with ARIMA.
- Validates model behavior with rolling-origin backtesting.
- Runs Monte Carlo stress tests with fan charts and breach-risk KPIs.
- Separates scenario uncertainty from model uncertainty to avoid double counting.
- Compares interventions with a common option scorecard (attainment, regret, dominance, cost, delivery risk).
- Produces governance-facing outputs: policy gap windows, conditional recommendations, and exportable artifacts.

## Dashboard Modules

1. Executive Overview (`pages/1_Executive_Overview.py`)
2. Forecasting and Backtesting (`pages/2_Forecasting_and_Backtesting.py`)
3. Stress Test (`pages/3_Stress_Test.py`)
4. Intervention Lab (`pages/4_Intervention_Lab.py`)
5. SHAP and Drivers (`pages/5_SHAP_and_Drivers.py`)
6. Governance and Risk Register (`pages/6_Governance_and_Risk_Register.py`)
7. Option Scorecard (`pages/7_Option_Scorecard.py`)

## Project Structure

```text
NetZero-Dashboard/
|- app.py
|- config.json
|- requirements.txt
|- METHODS_APPENDIX.md
|- data/
|- pages/
|- src/
|- tests/
```

## Run Locally

From `c:\Users\phil\Desktop\Project\Dashboard`:

```powershell
cd .\NetZero-Dashboard
python -m venv nz-env
.\nz-env\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

Core assumptions are defined in `config.json` and loaded at runtime in Executive Overview.

Key groups:
- `meta`: scenario label and data vintage metadata.
- `horizon`: start/end projection years.
- `targets`: stress-test target thresholds and carbon budget.
- `levers`: demand shock, policy delay, rebound, electrification pace, efficiency improvement.
- `interventions`: space heating, cooking electrification, industrial efficiency, services demand management.
- `uncertainty`: scenario/model uncertainty toggles, simulation size, sigmas, and seed.
- `policy`: overlay enable flag and default event coverage years.

Important note:
- `targets.target_values` are scenario assumptions in model units (MtCO2e), not direct statutory UK target values.
- `targets.target_basis` documents this explicitly.

### Config Precedence

1. App starts with `config.json` values.
2. UI changes in Executive Overview and Stress Test update `st.session_state["config"]`.
3. Session values are used for model runs and exports during that app session.
4. `config.json` is not auto-overwritten by UI edits unless explicit save logic is added.

## Data Provenance and Scope

Primary local dataset:
- `data/Merged_Dataset__Energy___Emissions___HDD.csv`
- Current local vintage in config metadata: `UK_1990_2022`

Policy metadata:
- `data/policy_events.json`
- `data/policy_overlays.csv`

## Data Preparation (Optional)

If you want to regenerate the processed modeling dataset used for analysis and EDA:

Inputs (place in `data/preprocessed_datasets/`):
- `ECUK_2024_Consumption_tables.xlsx`
- `2023-provisional-emissions-data-tables.xlsx`
- `ET_7.1_FEB_26.xlsx`

Generate processed output:

```powershell
cd .\NetZero-Dashboard\data\preprocessed_datasets
python preprocess.py
```

Outputs:
- `data/preprocessed_datasets/Processed_Energy_Modeling_Data.csv`

Notes:
- `Annual_HDD` is only available for the years covered by the HDD workbook (later years); earlier years will be null.
- `End Use` splits are derived from `data/Merged_Dataset__Energy___Emissions___HDD.csv` shares, not directly from the xlsx files.

## Unit Conventions

- Emissions series input: `ktCO2e`
- Energy series input: `ktoe`
- Dashboard policy targets and budget: `MtCO2e`
- Conversion used in KPI logic: `ktCO2e -> MtCO2e` via divide by `1000`

## Assumption Calibration Guidance

Recommended calibration workflow:
- Set forecast/backtest baseline first (ARIMA diagnostics, rolling errors, interval coverage).
- Tune uncertainty sigmas (`sigma_demand`, `sigma_rebound`, `sigma_model`, `sigma_emissions_intensity`) to match observed variability and calibration behavior.
- Keep policy overlays evidence-based during official policy windows.
- Mark post-policy-period continuation rows explicitly as modeled assumptions.

## Policy Files and Current Assumption Basis

The model can use:
- `data/policy_overlays.csv`
- `data/policy_events.json`

Current policy/event notes reflected in files:
- ECO4 extension included (confirmed January 23, 2026) to December 31, 2026.
- UK ETS event wording updated to domestic maritime inclusion from July 1, 2026.
- 2027-2030 domestic space-heating overlay rows are labeled as modeled continuation assumptions post GBIS/ECO4 end dates.

## Validation Snapshot

Record latest validation results here for quick reference:
- Last validation date: `TBD`
- Backtest RMSE/MAE/MAPE: `TBD`
- PI80 coverage and gap: `TBD`
- PI95 coverage and gap: `TBD`

## Testing

```powershell
cd .\NetZero-Dashboard
.\nz-env\Scripts\Activate.ps1
pytest -q
```

Tests cover forecast calibration, uncertainty behavior, metrics, and scorecard logic.

## Outputs

When export is enabled in the app, artifacts are written to:

- `outputs/tables/`
- `outputs/figures/`

## Interpretation Guide

- `AttainProb2050`: probability of meeting the configured 2050 threshold.
- `ExpectedRegretMt`: expected excess cumulative emissions vs best option under shared scenarios.
- `DominanceFreq`: share of simulations where an option is best on cumulative emissions.
- Use all three together; do not rank by a single metric alone.

## Limitations and Non-Goals

- Pathway engine is stylized and not a structural macroeconomic model.
- Policy multipliers are scenario assumptions, not causal estimates.
- Cost/delivery bands are categorical, not full cost optimization.
- Outputs support stress testing and comparison, not deterministic policy prediction.
- SHAP/XGBoost explainability can be slow on large datasets; use on smaller subsets if needed.

## Documentation

- Methods and assumptions: `METHODS_APPENDIX.md`
- Rewrite blueprint: `REPORT_REWRITE_FOCUS_AND_BLUEPRINT.md`

## Changelog (Key Data/Policy Updates)

- 2026-02-25: Added ECO4 extension event (confirmed 2026-01-23).
- 2026-02-25: Updated UK ETS domestic maritime event wording (from 2026-07-01).
- 2026-02-25: Relabeled domestic 2027-2030 space-heating overlays as modeled continuation assumptions.
- 2026-02-25: Updated config data vintage metadata to `UK_1990_2022`.

## Notes

This repository is designed for policy-facing clarity and reproducibility.
