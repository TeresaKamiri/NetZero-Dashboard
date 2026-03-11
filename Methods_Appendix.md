# Methods Appendix

## A1. Objective and Decision Context

The hybrid dashboard is designed for strategic policy stress-testing of UK net-zero pathways rather than point-forecast precision alone. Outputs are intended to support comparative decisions under uncertainty across policy options and sectoral interventions.

## A2. Data and Scope

- Primary dataset: `data/Merged_Energy_Dataset.csv`
- Core dimensions: `Year`, `Sector`, `End Use`
- Core measures: `Energy Consumption (ktoe)`, `Emissions (ktCO2e)`, `Annual_HDD`
- Forecast/stress-test horizon: configurable (default 2026-2050)
- Sector coverage: normalized to `Domestic`, `Industrial`, `Services`

Policy metadata:

- `data/policy_events.json`
- `data/policy_overlays.csv`

## A3. Model Structure

Two-layer modeling architecture:

1. Baseline Forecast Layer

- ARIMA-based projections for selected time series.
- Walk-forward backtesting used to evaluate one-step predictive performance.

1. Scenario Stress-Test Layer

- Sector pathways simulated over horizon with intervention and macro-lever perturbations.
- Monte Carlo simulation used to estimate distribution of outcomes.
- Aggregated outputs include p10/p50/p90 trajectories and risk KPIs.

## A4. Uncertainty Treatment (Separation to Avoid Double Counting)

Uncertainty channels are explicitly separated:

- Scenario uncertainty:
  - Draws over exogenous levers (`demand`, `rebound`, `delay`, `emissions_intensity`),
  - shared random draws across options for fair comparison.

- Model uncertainty:
  - residual/process perturbation layer applied after scenario path construction,
  - independently toggleable from scenario uncertainty.

Implementation references:

- `src/uncertainty.py`
- `src/stress_engine.py`

## A5. KPI Definitions

Implemented in `src/metrics.py`:

- `breach_risk_2035`: proportion of simulated outcomes above 2035 target.
- `breach_risk_2050`: proportion of simulated outcomes above 2050 target.
- `AttainProb2035/2050`: `1 - breach_risk`.
- `budget_gap_mt`: expected carbon budget exceedance (MtCO2e).
- `cumulative_emissions_p50_mt`: median-pathway cumulative emissions (MtCO2e).
- `budget_exhaustion_year`: first year cumulative median exceeds budget.

## A6. Sensitivity and Failure Diagnostics

- One-at-a-time sensitivity tornado evaluates lever influence.
- Adaptive tornado metric selection avoids saturated-probability blind spots:
  - 2050 breach risk where informative,
  - otherwise 2035 breach risk,
  - otherwise cumulative median emissions.

## A7. Policy Gap Window Method

Gap windows identify uncovered years by sector in the analysis horizon.

- Event-year mapping by policy keyword-to-sector rules.
- Coverage windows support:
  - explicit `start_year`/`end_year` or `duration_years` when provided,
  - fallback default coverage window (`default_coverage_years`) otherwise.

Implementation reference:

- `src/policy.py`

## A8. Backtesting and Calibration Protocol

For each selected series:

1. Rolling-origin walk-forward fitting.
2. One-step-ahead predictions collected each step.
3. Error metrics tracked:

- MAE, RMSE, MAPE.

1. Interval calibration tracked:

- PI80/PI95 empirical coverage,
- coverage gaps vs nominal levels,
- cumulative coverage by year visualized.

Implementation references:

- `pages/2_Forecasting_and_Backtesting.py`
- `src/forecast.py`

## A9. Comparative Option Evaluation

Option scorecard includes:

- Target attainment probability,
- expected regret (Mt),
- dominance frequency under shared scenarios,
- budget gap,
- cost band and delivery-risk band.

Composite ranking score (current implementation):

- 50% AttainProb2050 (normalized),
- 30% DominanceFreq (normalized),
- 20% inverse ExpectedRegret (normalized).

Implementation references:

- `src/evaluation.py`
- `pages/7_Option_Scorecard.py`

## A10. Leakage Controls and Reproducibility

Leakage controls:

- Time-ordered walk-forward backtesting.
- No future observations used in model fit at each forecast step.

Reproducibility:

- Seeded simulation in stress-test engine.
- Exportable assumptions and scenario artifacts.
- Runtime assumptions loaded from `config.json` and then managed in session state.
- One-click outputs for figures/tables under `outputs/`.
- Automated tests under `tests/`.

## A11. Assumption Basis Notes

- `targets.target_values` are scenario thresholds in model units (MtCO2e), not direct statutory UK targets.
- `targets.target_basis` in `config.json` records this explicitly.
- Policy overlay rows after formal policy end dates should be labeled as modeled continuation assumptions.

## A12. Current Limitations

- Sector pathway function remains stylized and may under-represent structural nonlinearity.
- Cost and delivery-risk models are currently categorical, not econometric.
- Policy mapping remains keyword-based; richer ontology could improve specificity.
- ARIMA class can be fragile on short/noisy slices; fallbacks are used.

## A13. Suggested Extensions

1. Add alternative forecasting benchmarks (ETS/Prophet/TBATS) and model selection protocol.
2. Calibrate policy effect sizes using external evidence priors.
3. Introduce explicit capacity constraints and implementation lags by sector.
4. Add decision-threshold logic to convert scorecard outputs into conditional policy triggers.
