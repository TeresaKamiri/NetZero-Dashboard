import copy
import json
from pathlib import Path

import streamlit as st


st.title("Executive Overview")
st.caption("Set baseline assumptions used across hybrid modules.")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

FALLBACK_CONFIG = {
    "meta": {
        "scenario_name": "Baseline",
        "created_at": "",
        "data_vintage": "",
        "model_version": "1.0.0",
    },
    "horizon": {"start_year": 2026, "end_year": 2050},
    "baseline_year": 2022,
    "levers": {
        "demand_growth_shock_pct": 0.0,
        "electrification_pace": 0.0,
        "efficiency_improvement_pct": 0.0,
        "policy_delay_years": 0,
        "rebound_intensity": 0.0,
    },
    "interventions": {
        "space_heating_reduction_pct": 0.0,
        "cooking_electrification_boost": 0.0,
        "industrial_efficiency_push_pct": 0.0,
        "services_demand_management_pct": 0.0,
    },
    "targets": {
        "target_values": {"2035": 100.0, "2050": 0.0},
        "carbon_budget_total": 1500.0,
    },
    "uncertainty": {
        "enabled": True,
        "scenario_uncertainty_enabled": True,
        "model_uncertainty_enabled": True,
        "n_sims": 300,
        "sigma_demand": 0.03,
        "sigma_rebound": 0.08,
        "sigma_model": 0.015,
        "sigma_emissions_intensity": 0.02,
        "seed": 42,
    },
    "policy": {"apply_overlays": True, "default_coverage_years": 3},
}


def _deep_merge(base: dict, override: dict) -> dict:
    # Recursive merge preserves defaults for keys missing in older config files.
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_initial_config() -> dict:
    if not CONFIG_PATH.exists():
        return copy.deepcopy(FALLBACK_CONFIG)
    try:
        # Merge disk config onto fallback to maintain backward-compatible schema evolution.
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return copy.deepcopy(FALLBACK_CONFIG)
        return _deep_merge(FALLBACK_CONFIG, loaded)
    except Exception:
        return copy.deepcopy(FALLBACK_CONFIG)


if "config" not in st.session_state:
    # Central session-state config is the shared contract across all dashboard pages.
    st.session_state["config"] = _load_initial_config()

cfg = st.session_state["config"]

col1, col2, col3 = st.columns(3)
cfg["horizon"]["start_year"] = int(
    col1.number_input("Start year", value=cfg["horizon"]["start_year"])
)
cfg["horizon"]["end_year"] = int(
    col2.number_input("End year", value=cfg["horizon"]["end_year"])
)
cfg["baseline_year"] = int(col3.number_input("Baseline year", value=cfg["baseline_year"]))

st.subheader("Targets")
t1, t2, t3 = st.columns(3)
cfg["targets"]["target_values"]["2035"] = float(
    t1.number_input(
        "2035 target (MtCO2e)", value=float(cfg["targets"]["target_values"]["2035"])
    )
)
cfg["targets"]["target_values"]["2050"] = float(
    t2.number_input(
        "2050 target (MtCO2e)", value=float(cfg["targets"]["target_values"]["2050"])
    )
)
cfg["targets"]["carbon_budget_total"] = float(
    t3.number_input("Budget total (MtCO2e)", value=float(cfg["targets"]["carbon_budget_total"]))
)

st.session_state["config"] = cfg
st.success("Scenario config loaded and saved to session.")
