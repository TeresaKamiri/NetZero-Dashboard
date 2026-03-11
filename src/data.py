"""Data loading helpers with light schema normalization for dashboard use."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    """Normalize text for case-insensitive, whitespace-tolerant matching."""
    return str(s).strip().lower()


def norm_sector(s: str) -> str:
    """Map sector aliases from source files into canonical labels."""
    t = _norm(s)
    if "industrial" in t:
        return "Industrial"
    if "service" in t or "commercial" in t:
        return "Services"
    if "residential" in t or "domestic" in t:
        return "Domestic"
    return str(s)


def norm_end_use(s: str) -> str:
    """Map end-use aliases from source files into canonical labels."""
    t = _norm(s)
    if "process heat" in t or "process heating" in t:
        return "Process heat"
    if "space heat" in t:
        return "Space heating"
    if "appliance" in t:
        return "Appliances"
    if "hvac" in t or "lighting" in t:
        return "HVAC & Lighting"
    return str(s)


@st.cache_data
def load_energy_data(path: str | None = None) -> pd.DataFrame:
    """Load historical dataset from explicit path, then known project locations."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [
        ROOT / "data" / "Merged_Energy_Dataset.csv",
        ROOT / "Merged_Energy_Dataset.csv",
    ]
    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        raise FileNotFoundError("Energy dataset not found.")

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Sector"] = df["Sector"].astype(str).apply(norm_sector)
    df["End Use"] = df["End Use"].astype(str).apply(norm_end_use)
    return df


@st.cache_data
def load_policy_events(path: str | None = None) -> pd.DataFrame:
    """Load policy events JSON and enrich with parsed date/year columns."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [
        ROOT / "data" / "policy_events.json",
        ROOT / "policy_events.json",
    ]
    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        return pd.DataFrame(columns=["date", "event", "source", "year"])

    with open(file_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "event", "source", "year"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    return df.dropna(subset=["year"]).copy()


@st.cache_data
def load_policy_overlays(path: str | None = None) -> pd.DataFrame:
    """Load policy multipliers used to modulate modeled trajectories."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [
        ROOT / "data" / "policy_overlays.csv",
        ROOT / "policy_overlays.csv",
    ]
    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"sector", "end_use", "year", "multiplier_emissions"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    if "multiplier_energy" not in df.columns:
        df["multiplier_energy"] = df["multiplier_emissions"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["sector_norm"] = df["sector"].astype(str).apply(norm_sector)
    df["end_use_norm"] = df["end_use"].astype(str).apply(norm_end_use)
    return df.dropna(subset=["year"]).copy()
