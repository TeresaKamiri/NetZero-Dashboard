"""Policy mapping, coverage-gap detection, and governance register helpers."""

import pandas as pd


POLICY_SECTOR_MAP = {
    "BUS": ["Domestic"],
    "Boiler Upgrade Scheme": ["Domestic"],
    "GBIS": ["Domestic"],
    "Great British Insulation Scheme": ["Domestic"],
    "PSDS": ["Services"],
    "Public Sector Decarbonisation Scheme": ["Services"],
    "UK ETS": ["Industrial", "Services"],
    "maritime": ["Industrial"],
    "CBAM": ["Industrial"],
}


def infer_affected_sectors(event_name: str) -> list[str]:
    """Infer likely affected sectors from event name keyword matches."""
    text = str(event_name).lower()
    hits = set()
    for k, vals in POLICY_SECTOR_MAP.items():
        if k.lower() in text:
            hits.update(vals)
    return list(hits) if hits else ["All"]


def add_affected_sectors(policy_events_df: pd.DataFrame) -> pd.DataFrame:
    """Attach an inferred affected-sector list to each policy event row."""
    if policy_events_df.empty:
        return policy_events_df
    out = policy_events_df.copy()
    out["affected_sectors"] = out["event"].apply(infer_affected_sectors)
    return out


def compute_policy_gap_windows(
    policy_events_df: pd.DataFrame,
    horizon_start: int,
    horizon_end: int,
    sectors: list[str],
    default_coverage_years: int = 1,
) -> dict:
    """Compute uncovered years per sector across the planning horizon."""
    def _safe_int(v):
        try:
            if pd.isna(v):
                return None
            return int(v)
        except Exception:
            return None

    def _event_window(row) -> tuple[int, int]:
        event_year = _safe_int(row.get("year", None))
        if event_year is None:
            return horizon_start, horizon_start - 1  # empty window

        # Preferred explicit window from data file
        start = _safe_int(row.get("start_year", None))
        end = _safe_int(row.get("end_year", None))
        duration = _safe_int(row.get("duration_years", None))

        if start is None:
            start = event_year
        if end is None:
            if duration is not None and duration > 0:
                end = start + duration - 1
            else:
                span = max(1, int(default_coverage_years))
                end = start + span - 1
        return start, end

    years = set(range(horizon_start, horizon_end + 1))
    gaps = {}
    events = add_affected_sectors(policy_events_df)
    for sector in sectors:
        covered = set()
        for _, row in events.iterrows():
            affected = row.get("affected_sectors", ["All"])
            if sector in affected or "All" in affected:
                start, end = _event_window(row)
                if start <= end:
                    start_clip = max(start, horizon_start)
                    end_clip = min(end, horizon_end)
                    if start_clip <= end_clip:
                        covered.update(range(start_clip, end_clip + 1))
        gap_years = sorted(list(years - covered))
        gaps[sector] = {"gap_years": gap_years, "gap_length": len(gap_years)}
    return gaps


def build_risk_register(kpis: dict, policy_gaps: dict, exhaustion_year) -> pd.DataFrame:
    """Build a simple RAG-style risk register from model outputs and policy gaps."""
    rows = []
    breach_2050 = float(kpis.get("breach_risk_2050", 0.0))
    rag = "Green"
    if breach_2050 > 0.6:
        rag = "Red"
    elif breach_2050 > 0.3:
        rag = "Amber"

    rows.append(
        {
            "Risk": "2050 target breach",
            "RAG": rag,
            "Owner": "Net Zero Strategy Unit",
            "Leading indicator": "Breach risk (2050)",
            "Current value": f"{breach_2050:.1%}",
        }
    )
    rows.append(
        {
            "Risk": "Carbon budget exhaustion",
            "RAG": "Red" if isinstance(exhaustion_year, int) else "Green",
            "Owner": "Treasury + DESNZ",
            "Leading indicator": "Exhaustion year",
            "Current value": str(exhaustion_year),
        }
    )
    for sector, info in policy_gaps.items():
        rows.append(
            {
                "Risk": f"{sector} policy coverage gap",
                "RAG": "Red" if info["gap_length"] > 8 else "Amber" if info["gap_length"] > 4 else "Green",
                "Owner": f"{sector} delivery lead",
                "Leading indicator": "Gap years count",
                "Current value": str(int(info["gap_length"])),
            }
        )
    out = pd.DataFrame(rows)
    out["Current value"] = out["Current value"].astype(str)
    return out
