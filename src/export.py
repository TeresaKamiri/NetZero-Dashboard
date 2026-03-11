"""Export helpers for report tables, figures, and config snapshots."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _ensure_dir(kind: str) -> Path:
    """Create output subdirectory if needed and return its path."""
    d = ROOT / "outputs" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp() -> str:
    """Return timestamp suffix for deterministic, collision-resistant file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_dataframe_csv(df: pd.DataFrame, name: str, kind: str = "tables") -> str:
    """Export a DataFrame to CSV and return the absolute output path."""
    out = _ensure_dir(kind) / f"{name}_{_stamp()}.csv"
    df.to_csv(out, index=False)
    return str(out)


def export_dict_json(data: dict[str, Any], name: str, kind: str = "tables") -> str:
    """Export a dictionary payload as formatted JSON and return output path."""
    import json

    out = _ensure_dir(kind) / f"{name}_{_stamp()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(out)


def export_matplotlib_png(fig, name: str, kind: str = "figures") -> str:
    """Export a matplotlib figure as PNG and return output path."""
    out = _ensure_dir(kind) / f"{name}_{_stamp()}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return str(out)
