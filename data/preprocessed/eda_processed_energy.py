from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = [
        "Year",
        "Sector",
        "End Use",
        "Energy Consumption (ktoe)",
        "Emissions (ktCO2e)",
        "Annual_HDD",
        "Energy_per_HDD",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for c in ["Sector", "End Use"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["Energy Consumption (ktoe)", "Emissions (ktCO2e)", "Annual_HDD", "Energy_per_HDD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = {
        "rows": len(df),
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "n_sectors": int(df["Sector"].nunique(dropna=True)),
        "n_end_uses": int(df["End Use"].nunique(dropna=True)),
        "missing_energy": int(df["Energy Consumption (ktoe)"].isna().sum()),
        "missing_emissions": int(df["Emissions (ktCO2e)"].isna().sum()),
        "missing_hdd": int(df["Annual_HDD"].isna().sum()),
        "missing_energy_per_hdd": int(df["Energy_per_HDD"].isna().sum()),
    }
    profile_df = pd.DataFrame([summary])

    stats_df = (
        df[["Energy Consumption (ktoe)", "Emissions (ktCO2e)", "Annual_HDD", "Energy_per_HDD"]]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .T
    )
    return profile_df, stats_df


def sector_enduse_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_year = (
        df.groupby(["Year", "Sector"], as_index=False)[["Energy Consumption (ktoe)", "Emissions (ktCO2e)"]]
        .sum()
        .sort_values(["Year", "Sector"])
    )
    enduse_totals = (
        df.groupby(["Sector", "End Use"], as_index=False)[["Energy Consumption (ktoe)", "Emissions (ktCO2e)"]]
        .sum()
        .sort_values(["Sector", "Energy Consumption (ktoe)"], ascending=[True, False])
    )
    return sector_year, enduse_totals


def add_intensity_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Emissions_per_Energy"] = np.where(
        out["Energy Consumption (ktoe)"] > 0,
        out["Emissions (ktCO2e)"] / out["Energy Consumption (ktoe)"],
        np.nan,
    )
    return out


def build_yoy_table(df: pd.DataFrame) -> pd.DataFrame:
    t = (
        df.groupby("Year", as_index=False)[["Energy Consumption (ktoe)", "Emissions (ktCO2e)"]]
        .sum()
        .sort_values("Year")
    )
    t["Energy_YoY_pct"] = t["Energy Consumption (ktoe)"].pct_change() * 100.0
    t["Emissions_YoY_pct"] = t["Emissions (ktCO2e)"].pct_change() * 100.0
    return t


def save_trend_plots(df: pd.DataFrame, outdir: Path) -> None:
    annual = (
        df.groupby("Year", as_index=False)[["Energy Consumption (ktoe)", "Emissions (ktCO2e)"]]
        .sum()
        .sort_values("Year")
    )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(annual["Year"], annual["Energy Consumption (ktoe)"], color="#1f77b4", label="Energy (ktoe)")
    ax1.set_ylabel("Energy (ktoe)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(annual["Year"], annual["Emissions (ktCO2e)"], color="#d62728", label="Emissions (ktCO2e)")
    ax2.set_ylabel("Emissions (ktCO2e)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Annual Energy and Emissions Trend")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(outdir / "trend_energy_emissions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    sector = (
        df.groupby(["Year", "Sector"], as_index=False)[["Energy Consumption (ktoe)", "Emissions (ktCO2e)"]]
        .sum()
        .sort_values(["Year", "Sector"])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=sector,
        x="Year",
        y="Emissions (ktCO2e)",
        hue="Sector",
        marker="o",
        ax=ax,
    )
    ax.set_title("Sector Emissions Trend")
    fig.tight_layout()
    fig.savefig(outdir / "trend_sector_emissions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plots(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["Energy Consumption (ktoe)"], bins=30, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Distribution: Energy Consumption (ktoe)")
    sns.histplot(df["Emissions (ktCO2e)"], bins=30, kde=True, ax=axes[1], color="#d62728")
    axes[1].set_title("Distribution: Emissions (ktCO2e)")
    fig.tight_layout()
    fig.savefig(outdir / "distribution_energy_emissions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Sector", y="Energy_per_HDD", ax=ax)
    ax.set_title("Energy_per_HDD by Sector")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(outdir / "box_energy_per_hdd_by_sector.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_plot(df: pd.DataFrame, outdir: Path) -> None:
    x = add_intensity_fields(df)
    corr_cols = ["Energy Consumption (ktoe)", "Emissions (ktCO2e)", "Annual_HDD", "Energy_per_HDD", "Emissions_per_Energy"]
    corr = x[corr_cols].corr(numeric_only=True)
    corr.to_csv(outdir / "correlation_matrix.csv", index=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_ranked_tables(df: pd.DataFrame, outdir: Path) -> None:
    top_energy = (
        df.groupby(["Sector", "End Use"], as_index=False)["Energy Consumption (ktoe)"]
        .sum()
        .sort_values("Energy Consumption (ktoe)", ascending=False)
    )
    top_emissions = (
        df.groupby(["Sector", "End Use"], as_index=False)["Emissions (ktCO2e)"]
        .sum()
        .sort_values("Emissions (ktCO2e)", ascending=False)
    )
    top_energy.to_csv(outdir / "ranked_end_uses_by_energy.csv", index=False)
    top_emissions.to_csv(outdir / "ranked_end_uses_by_emissions.csv", index=False)


def run_eda(input_csv: Path, outdir: Path) -> None:
    sns.set_theme(style="whitegrid")
    out = ensure_outdir(outdir)
    df = load_dataset(input_csv)

    profile_df, stats_df = build_profile(df)
    profile_df.to_csv(out / "profile_summary.csv", index=False)
    stats_df.to_csv(out / "numeric_stats.csv", index=True)

    sector_year, enduse_totals = sector_enduse_tables(df)
    sector_year.to_csv(out / "sector_year_totals.csv", index=False)
    enduse_totals.to_csv(out / "sector_end_use_totals.csv", index=False)

    yoy = build_yoy_table(df)
    yoy.to_csv(out / "annual_yoy_totals.csv", index=False)

    save_trend_plots(df, out)
    save_distribution_plots(df, out)
    save_correlation_plot(df, out)
    save_ranked_tables(df, out)

    print(f"EDA complete. Output folder: {out.resolve()}")
    print(f"Rows: {len(df)}, Years: {int(df['Year'].min())}-{int(df['Year'].max())}")
    print(f"Sectors: {sorted(df['Sector'].dropna().unique().tolist())}")
    print(f"End uses: {df['End Use'].nunique(dropna=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on Processed_Energy_Modeling_Data.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Merged_Dataset__Energy___Emissions___HDD.csv"),
        help="Path to processed dataset CSV",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("eda_outputs"),
        help="Directory to save EDA outputs",
    )
    args = parser.parse_args()
    run_eda(args.input, args.outdir)


if __name__ == "__main__":
    main()
