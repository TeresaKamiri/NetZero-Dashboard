import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_year_from_header(value):
    """Extract 4-digit year from header tokens like '2023 [p]'."""
    if pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def _read_excel_raw(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name, header=None)


def _extract_emissions_from_table1a(file_path):
    """
    Extract emissions (MtCO2e) from DESNZ Table1a_GHG and convert to ktCO2e.
    Returns: Year, Sector, Emissions (ktCO2e)
    """
    df = _read_excel_raw(file_path, "Table1a_GHG")

    header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().eq("TES Sector")]
    if len(header_idx) == 0:
        raise ValueError("Could not locate 'TES Sector' header in emissions workbook.")
    h = int(header_idx[0])

    years = {}
    for col_idx, token in enumerate(df.iloc[h].tolist()):
        year = _parse_year_from_header(token)
        if year is not None:
            years[col_idx] = year
    if not years:
        raise ValueError("Could not parse year columns in emissions workbook.")

    # Use unadjusted block rows for sector values.
    row_map = {
        "Domestic": "Residential buildings",
        "Services": "Commercial buildings",
        "Industrial": "Industry",
    }

    out_rows = []
    first_col = df.iloc[:, 0].astype(str).str.strip()
    for sector, source_row_label in row_map.items():
        candidates = df.index[first_col.eq(source_row_label)]
        if len(candidates) == 0:
            continue
        r = int(candidates[0])
        for c, year in years.items():
            val = pd.to_numeric(df.iat[r, c], errors="coerce")
            if pd.notna(val):
                out_rows.append(
                    {
                        "Year": int(year),
                        "Sector": sector,
                        "Emissions (ktCO2e)": float(val) * 1000.0,  # Mt -> kt
                    }
                )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No emissions rows extracted from Table1a_GHG.")
    return out


def _extract_energy_from_table_c1(file_path):
    """
    Extract total final energy (ktoe) by sector from DESNZ Table C1.
    Returns: Year, Sector, Energy Consumption (ktoe)
    """
    df = _read_excel_raw(file_path, "Table C1")

    # Layout-specific total columns in Table C1:
    # Industry total=11, Domestic total=35, Services total=44
    # Data starts below row 5 (header row).
    sector_cols = {"Industrial": 11, "Domestic": 35, "Services": 44}
    year_col = 0
    start_row = 6

    out_rows = []
    for r in range(start_row, len(df)):
        year = pd.to_numeric(df.iat[r, year_col], errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)

        for sector, c in sector_cols.items():
            val = pd.to_numeric(df.iat[r, c], errors="coerce")
            if pd.notna(val):
                out_rows.append(
                    {
                        "Year": year,
                        "Sector": sector,
                        "Energy Consumption (ktoe)": float(val),
                    }
                )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No energy rows extracted from Table C1.")
    return out


def _extract_hdd_from_data_sheet(file_path):
    """
    Extract annual HDD from DESNZ Data Heating Degree Days sheet.
    Uses the 'January-December' average row.
    Returns: Year, Annual_HDD
    """
    df = _read_excel_raw(file_path, "Data Heating Degree Days")

    header_row_idx = df.index[df.iloc[:, 0].astype(str).str.strip().eq("Calendar period")]
    if len(header_row_idx) == 0:
        raise ValueError("Could not locate 'Calendar period' header in HDD workbook.")
    h = int(header_row_idx[0])  # first block: averages

    jan_dec_idx = df.index[df.iloc[:, 0].astype(str).str.strip().eq("January-December")]
    jan_dec_idx = [i for i in jan_dec_idx if i > h]
    if not jan_dec_idx:
        raise ValueError("Could not locate 'January-December' row in HDD workbook.")
    r = int(jan_dec_idx[0])

    out_rows = []
    for c, token in enumerate(df.iloc[h].tolist()):
        year = _parse_year_from_header(token)
        if year is None:
            continue
        val = pd.to_numeric(df.iat[r, c], errors="coerce")
        if pd.notna(val):
            out_rows.append({"Year": int(year), "Annual_HDD": float(val)})

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No HDD rows extracted from data sheet.")
    return out


def _derive_end_use_shares(template_csv_path):
    """
    Build sector->end-use share weights from an existing modeled dataset.
    Shares are based on Energy Consumption and sum to 1 per sector.
    """
    path = Path(template_csv_path)
    if not path.exists():
        return None

    tdf = pd.read_csv(path)
    required = {"Sector", "End Use", "Energy Consumption (ktoe)"}
    if not required.issubset(set(tdf.columns)):
        return None

    w = (
        tdf.dropna(subset=["Sector", "End Use"])
        .groupby(["Sector", "End Use"], as_index=False)["Energy Consumption (ktoe)"]
        .sum()
    )
    w["sector_total"] = w.groupby("Sector")["Energy Consumption (ktoe)"].transform("sum")
    w = w[w["sector_total"] > 0].copy()
    w["share"] = w["Energy Consumption (ktoe)"] / w["sector_total"]
    return w[["Sector", "End Use", "share"]]


def _expand_by_end_use(sector_df, end_use_shares):
    """Disaggregate sector totals into end-use rows using sector-specific shares."""
    if end_use_shares is None or end_use_shares.empty:
        out = sector_df.copy()
        out["End Use"] = "Aggregate"
        return out

    expanded = pd.merge(sector_df, end_use_shares, on="Sector", how="left")
    expanded["share"] = expanded["share"].fillna(1.0)
    expanded["End Use"] = expanded["End Use"].fillna("Aggregate")
    expanded["Energy Consumption (ktoe)"] = expanded["Energy Consumption (ktoe)"] * expanded["share"]
    expanded["Emissions (ktCO2e)"] = expanded["Emissions (ktCO2e)"] * expanded["share"]
    return expanded.drop(columns=["share"])


def process_energy_data(energy_file, emissions_file, hdd_file):
    df_energy = _extract_energy_from_table_c1(energy_file)
    df_emissions = _extract_emissions_from_table1a(emissions_file)
    df_hdd = _extract_hdd_from_data_sheet(hdd_file)

    merged_df = pd.merge(df_energy, df_emissions, on=["Year", "Sector"], how="inner")

    # Use existing project dataset as the taxonomy source for end-use categories.
    project_root = Path(__file__).resolve().parents[2]
    template_csv = project_root / "data" / "Merged_Dataset__Energy___Emissions___HDD.csv"
    shares = _derive_end_use_shares(template_csv)
    merged_df = _expand_by_end_use(merged_df, shares)
    merged_df = pd.merge(merged_df, df_hdd, on="Year", how="left")

    merged_df["Energy_per_HDD"] = np.where(
        merged_df["Annual_HDD"] > 0,
        merged_df["Energy Consumption (ktoe)"] / merged_df["Annual_HDD"],
        np.nan,
    )

    # Keep only informative rows after end-use expansion.
    merged_df = merged_df[
        (merged_df["Energy Consumption (ktoe)"].fillna(0) != 0)
        | (merged_df["Emissions (ktCO2e)"].fillna(0) != 0)
    ].copy()

    merged_df = merged_df.sort_values(["Year", "Sector"]).reset_index(drop=True)
    merged_df = merged_df[
        [
            "Year",
            "Sector",
            "End Use",
            "Energy Consumption (ktoe)",
            "Emissions (ktCO2e)",
            "Annual_HDD",
            "Energy_per_HDD",
        ]
    ]

    out_path = Path("Processed_Energy_Modeling_Data.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"Dataset successfully merged and exported: {out_path.resolve()}")
    print(f"Rows: {len(merged_df)}, Years: {merged_df['Year'].min()}-{merged_df['Year'].max()}")
    return merged_df


if __name__ == "__main__":
    # Energy file should be the ECUK workbook, emissions file should be provisional emissions workbook.
    df = process_energy_data(
        "ECUK_2024_Consumption_tables.xlsx",
        "2023-provisional-emissions-data-tables.xlsx",
        "ET_7.1_FEB_26.xlsx",
    )
