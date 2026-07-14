"""
Calculates commuter inflow density for each municipality and joins it with
communal geometries.

This script implements "Approach A: Destination Magnetism." It reads the
national commuter matrix, aggregates the number of commuters ('Pendolari')
by their destination municipality ('Procom_lav'), and joins this total inflow
value to the municipal boundary layer ('com01012026_wgs84').

The output is a shapefile where each municipal polygon has an attribute
representing the total number of people commuting into it for work.
"""

import csv
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH, PENDOLARISMO_PATH


def main() -> None:
    """
    Main function to perform the data loading, aggregation, and export.
    """
    # 1. Define file paths
    pendolarismo_file = PENDOLARISMO_PATH / "matrix_pendoLAVORO_2021.txt"
    gpkg_file = GPKG_PATH / "merged_pedologgia_static.gpkg"
    layer_name = "com01012026_wgs84"

    output_dir = MIDPOINTS_PATH / "pendolarismo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pendolarismo_inflow_comunal.gpkg"
    output_layer_name = "pendolarismo_inflow_comunal"
    csv_temp_file = output_dir / "matrix_pendoLAVORO_2021.csv"

    # 2. Load and process commuter data
    # First, convert the whitespace-delimited .txt to a standard .csv
    # This avoids parsing ambiguities with pandas' read_csv on complex separators.
    print(f"Converting {pendolarismo_file.name} to CSV format...")
    with open(pendolarismo_file, "r", encoding="utf-8") as infile, open(
        csv_temp_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        writer = csv.writer(outfile)
        # Read header and write to CSV
        header = infile.readline().strip().split()
        writer.writerow(header)
        # Process and write data rows
        for line in infile:
            writer.writerow(line.strip().split())

    print(f"Loading standardized commuter data from: {csv_temp_file.name}")
    pendo_df = pd.read_csv(
        csv_temp_file,
        dtype={"procom_res": str, "Procom_lav": str},
    )

    # Explicitly convert 'Pendolari' to a numeric type after loading.
    # `errors='coerce'` will turn any non-numeric values into NaN (Not a Number).
    pendo_df["Pendolari"] = pd.to_numeric(pendo_df["Pendolari"], errors="coerce")

    # Calculate inflow: sum of commuters grouped by destination
    print("Calculating total commuter inflow for each destination municipality...")
    inflow = pendo_df.groupby("Procom_lav")["Pendolari"].sum().reset_index()
    inflow.rename(columns={"Pendolari": "inflow_total"}, inplace=True)

    # 3. Load municipal boundaries
    print(f"Loading municipal boundaries from layer '{layer_name}' in {gpkg_file}")
    comuni_gdf = gpd.read_file(gpkg_file, layer=layer_name)

    # 4. Join commuter data to geometries
    # Ensure join keys are the same string format
    print("Joining commuter inflow data to municipal geometries...")
    comuni_gdf["PRO_COM_str"] = comuni_gdf["PRO_COM_T"].astype(str).str.strip()
    inflow["Procom_lav_str"] = inflow["Procom_lav"].astype(str).str.strip()

    merged_gdf = comuni_gdf.merge(
        inflow,
        left_on="PRO_COM_str",
        right_on="Procom_lav_str",
        how="left",
    )

    # Fill missing inflow values with 0 and clean up columns
    merged_gdf["inflow_total"] = merged_gdf["inflow_total"].fillna(0)
    merged_gdf = merged_gdf.drop(columns=["PRO_COM_str", "Procom_lav_str", "Procom_lav"])

    # 5. Save the output to a GeoPackage
    print(f"Saving output GeoPackage to: {output_file}")
    merged_gdf.to_file(output_file, layer=output_layer_name, driver="GPKG")

    print("\nDone.")
    print(f"{len(merged_gdf)} municipalities with inflow data saved.")


if __name__ == "__main__":
    main()