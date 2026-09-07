"""
Step 7: Finalizes the data cube.

This is the final script in the integration pipeline. It loads the enriched
grid from the previous step and performs the last merge operations.

Specifically, it joins the pre-calculated DTM (Digital Terrain Model) zonal
statistics, which are already based on the H3 grid.

Finally, it performs any last-minute cleanup, such as handling remaining null
values or checking data types, and saves the completed data cube to a final
GeoPackage file.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import MIDPOINTS_PATH, GRID_PATH, GPKG_PATH


def main() -> None:
    """
    Main function to execute the final data cube assembly.
    """
    # 1. Define file paths
    dtm_stats_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_stations.gpkg"

    # Define the final output directory as requested
    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "h310_grid_final_datacube.gpkg"

    print("--- Starting Step 6: Finalize Data Cube ---")

    # 2. Load the main grid and the DTM stats grid
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    final_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_stations")

    print(f"Loading DTM stats from: {dtm_stats_gpkg}")
    dtm_stats_grid = gpd.read_file(dtm_stats_gpkg, layer="h3_grid_with_dtm_stats")

    # 3. Merge DTM stats into the final grid
    # We only need the statistics columns, not the geometry
    dtm_cols = [col for col in dtm_stats_grid.columns if col.endswith(('_mean', '_max'))]
    print(f"  - Merging DTM columns: {dtm_cols}")
    final_grid = final_grid.merge(dtm_stats_grid[['index'] + dtm_cols], on="index", how="left")

    # 4. Final cleanup: Fill NaN values and round all float columns to 4 decimals
    print("\nPerforming final cleanup of NaN values and rounding float variables...")
    for col in final_grid.columns:
        if pd.api.types.is_numeric_dtype(final_grid[col]):
            final_grid[col] = final_grid[col].fillna(0)
            if pd.api.types.is_float_dtype(final_grid[col]):
                final_grid[col] = final_grid[col].round(4)
        elif pd.api.types.is_object_dtype(final_grid[col]):
            final_grid[col] = final_grid[col].fillna(None)

    # 5. Save the final data cube
    print(f"\nSaving FINAL DATA CUBE to: {output_gpkg}")
    final_grid.to_file(output_gpkg, layer="final_data_cube", driver="GPKG")

    print("\n--- Data cube finalization complete. ---")


if __name__ == "__main__":
    main()