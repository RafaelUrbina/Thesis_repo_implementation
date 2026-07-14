"""
Step 6: Integrates environmental monitoring station data.

This script transfers data from sparse monitoring stations to the continuous
master grid. Because stations are often far apart, a simple join would leave
most hexagons empty.

This script implements two key approaches:
- Nearest Neighbor: For proximity metrics (e.g., risk from a river), it
  calculates the distance from each hexagon to the nearest station of a
  given type using `gpd.sjoin_nearest`.
- Spatial Interpolation: For continuous environmental variables (e.g.,
  temperature, rainfall), this script provides a placeholder framework.
  Implementing methods like IDW or Kriging is a complex task that typically
  requires libraries like `scipy` and careful parameter tuning.

The output is an enriched grid with new columns for station-derived features.
"""

import sys
from pathlib import Path

import geopandas as gpd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import MIDPOINTS_PATH, GRID_PATH


def main() -> None:
    """
    Main function to execute the station data integration workflow.
    """
    # 1. Define file paths
    stations_gpkg = MIDPOINTS_PATH / "aggregated_stations" / "aggregated_station_metrics.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_communal.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_stations.gpkg"

    print("--- Starting Step 5: Station Data Integration ---")

    # 2. Load master grid from previous step
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    master_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_communal")

    # --- Process Distance to Nearest Hydrometer ---
    layer_name = "idrometri_stations_firenze_aggregated"
    print(f"\n--- Processing Distance for layer: {layer_name} ---")
    try:
        hydrometers_gdf = gpd.read_file(stations_gpkg, layer=layer_name)
        hydrometers_gdf = hydrometers_gdf.to_crs(master_grid.crs)

        # sjoin_nearest calculates the distance to the nearest feature
        # The 'distance_col' parameter automatically adds a distance column.
        master_grid = gpd.sjoin_nearest(
            master_grid,
            hydrometers_gdf,
            how="left",
            distance_col="dist_to_hydrometer_m"
        )
        # Clean up extra columns from the join
        master_grid.drop(columns=['index_right'], inplace=True)
        print("  - Merged 'dist_to_hydrometer_m' into master grid.")

    except Exception as e:
        print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # --- Placeholder for Spatial Interpolation ---
    print("\n--- Placeholder for Spatial Interpolation (Temperature, Rain, Wind) ---")
    print("  - NOTE: Implementing IDW or Kriging is an advanced task.")
    print("  - This would involve creating a raster surface from station points and sampling it at hexagon centroids.")
    # Example: master_grid['interpolated_temp'] = sample_raster(temp_raster, master_grid.geometry.centroid)

    # 4. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_stations", driver="GPKG")

    print("\n--- Station data integration complete. ---")


if __name__ == "__main__":
    main()