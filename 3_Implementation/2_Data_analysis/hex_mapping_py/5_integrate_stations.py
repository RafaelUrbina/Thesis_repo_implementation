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

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, GRID_PATH

def inverse_distance_weighting(grid_centroids: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, value_col: str, k: int, power: int = 2) -> pd.Series:
    """
    Performs Inverse Distance Weighting (IDW) interpolation using a vectorized
    approach with scipy.cKDTree for high performance.

    Args:
        grid_centroids: GeoDataFrame of hexagon centroids.
        stations: GeoDataFrame of station points with data.
        value_col: The name of the column in 'stations' to interpolate.
        k: The number of nearest neighbors to use.
        power: The power to raise the inverse distance to.

    Returns:
        A pandas Series with the interpolated values for each grid centroid.
    """
    # 1. Extract coordinates and values into NumPy arrays for efficiency.
    #    Using .get_coordinates().values is a robust way to get a (n, 2) array.
    station_coords = stations.geometry.get_coordinates().values
    grid_coords = grid_centroids.geometry.get_coordinates().values
    station_values_np = stations[value_col].to_numpy()

    # 2. Build a k-d tree from station coordinates for efficient nearest neighbor search.
    tree = cKDTree(station_coords)

    # 3. Query the tree to find the k nearest stations for each grid point.
    #    `distances` and `indices` will have shape (n_grid_points, k).
    distances, indices = tree.query(grid_coords, k=k)

    # 4. Handle the case where k=1, as tree.query returns 1D arrays.
    #    We reshape them to 2D to keep the logic consistent.
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # 5. Calculate IDW weights.
    #    Avoid division by zero if a grid point is exactly on a station.
    distances[distances == 0] = 1e-10
    weights = 1.0 / (distances ** power)
    normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)

    # 6. Get the values of the neighboring stations using the indices.
    #    This is a direct, reliable NumPy indexing operation.
    neighbor_values = station_values_np[indices]

    # 7. Calculate the weighted average (the interpolated value).
    #    This is an element-wise multiplication and a sum over the k-neighbor axis.
    interpolated_values = np.sum(normalized_weights * neighbor_values, axis=1)

    return pd.Series(interpolated_values, index=grid_centroids.index)


def k_nearest_neighbor_stats(grid_centroids: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, k: int = 3) -> pd.Series:
    """Calculates the average distance to the k-nearest stations."""
    station_coords = np.array([stations.geometry.x, stations.geometry.y]).T
    grid_coords = np.array([grid_centroids.geometry.x, grid_centroids.geometry.y]).T

    tree = cKDTree(station_coords)
    distances, _ = tree.query(grid_coords, k=k)

    # Calculate the mean distance to the k neighbors
    mean_distances = np.mean(distances, axis=1)
    return pd.Series(mean_distances, index=grid_centroids.index)


def nearest_neighbor_join(grid_gdf: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, cols_to_join: list) -> gpd.GeoDataFrame:
    """Performs a nearest neighbor join to get distance and attributes."""
    joined_gdf = gpd.sjoin_nearest(
        grid_gdf,
        stations,
        how="left",
        distance_col="distance_m"
    )
    # Clean up extra columns and duplicates, keeping only the first (nearest) match
    joined_gdf = joined_gdf.drop(columns=['index_right']).drop_duplicates(subset='index', keep='first')
    return joined_gdf[['distance_m'] + cols_to_join]


def main() -> None:
    """
    Main function to execute the station data integration workflow.
    """
    # 1. Define file paths
    stations_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_communal.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_stations.gpkg"

    print("--- Starting Step 5: Station Data Integration ---")

    # 2. Load master grid from previous step
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    master_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_communal")
    grid_centroids = master_grid.copy()
    grid_centroids['geometry'] = grid_centroids.geometry.centroid

    # 3. Define configuration for station data processing
    station_processing_config = {
        # For hydrometers, we find the single nearest station and get its distance and attributes.
        "idrometri_stations_firenze_aggregated": {
            "method": nearest_neighbor_join,
            "params": {
                "cols_to_join": ["river_stage_max_m", "river_stage_mean_m", "river_stage_std_m", "quota"]
            },
            "rename_prefix": "nearest_hydro_"
        },
        # For continuous weather data, we interpolate each metric using IDW.
        "termometri_stations_firenze_aggregated": {
            "method": inverse_distance_weighting,
            "interpolate_cols": ["temp_max_peak", "temp_min_nadir", "temp_thermal_range"],
            "k": 6,  # Increased k for smoother results
            "power": 1.2
        },
        "cf_pluviometri_aggregated": {
            "method": inverse_distance_weighting,
            "interpolate_cols": [
                "rain_mm_sum_annual", "rain_mm_max_monthly", "rain_mm_min_monthly",
                "rain_mm_avg_monthly", "rain_mm_std"
            ],
            "k": 6,
            "power": 1.2
        },
        "anemometri_stations_firenze_aggregated": {
            "method": inverse_distance_weighting,
            "interpolate_cols": ["wind_speed_max_max", "wind_speed_max_95p", "wind_speed_avg_mean"],
            "k": 6,
            "power": 1.2
        },
    }

    # 4. Process each station layer according to the configuration
    for layer_name, config in station_processing_config.items():
        print(f"\n--- Processing layer: {layer_name} ---")
        try:
            stations_gdf = gpd.read_file(stations_gpkg, layer=layer_name)
            stations_gdf = stations_gdf.to_crs(master_grid.crs)
            stations_gdf.dropna(subset=['geometry'], inplace=True)

            if stations_gdf.empty:
                print("  - Warning: Layer is empty after cleaning. Skipping.")
                continue

            # --- Handle Interpolation vs. Nearest Join ---
            if config["method"] == inverse_distance_weighting:
                # This is an interpolation task for multiple columns
                for value_col in config["interpolate_cols"]:
                    print(f"  - Interpolating '{value_col}'...")
                    # Ensure stations have valid data for the column being interpolated
                    valid_stations = stations_gdf.dropna(subset=[value_col]).copy()
                    if valid_stations.empty:
                        print(f"    - Warning: No valid data for '{value_col}'. Skipping.")
                        continue
                    
                    # Dynamically adjust k to be no more than the number of available stations.
                    # This prevents errors when fewer than `k` stations have valid data.
                    k_neighbors = min(config.get("k", 5), len(valid_stations))
                    power_val = config.get("power", 2)

                    if k_neighbors == 0: continue # Should not happen due to earlier check, but for safety.

                    idw_params = {"value_col": value_col, "k": k_neighbors, "power": power_val}
                    output_col_name = f"idw_{value_col}"
                    
                    # Call the interpolation function
                    interpolated_values = inverse_distance_weighting(grid_centroids, valid_stations, **idw_params)
                    master_grid[output_col_name] = interpolated_values
                    print(f"    - Merged '{output_col_name}' into master grid.")

            elif config["method"] == nearest_neighbor_join:
                # This is a nearest neighbor join task
                print("  - Performing nearest neighbor join...")
                new_cols_df = nearest_neighbor_join(master_grid[['index', 'geometry']], stations_gdf, **config["params"])
                
                # Add a prefix to the new columns to avoid name collisions
                rename_dict = {
                    col: f"{config['rename_prefix']}{col}" 
                    for col in new_cols_df.columns
                }
                new_cols_df.rename(columns=rename_dict, inplace=True)

                # Merge the new columns back into the master grid
                master_grid = master_grid.merge(new_cols_df, left_index=True, right_index=True, how="left")
                print(f"  - Merged {list(new_cols_df.columns)} into master grid.")

        except Exception as e:
            print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # Final cleanup: Fill NaN values and round all float columns to 4 decimals
    print("\nPerforming final cleanup of NaN values and rounding float variables...")
    numeric_cols = master_grid.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        master_grid[col] = master_grid[col].fillna(0)
        if pd.api.types.is_float_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].round(4)

    # 5. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_stations", driver="GPKG")

    print("\n--- Station data integration complete. ---")


if __name__ == "__main__":
    main()