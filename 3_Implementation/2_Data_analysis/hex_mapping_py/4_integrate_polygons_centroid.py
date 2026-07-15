"""
Step 5: Integrates communal data using centroid-based assignment.

This script assigns attributes from large administrative polygons (e.g.,
municipalities with census or economic data) to the hexagons they contain.

It calculates the centroid of each hexagon and performs a spatial join
(`gpd.sjoin` with `predicate='within'`) to determine which administrative
polygon each centroid falls into. This efficiently transfers the attributes
of the encompassing polygon to the hexagon.

The output is an enriched grid with new columns for these administrative
features.
"""

import sys
from pathlib import Path

import geopandas as gpd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH, GRID_PATH


def main() -> None:
    """
    Main function to execute the centroid-based integration workflow.
    """
    # 1. Define file paths
    source_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
    pendolarismo_gpkg = MIDPOINTS_PATH / "pendolarismo" / "pendolarismo_inflow_comunal.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_polygons.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_communal.gpkg"

    print("--- Starting Step 4: Communal Data Integration (Centroid) ---")

    # 2. Load master grid from previous step
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    master_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_polygons")

    # Create a new GeoDataFrame with hexagon centroids
    grid_centroids = master_grid.copy()
    grid_centroids['geometry'] = grid_centroids.geometry.centroid

    # 3. Define layers to process
    # Using pre-processed commuter data as an example
    communal_layers_config = [
        (pendolarismo_gpkg, "pendolarismo_inflow_comunal", ["inflow_total"]),
        # Add other tabular_comunal layers here in the same format
        # (gpkg_path, layer_name, [list_of_columns_to_keep])
    ]

    # 4. Process each communal layer
    for path, layer_name, columns_to_keep in communal_layers_config:
        print(f"\n--- Processing layer: {layer_name} ---")
        try:
            data_gdf = gpd.read_file(path, layer=layer_name)
            data_gdf = data_gdf.to_crs(master_grid.crs)

            # Perform the spatial join
            joined_gdf = gpd.sjoin(grid_centroids, data_gdf[columns_to_keep + ['geometry']], how="left", predicate="within")

            # Merge the new columns back to the main grid using the index
            master_grid = master_grid.join(joined_gdf[columns_to_keep])
            print(f"  - Merged {columns_to_keep} into master grid.")

        except Exception as e:
            print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # Final cleanup: Fill NaN values based on data type
    print("\nPerforming final cleanup of NaN values...")
    for col in master_grid.columns:
        # Check if the column is numeric (integer, float)
        if pd.api.types.is_numeric_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].fillna(0)
        # Check if the column is categorical (object/string)
        elif pd.api.types.is_object_dtype(master_grid[col]):
            # Using fillna with None on object columns is effective
            master_grid[col] = master_grid[col].fillna(None)

    # 5. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_communal", driver="GPKG")

    print("\n--- Communal data integration complete. ---")


if __name__ == "__main__":
    main()