"""
Step 3: Integrates line layers to calculate density and proximity.

This script processes linear features (_road, waterwaysL) and calculates
meaningful metrics for each hexagon in the master grid.

For road networks, it calculates road density (total length of roads per
hexagon area). For features like waterways, it calculates the distance from
each hexagon's centroid to the nearest line segment, providing a proximity
metric.

The output is an enriched grid with new columns for these linear metrics.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH, GRID_PATH


def main() -> None:
    """
    Main function to execute the line layer integration workflow.
    """
    # 1. Define file paths
    source_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_points.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_lines.gpkg"

    print("--- Starting Step 2: Line Layer Integration ---")

    # 2. Load master grid from previous step
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    master_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_points")
    grid_crs = master_grid.crs

    # --- Process Road Density ---
    layer_name = "_road"
    print(f"\n--- Processing Road Density for layer: {layer_name} ---")
    try:
        roads_gdf = gpd.read_file(source_gpkg, layer=layer_name)
        roads_gdf = roads_gdf.to_crs(grid_crs)

        # Filter for valid line geometries to prevent errors/warnings in overlay
        valid_geom_types = ["LineString", "MultiLineString"]
        roads_gdf = roads_gdf[roads_gdf.geometry.geom_type.isin(valid_geom_types)]

        # Intersect roads with the grid
        print("  - Intersecting roads with grid...")
        # The order is crucial: roads_gdf must be the first argument to ensure
        # the output contains the clipped LINE geometries, not empty polygons.
        # We also select only the 'index' and 'geometry' from the master_grid
        # to make the overlay cleaner and faster.
        intersected_roads = gpd.overlay(roads_gdf, master_grid[['index', 'geometry']], how="intersection")

        # Calculate length of road segments within each hexagon
        intersected_roads['road_length_m'] = intersected_roads.geometry.length


        # Aggregate total length per hexagon
        road_length_per_hex = intersected_roads.groupby('index')['road_length_m'].sum()

        # --- Calculate Road Density in multiple units for flexibility ---
        # Unit 1: m/m² (raw density)
        hex_area_m2 = master_grid.geometry.iloc[0].area
        density_m_per_m2 = (road_length_per_hex / hex_area_m2).rename("road_density_m_per_m2")

        # Unit 2: m_per_hex (total meters of road per hexagon)
        m_per_hex = road_length_per_hex.rename("m_per_hex")

        # Merge both density metrics into the master grid
        master_grid = master_grid.merge(density_m_per_m2, on="index", how="left")
        master_grid = master_grid.merge(m_per_hex, on="index", how="left")
        print("  - Merged 'road_density_m_per_m2' and 'm_per_hex' into master grid.")

        # --- Calculate Dominant Road Category by Length ---
        print("  - Calculating dominant road categories by length...")
        categorical_cols = ['highway', 'surface', 'tunnel', 'bridge']
        for col in categorical_cols:
            if col not in intersected_roads.columns:
                print(f"    - Warning: Categorical column '{col}' not found in road layer. Skipping.")
                continue

            # Sum the length for each category within each hexagon
            length_by_cat = intersected_roads.groupby(['index', col])['road_length_m'].sum().reset_index()

            # Find the category with the maximum length for each hexagon
            idx = length_by_cat.loc[length_by_cat.groupby('index')['road_length_m'].idxmax()]
            dominant_category = idx.set_index('index')[col]

            # Merge the new dominant category column into the master grid
            new_col_name = f"dominant_{col}"
            master_grid = master_grid.merge(dominant_category.rename(new_col_name), on="index", how="left")
            print(f"    - Merged '{new_col_name}' into master grid.")

    except Exception as e:
        print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")
    # --- Process Distance to Waterways ---
    layer_name = "_waterwaysL"
    print(f"\n--- Processing Distance to Waterways for layer: {layer_name} ---")
    try:
        waterways_gdf = gpd.read_file(source_gpkg, layer=layer_name)
        waterways_gdf = waterways_gdf.to_crs(grid_crs)

        # Unify all waterway lines for efficient distance calculation
        unified_waterways = waterways_gdf.union_all()

        # Calculate distance from each hexagon centroid to the nearest waterway
        print("  - Calculating distance from centroids to nearest waterway...")
        master_grid['dist_to_waterway_m'] = master_grid.geometry.centroid.distance(unified_waterways)
        print("  - Merged 'dist_to_waterway_m' into master grid.")

    except Exception as e:
        print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # Final cleanup: Fill NaN values and round all float columns to 4 decimals
    print("\nPerforming final cleanup of NaN values and rounding float variables...")
    for col in master_grid.columns:
        if pd.api.types.is_numeric_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].fillna(0)
            if pd.api.types.is_float_dtype(master_grid[col]):
                master_grid[col] = master_grid[col].round(4)
        elif pd.api.types.is_object_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].fillna(None)

    # 4. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_lines", driver="GPKG")

    print("\n--- Line layer integration complete. ---")


if __name__ == "__main__":
    main()