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

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
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

        # Intersect roads with the grid
        print("  - Intersecting roads with grid...")
        intersected_roads = gpd.overlay(master_grid, roads_gdf, how="intersection")

        # Calculate length of road segments within each hexagon
        intersected_roads['road_length_m'] = intersected_roads.geometry.length

        # Aggregate total length per hexagon
        road_length_per_hex = intersected_roads.groupby('index')['road_length_m'].sum()

        # Calculate hexagon area (assuming regular hexagons, area is constant)
        hex_area = master_grid.geometry.iloc[0].area

        # Calculate road density and merge back to grid
        road_density = (road_length_per_hex / hex_area).rename("road_density_m_per_m2")
        master_grid = master_grid.merge(road_density, on="index", how="left").fillna({"road_density_m_per_m2": 0})
        print("  - Merged 'road_density_m_per_m2' into master grid.")

    except Exception as e:
        print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # --- Process Distance to Waterways ---
    layer_name = "waterwaysL"
    print(f"\n--- Processing Distance to Waterways for layer: {layer_name} ---")
    try:
        waterways_gdf = gpd.read_file(source_gpkg, layer=layer_name)
        waterways_gdf = waterways_gdf.to_crs(grid_crs)

        # Unify all waterway lines for efficient distance calculation
        unified_waterways = waterways_gdf.unary_union

        # Calculate distance from each hexagon centroid to the nearest waterway
        print("  - Calculating distance from centroids to nearest waterway...")
        master_grid['dist_to_waterway_m'] = master_grid.geometry.centroid.distance(unified_waterways)
        print("  - Merged 'dist_to_waterway_m' into master grid.")

    except Exception as e:
        print(f"Warning: Could not process layer '{layer_name}'. Skipping. Error: {e}")

    # 4. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_lines", driver="GPKG")

    print("\n--- Line layer integration complete. ---")


if __name__ == "__main__":
    main()