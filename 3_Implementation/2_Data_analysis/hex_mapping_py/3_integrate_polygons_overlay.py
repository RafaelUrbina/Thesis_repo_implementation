"""
Step 4: Integrates polygon layers using overlay operations.

This script transfers attributes from complex polygon layers to the master
grid using area-based or priority-based rules. It uses
`gpd.overlay(how='intersection')` to fragment the polygon layers against the
grid.

For each hexagon, it then applies a rule:
- Majority Area Rule: Calculates the area of all intersecting fragments and
  assigns the attribute of the fragment with the largest area.
- Ordinal Max Priority Rule: Assigns the highest (max) categorical value
  among all intersecting fragments.

The output is an enriched grid with new columns for these polygon-derived
features.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH, GRID_PATH


def apply_majority_rule(grid_gdf: gpd.GeoDataFrame, data_gdf: gpd.GeoDataFrame, attribute_col: str) -> pd.Series:
    """Applies the majority area rule for a given attribute."""
    print(f"  - Applying Majority Area Rule for '{attribute_col}'...")
    
    # Intersect the data layer with the grid
    intersected = gpd.overlay(grid_gdf[['index', 'geometry']], data_gdf, how='intersection')
    
    # Calculate the area of each fragment
    intersected['area'] = intersected.geometry.area
    
    # Find the index of the fragment with the largest area within each hexagon
    idx = intersected.groupby('index')['area'].idxmax()
    
    # Get the corresponding attribute value
    majority_values = intersected.loc[idx, ['index', attribute_col]].set_index('index')
    
    return majority_values[attribute_col]

def apply_max_priority_rule(grid_gdf: gpd.GeoDataFrame, data_gdf: gpd.GeoDataFrame, attribute_col: str) -> pd.Series:
    """Applies the max priority rule for a given attribute."""
    print(f"  - Applying Ordinal Max Priority Rule for '{attribute_col}'...")
    
    # Intersect and find the max value
    intersected = gpd.overlay(grid_gdf[['index', 'geometry']], data_gdf, how='intersection')
    max_values = intersected.groupby('index')[attribute_col].max()
    
    return max_values


def main() -> None:
    """
    Main function to execute the polygon overlay integration workflow.
    """
    # 1. Define file paths
    source_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
    grid_input_gpkg = GRID_PATH / "master_grid_with_lines.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_polygons.gpkg"

    print("--- Starting Step 3: Polygon Overlay Integration ---")

    # 2. Load master grid from previous step
    print(f"Loading enriched grid from: {grid_input_gpkg}")
    master_grid = gpd.read_file(grid_input_gpkg, layer="master_grid_with_lines")

    # 3. Define processing configuration for polygon layers
    # This list can be greatly expanded based on the full matrix.
    # Format: (layer_name, attribute_column, rule_function, new_column_name)
    polygon_layers_config = [
        ("profondita_utile_per_le_radici_cm", "profond", apply_majority_rule, "rooting_depth_class"),
        ("pietrosita_superficiale_", "ciottoli", apply_majority_rule, "surface_stoniness_class"),
        ("aggr_mosaicatura_ispra_2020_pericolosita_idraulica_firenze", "pericolo", apply_max_priority_rule, "hydraulic_hazard_level"),
        ("gruppo_idrologico_usda", "gi", apply_majority_rule, "usda_hydrologic_group"),
        ("franosita__di_superficie_interessata_da_frane", "franosita", apply_majority_rule, "landslide_surface_class"),
        ("mosaicatura_ispra_2024_pericolosita_frana_pai", "per_fr_ita", apply_max_priority_rule, "pai_landslide_hazard_level"),
        ("building_", "building", apply_majority_rule, "dominant_building_type"),
    ]

    # 4. Process each polygon layer
    for layer_name, attr_col, rule_func, new_col in polygon_layers_config:
        print(f"\n--- Processing layer: {layer_name} ---")
        try:
            data_gdf = gpd.read_file(source_gpkg, layer=layer_name)
            data_gdf = data_gdf.to_crs(master_grid.crs)
        except Exception as e:
            print(f"Warning: Could not read layer '{layer_name}'. Skipping. Error: {e}")
            continue

        # Apply the specified rule function
        new_values = rule_func(master_grid, data_gdf, attr_col)
        new_values.name = new_col

        # Merge the new feature back into the master grid
        master_grid = master_grid.merge(new_values, on="index", how="left")
        print(f"  - Merged '{new_col}' into master grid.")

    # 5. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_polygons", driver="GPKG")

    print("\n--- Polygon overlay integration complete. ---")


if __name__ == "__main__":
    main()