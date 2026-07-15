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
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, GRID_PATH


def main() -> None:
    """
    Main function to execute the centroid-based integration workflow.
    """
    # 1. Define file paths
    source_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"
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
    # This configuration defines all communal-level data to be joined via centroid assignment.
    # Format: (gpkg_path, layer_name, [list_of_columns_to_keep])
    communal_layers_config = [
        (source_gpkg, "tabular_comunal_census", ['census_pop']),
        (source_gpkg, "tabular_comunal_economical_princ", [
            'epr_NTAXP', 'epr_TAXABINC', 'epr_CADINCR', 'epr_CADINCF', 'epr_SUBEMPTR',
            'epr_PENSINCR', 'epr_PENSINCF', 'epr_ENTROAIN', 'epr_ENTROAIN01'
        ]),
        (source_gpkg, "tabular_comunal_economical_reddito", [
            'erd_E0_10000', 'erd_E10000_1', 'erd_E15000_2', 'erd_E26000_5', 'erd_E55000_7', 
            'erd_E75000_1', 'erd_E_GE1200'
        ]),
        (source_gpkg, "tabular_comunal_economical_distrib", ['edst_acq_imm', 'edst_acq_erog']),
        (source_gpkg, "tabular_comunal_economical_indice_comp", [
            'eidx_COMP_FRA', 'eidx_LAND_CON', 'eidx_EMPL_RAT', 
            'eidx_POP_25_6', 'eidx_POP_DEPE', 
            'eidx_INDEX_AC', 'eidx_PERSEMP'
        ]),
        (source_gpkg, "pendolarismo_inflow_comunal", ['inflow_total']),
    ]

    # 4. Process each communal layer
    for path, layer_name, columns_to_keep in communal_layers_config:
        print(f"\n--- Processing layer: {layer_name} ---")
        try:
            # All layers are read from the single cleaned source GPKG
            data_gdf = gpd.read_file(source_gpkg, layer=layer_name)
            data_gdf = data_gdf.to_crs(master_grid.crs)

            # Perform the spatial join
            joined_gdf = gpd.sjoin(grid_centroids[['index', 'geometry']], data_gdf[columns_to_keep + ['geometry']], how="left", predicate="within")

            # The sjoin can create duplicate rows if a hexagon is on a boundary.
            # We drop duplicates, keeping only the first match for each hexagon.
            # We also handle the 'index_right' column created by the join.
            joined_gdf.drop(columns=['index_right'], inplace=True, errors='ignore')
            joined_gdf.drop_duplicates(subset='index', keep='first', inplace=True)

            # Merge the new columns back to the main grid using a robust merge on the index
            master_grid = master_grid.merge(
                joined_gdf[['index'] + columns_to_keep], on='index', how='left'
            )
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