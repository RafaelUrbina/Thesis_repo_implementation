"""
Step 2: Integrates point layers using spatial joins.

This script aggregates point data to calculate density, presence, or summary
statistics within each hexagon of the master grid. It loads the master grid
and performs a spatial join (`gpd.sjoin`) for each point layer defined in
the configuration.

The script then groups the joined data by the hexagon index (`h3_index`) and
applies the specified aggregation rules (e.g., count, sum, mean, max, mode)
to generate new features for the grid.

The output is an enriched grid saved to a new GeoPackage.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import h3pandas  # Required for k-ring neighbor analysis

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, GRID_PATH


def main() -> None:
    """
    Main function to execute the point layer integration workflow.
    """
    # 1. Define file paths
    source_gpkg = GPKG_PATH / "merged_pedologgia_static_cleaned.gpkg"

    GRID_PATH.mkdir(parents=True, exist_ok=True)
    output_gpkg = GRID_PATH / "master_grid_with_points.gpkg"

    print("--- Starting Step 1: Point Layer Integration (sjoin) ---")

    # 2. Load the master h310 grid directly from the cleaned source file
    print(f"Loading master grid from: {source_gpkg}")
    try:
        master_grid = gpd.read_file(source_gpkg, layer="h310grid")
    except Exception as e:
        print(f"Error: Could not load 'h310grid' from source file. Aborting. Details: {e}")
        return

    # 3. Define processing configuration for point layers
    # Note: 1-ring neighbor analysis is a more complex operation and is not
    # implemented here. This script focuses on direct `sjoin` aggregations.
    point_layers_config = {
        "info_lotti_multipmpoint": {
            "use_1ring": True,
            "agg": {"importo_lo": "sum", "tipo_disse": "first"},
            "rename": {"importo_lo": "total_intervention_cost", "tipo_disse": "class_intervention"}
        },
        "pubacq_acq_fontanello_aq_attivipoint": {
            "use_1ring": True,
            "agg_method": "size",
            "rename": "active_fountains_count"
        },
        "fontanellipoint": {
            "use_1ring": True,
            "agg_method": "size",
            "rename": "total_fountains_count"
        },
        "frane_piff_toscana_opendata": {
            "use_1ring": True,
            "agg": {"tipo_movimento": ["count", lambda x: x.mode()[0] if not x.empty else None]},
            "multi_index_rename": {"tipo_movimento_count": "landslide_point_count", "tipo_movimento_<lambda>": "dominant_landslide_type"}
        },
        "_peaks": {
            "use_1ring": True,
            "preprocess": lambda gdf: gdf.assign(ele=pd.to_numeric(gdf["ele"], errors="coerce")),
            "agg": {"ele": "max"},
            "rename": {"ele": "max_peak_elevation"}
        },
        "_places": {
            "use_1ring": False,  # This layer should only have direct aggregation
            "preprocess": lambda gdf: gdf.assign(is_locality=1),
            "agg": {"is_locality": "first", "name": "first"},
            "rename": {"is_locality": "is_locality", "name": "locality_name"}
        },
        "seismic_points_utm32n": {
            "use_1ring": True,
            "agg": {"MwDef": ["count", "max", "mean"]},
            "multi_index_rename": {"MwDef_count": "seismic_event_count", "MwDef_max": "max_seismic_magnitude", "MwDef_mean": "avg_seismic_magnitude"}
        }
        # NOTE: 'celle_soli_PS_discendenti' and 'celle_soli_PS_ascendenti' have been removed
        # as they are polygon layers and should be processed in a polygon integration script
        # to ensure correct area-based analysis, not point-based joins.
    }

    # 4. Process each point layer
    for layer_name, config in point_layers_config.items():
        print(f"\n--- Processing layer: {layer_name} ---")
        try:
            points_gdf = gpd.read_file(source_gpkg, layer=layer_name)
        except Exception as e:
            print(f"Warning: Could not read layer '{layer_name}'. Skipping. Error: {e}")
            continue

        # Ensure CRS matches master grid
        points_gdf = points_gdf.to_crs(master_grid.crs)

        # Preprocessing step if defined
        if "preprocess" in config:
            points_gdf = config["preprocess"](points_gdf)

        # --- Direct Aggregation (Inside each hexagon) ---
        points_in_hex = gpd.sjoin(points_gdf, master_grid[['index', 'geometry']], how="inner", predicate="within")

        if config.get("agg_method") == "size":
            direct_agg = points_in_hex.groupby("index").size()
            direct_agg.name = config["rename"]
            direct_agg = direct_agg.to_frame()
        else:
            direct_agg = points_in_hex.groupby("index").agg(config["agg"])
            if "multi_index_rename" in config:
                direct_agg.columns = ["_".join(col).strip() for col in direct_agg.columns.values]
                direct_agg = direct_agg.rename(columns=config["multi_index_rename"])
            else:
                direct_agg = direct_agg.rename(columns=config["rename"])

        # --- Conditional 1-Ring Neighbor Aggregation ---
        if config.get("use_1ring"):
            print("  - Performing 1-ring neighbor aggregation...")
            
            # Start with the non-zero direct aggregations
            initial_agg = direct_agg.copy().dropna(how='all')
            if initial_agg.empty:
                print("  - No data for neighbor aggregation. Skipping.")
                continue

            # Find 1-ring neighbors for each hexagon that has a value
            neighbor_sets = initial_agg.h3.k_ring(1)

            # Create a mapping from each active hexagon to all its neighbors
            exploded_neighbors = neighbor_sets.rename(columns={"h3_k_ring": "neighbors"}).explode("neighbors").reset_index()

            # Perform the final aggregation: for each hexagon, sum up the contributions from all its neighbors
            # The exploded_neighbors DataFrame already contains all necessary data.
            
            # --- Intelligent Neighbor Aggregation ---
            # Define which columns should be summed vs. which should take the max value.
            # Columns with 'count' or 'cost' in their name are summed. All others take the max.
            # Categorical columns (dtype='object') will use the mode.
            agg_rules = {}
            for col in initial_agg.columns:
                if 'count' in col or 'cost' in col:
                    agg_rules[col] = 'sum'
                elif initial_agg[col].dtype == 'object':
                    # For categorical data, find the most frequent value (mode)
                    agg_rules[col] = lambda x: x.mode()[0] if not x.empty else None
                else:
                    agg_rules[col] = 'max'
            
            neighbor_agg = exploded_neighbors.groupby("neighbors").agg(agg_rules)

            # Rename neighbor aggregation columns and merge them
            final_agg = neighbor_agg.add_suffix('_1ring')
            master_grid = master_grid.merge(final_agg, left_on="index", right_index=True, how="left")
            print(f"  - Merged 1-ring features: {list(final_agg.columns)}")

        else: # If use_1ring is False or not specified
            print("  - Performing direct aggregation only...")
            master_grid = master_grid.merge(direct_agg, left_on="index", right_index=True, how="left")
            print(f"  - Merged direct features: {list(direct_agg.columns)}")

    # Final cleanup: Fill NaN values based on data type
    print("\nPerforming final cleanup of NaN values...")
    for col in master_grid.columns:
        # Check if the column is numeric (integer, float)
        if pd.api.types.is_numeric_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].fillna(0)
        # Check if the column is categorical (object/string)
        elif pd.api.types.is_object_dtype(master_grid[col]):
            master_grid[col] = master_grid[col].fillna(None)

    # 5. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_points", driver="GPKG")

    print("\n--- Point layer integration complete. ---")


if __name__ == "__main__":
    main()