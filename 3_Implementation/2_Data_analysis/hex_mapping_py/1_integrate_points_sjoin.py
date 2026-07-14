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
            "agg": {"importo_lo": "sum", "tipo_disse": "first"},
            "rename": {"importo_lo": "total_intervention_cost", "tipo_disse": "class_intervention"}
        },
        "pubacq_acq_fontanello_aq_attivipoint": {
            "agg_method": "size",
            "rename": "active_fountains_count"
        },
        "fontanellipoint": {
            "agg_method": "size",
            "rename": "total_fountains_count"
        },
        "frane_piff_toscana_opendata": {
            "agg": {"tipo_movimento": ["count", lambda x: x.mode()[0] if not x.empty else None]},
            "multi_index_rename": {"tipo_movimento_count": "landslide_point_count", "tipo_movimento_<lambda>": "dominant_landslide_type"}
        },
        "_peaks": {
            "preprocess": lambda gdf: gdf.assign(ele=pd.to_numeric(gdf["ele"], errors="coerce")),
            "agg": {"ele": "max"},
            "rename": {"ele": "max_peak_elevation"}
        },
        "_places": {
            "preprocess": lambda gdf: gdf.assign(is_locality=1),
            "agg": {"is_locality": "first", "name": "first"},
            "rename": {"is_locality": "is_locality", "name": "locality_name"}
        },
        "seismic_points_utm32n": {
            "agg": {"MwDef": ["count", "max", "mean"]},
            "multi_index_rename": {"MwDef_count": "seismic_event_count", "MwDef_max": "max_seismic_magnitude", "MwDef_mean": "avg_seismic_magnitude"}
        },
        "celle_soli_PS_discendenti": {
            "agg": {"ave_vdesc": "mean"},
            "rename": {"ave_vdesc": "avg_descending_soil_speed"}
        },
        "celle_soli_PS_ascendenti": {
            "agg": {"ave_vasc": "mean"},
            "rename": {"ave_vasc": "avg_ascending_soil_speed"}
        }
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

        # Perform spatial join
        joined_gdf = gpd.sjoin(points_gdf, master_grid, how="inner", predicate="within")

        # --- Aggregate Data ---
        # Use the robust .size() method for simple counts.
        if config.get("agg_method") == "size":
            aggregated_data = joined_gdf.groupby("index").size()
            aggregated_data.name = config["rename"]
        # Use the flexible .agg() method for complex aggregations.
        else:
            aggregated_data = joined_gdf.groupby("index").agg(config["agg"])
            # Handle multi-index columns from aggregations like ['mean', 'max']
            if "multi_index_rename" in config:
                aggregated_data.columns = ["_".join(col).strip() for col in aggregated_data.columns.values]
                aggregated_data = aggregated_data.rename(columns=config["multi_index_rename"])
            else:
                aggregated_data = aggregated_data.rename(columns=config["rename"])

        # Merge aggregated data back into the master grid
        master_grid = master_grid.merge(aggregated_data, on="index", how="left")

        if isinstance(aggregated_data, pd.DataFrame):
            print(f"  - Merged {list(aggregated_data.columns)} into master grid.")
        else:  # It's a Series
            print(f"  - Merged ['{aggregated_data.name}'] into master grid.")
    # Fill NaNs created by non-joining hexagons with appropriate values (0 for counts/sums)
    for col in master_grid.columns:
        if "count" in col or "cost" in col or col == "is_locality":
            master_grid[col] = master_grid[col].fillna(0)

    # 5. Save the enriched grid
    print(f"\nSaving enriched grid to: {output_gpkg}")
    master_grid.to_file(output_gpkg, layer="master_grid_with_points", driver="GPKG")

    print("\n--- Point layer integration complete. ---")


if __name__ == "__main__":
    main()