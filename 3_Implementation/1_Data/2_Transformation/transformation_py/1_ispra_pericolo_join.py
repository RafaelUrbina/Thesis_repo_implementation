"""
Aggregates hydraulic hazard layers into a single layer with a unified hazard
level.

This script reads three hydraulic hazard layers (high, medium, and low) from
the main project GeoPackage. It assigns a numeric hazard level to each layer
based on a defined hierarchy (High=3, Medium=2, Low=1).

The script then performs a cascading union overlay to combine the layers.
This ensures that areas with multiple hazard levels are assigned the highest
level from the hierarchy (e.g., an area in both high and medium hazard zones
will be classified as high hazard).

The final aggregated layer, containing all geometries and a single 'pericolo'
column representing the hazard level, is saved back to the same GeoPackage.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH


def main() -> None:
    """
    Main function to execute the hydraulic hazard aggregation workflow.
    """
    # 1. Define file paths and layer names
    gpkg_path = GPKG_PATH / "merged_pedologgia_static.gpkg"
    output_layer_name = "aggr_mosaicatura_ispra_2020_pericolosita_idraulica_firenze"

    # Layers are defined in order of importance (highest to lowest)
    hazard_layers = {
        "hph_mosaicatura_ispra_2020_pericolosita_idraulica_firenze": 3,  # High
        "mph_mosaicatura_ispra_2020_pericolosita_idraulica_firenze": 2,  # Medium
        "lph_mosaicatura_ispra_2020_pericolosita_idraulica_firenze": 1,  # Low
    }

    print(f"Loading layers from GeoPackage: {gpkg_path.name}")
    
    # 2. Load layers into a dictionary
    gdfs = {}
    try:
        for layer_name in hazard_layers:
            gdfs[layer_name] = gpd.read_file(gpkg_path, layer=layer_name)
            print(f"  - Loaded layer '{layer_name}' with {len(gdfs[layer_name])} features.")
    except Exception as e:
        print(f"Error: Could not read a required layer. Aborting. Details: {e}")
        return

    # 3. Apply hierarchical overlay logic
    print("\nApplying hierarchical overlay to combine layers...")

    hph_name, mph_name, lph_name = hazard_layers.keys()

    # Start with the highest priority layer
    hph_gdf = gdfs[hph_name]
    hph_gdf['pericolo'] = hazard_layers[hph_name]

    # Create a unified geometry of the highest priority layer to clip against
    hph_union = unary_union(hph_gdf.geometry)

    # Process medium priority: remove areas that overlap with high priority
    mph_gdf = gdfs[mph_name]
    mph_clipped = gpd.overlay(mph_gdf, gpd.GeoDataFrame(geometry=[hph_union], crs=hph_gdf.crs), how='difference')
    mph_clipped['pericolo'] = hazard_layers[mph_name]

    # Create a unified geometry of high and medium priority layers
    hph_mph_union = unary_union(list(hph_gdf.geometry) + list(mph_clipped.geometry))

    # Process low priority: remove areas that overlap with high and medium
    lph_gdf = gdfs[lph_name]
    lph_clipped = gpd.overlay(lph_gdf, gpd.GeoDataFrame(geometry=[hph_mph_union], crs=hph_gdf.crs), how='difference')
    lph_clipped['pericolo'] = hazard_layers[lph_name]

    # Combine the processed layers
    aggregated_gdf = pd.concat([hph_gdf, mph_clipped, lph_clipped], ignore_index=True)

    # 4. Save the aggregated layer back to the GeoPackage
    print(f"\nSaving aggregated layer '{output_layer_name}' to {gpkg_path.name}...")
    aggregated_gdf.to_file(gpkg_path, layer=output_layer_name, driver="GPKG")

    print(f"\nDone. Wrote {len(aggregated_gdf)} features to the new layer.")


if __name__ == "__main__":
    main()