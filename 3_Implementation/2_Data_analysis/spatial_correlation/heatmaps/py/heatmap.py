from pathlib import Path
import geopandas as gpd
import pandas as pd
import sys

# Add the project root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GRID_PATH

def process_all_h3_pairs(
    gpkg_path: str,
    pairs_list: list[tuple[str, str]],
    output_gpkg_path: str,
    h3_col: str = "index",
):
    """Processes multiple categorical feature pairs over an H3 grid.

    Parameters:
    -----------
    gpkg_path : str
        Path to input GeoPackage containing individual feature layers.
    pairs_list : list of tuples
        List of (feature_1, feature_2) pairings to process.
    output_gpkg_path : str
        Path where output GeoPackage containing pair heatmaps will be written.
    h3_col : str
        The common H3 index column present in all layers.
    """
    gpkg_path = Path(gpkg_path)
    output_gpkg_path = Path(output_gpkg_path)

    # 1. Identify unique layers needed and cache them in memory to avoid duplicate reads
    unique_features = set()
    for f1, f2 in pairs_list:
        unique_features.add(f1)
        unique_features.add(f2)

    print(f"Pre-loading {len(unique_features)} unique layers from {gpkg_path}...")
    layers_cache = {}
    for feat in unique_features:
        gdf = gpd.read_file(gpkg_path, layer=feat)

        if h3_col not in gdf.columns:
            raise KeyError(
                f"Column '{h3_col}' not found in layer '{feat}'. Available columns: {list(gdf.columns)}"
            )

        gdf[h3_col] = gdf[h3_col].astype(str)

        # Identify category attribute column (picks layer name match or first non-geometry column)
        cat_col = (
            feat
            if feat in gdf.columns
            else [c for c in gdf.columns if c not in (h3_col, "geometry")][0]
        )
        gdf[cat_col] = gdf[cat_col].astype(str)

        layers_cache[feat] = {"gdf": gdf, "cat_col": cat_col}

    print("All input layers loaded successfully.\n" + "-" * 50)

    summary_records = []

    # 2. Iterate through each requested pair
    for idx, (feat1, feat2) in enumerate(pairs_list, start=1):
        layer_name = f"pair_{feat1}__vs__{feat2}"
        print(f"[{idx}/{len(pairs_list)}] Processing: {feat1} <-> {feat2}")

        gdf1 = layers_cache[feat1]["gdf"]
        col1 = layers_cache[feat1]["cat_col"]

        gdf2 = layers_cache[feat2]["gdf"]
        col2 = layers_cache[feat2]["cat_col"]

        # Inner join on matching H3 cell index
        merged_gdf = gdf1[[h3_col, col1, "geometry"]].merge(
            gdf2[[h3_col, col2]], on=h3_col, how="inner"
        )

        # Create combined category pair string
        merged_gdf["cat_pair"] = (
            merged_gdf[col1].astype(str) + " | " + merged_gdf[col2].astype(str)
        )

        # Compute pair frequencies & grid percentage
        pair_counts = (
            merged_gdf["cat_pair"].value_counts().reset_index()
        )
        pair_counts.columns = ["cat_pair", "pair_frequency"]
        pair_counts["percentage"] = (
            pair_counts["pair_frequency"] / len(merged_gdf)
        ) * 100

        # Merge frequency metrics back to spatial cells
        merged_gdf = merged_gdf.merge(pair_counts, on="cat_pair", how="left")

        # Sort so rare/unusual combinations draw on top when visualized
        merged_gdf = merged_gdf.sort_values(
            by="pair_frequency", ascending=True
        ).reset_index(drop=True)

        # Save to output GeoPackage as a distinct layer
        merged_gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")

        # Track summary metadata
        summary_records.append({
            "feature_1": feat1,
            "feature_2": feat2,
            "layer_name": layer_name,
            "total_h3_cells": len(merged_gdf),
            "unique_pair_combos": len(pair_counts),
            "top_pair": pair_counts.iloc[0]["cat_pair"],
            "top_pair_pct": pair_counts.iloc[0]["percentage"],
        })

    # 3. Print overall summary table
    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 60)
    print("ALL PAIRS PROCESSED SUCCESSFULLY")
    print(f"Output saved to: {output_gpkg_path.resolve()}")
    print("=" * 60 + "\n")

    print(
        summary_df[
            [
                "layer_name",
                "total_h3_cells",
                "unique_pair_combos",
                "top_pair",
                "top_pair_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "total_h3_cells": "{:,}".format,
                "top_pair_pct": "{:.2f}%".format,
            },
        )
    )

    return summary_df


if __name__ == "__main__":
    # Cleaned list of feature pairings (without Cramer's V values)
    PAIRS_TO_PROCESS = [
        ("surface_stoniness_class", "rooting_depth_class"),
        ("surface_stoniness_class", "usda_hydrologic_group"),
        ("landslide_surface_class", "rooting_depth_class"),
        ("landslide_surface_class", "surface_stoniness_class"),
        ("rooting_depth_class", "usda_hydrologic_group"),
        ("landslide_surface_class", "usda_hydrologic_group"),
        ("landslide_surface_class", "hydraulic_hazard_level"),
        ("pai_landslide_hazard_level", "landslide_surface_class"),
    ]

    INPUT_GPKG = GRID_PATH / "h310_grid_final_datacube.gpkg"
    OUTPUT_GPKG = "../output/h3_pair_heatmaps.gpkg"
    H3_INDEX_COLUMN = "index"

    summary_results = process_all_h3_pairs(
        gpkg_path=INPUT_GPKG,
        pairs_list=PAIRS_TO_PROCESS,
        output_gpkg_path=OUTPUT_GPKG,
        h3_col=H3_INDEX_COLUMN,
    )